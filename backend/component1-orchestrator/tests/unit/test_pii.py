"""PII redact-restore: identifiers are replaced, everything the post-check reads is not."""

import json
from decimal import Decimal as D

import pytest
from hypothesis import given
from hypothesis import strategies as st

from explanation_agent.explainer import build_prompt, explain, template_lines
from explanation_agent.post_check import TAG, numbers_in
from orchestrator.clients import MockAssuranceClient, MockOcrClient, MockRetrievalClient
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps
from pii import redact, restore
from tax_calculation_agent.engine import calculate
from tax_calculation_agent.ruleset import load_seed_rules

NAME = "Nimal Perera"
PII = {
    "NAME": NAME,
    "NIC": "199012345678",
    "EMP_NO": "EMP-0042",
    "EPF_NO": "A/12345",
    "EMAIL": "nimal.perera@example.com",
    "PHONE": "+94 77 123 4567",
}
PLANTED = (
    f"Payslip of {NAME}, NIC {PII['NIC']}, employee {PII['EMP_NO']}, "
    f"EPF No: {PII['EPF_NO']}, email {PII['EMAIL']}, phone {PII['PHONE']}."
)


def test_planted_pii_is_replaced_by_placeholders():
    r = redact(PLANTED, known_names=[NAME])
    for value in PII.values():
        assert value not in r.text
    assert r.text == (
        "Payslip of [NAME_1], NIC [NIC_1], employee [EMP_NO_1], "
        "EPF No: [EPF_NO_1], email [EMAIL_1], phone [PHONE_1]."
    )
    assert r.count == 6


@pytest.mark.parametrize(
    "text, kind",
    [
        ("NIC 901234567V", "NIC"),
        ("NIC 901234567x", "NIC"),
        ("call 0771234567", "PHONE"),
        ("call 011-234-5678", "PHONE"),
        ("call 0094 77 123 4567", "PHONE"),
        ("EPF Number 98765", "EPF_NO"),
        ("staff no emp0042", "EMP_NO"),
        ("Dear NIMAL   perera,", "NAME"),
    ],
)
def test_each_kind_of_identifier(text, kind):
    r = redact(text, known_names=[NAME])
    assert list(r.mapping) == [f"[{kind}_1]"]
    assert restore(r.text, r.mapping) == text


def test_same_value_same_placeholder():
    r = redact(f"{PII['NIC']} and again {PII['NIC']}; other 200012345678")
    assert r.text == "[NIC_1] and again [NIC_1]; other [NIC_2]"


def test_round_trip():
    r = redact(PLANTED, known_names=[NAME])
    assert restore(r.text, r.mapping) == PLANTED


def test_restore_after_llm_rewording():
    r = redact(PLANTED, known_names=[NAME])
    reply = "Hello [NAME_1]. Your NIC is [NIC_1]; we will email [EMAIL_1]. [UNKNOWN_9] stays."
    assert restore(reply, r.mapping) == (
        f"Hello {NAME}. Your NIC is {PII['NIC']}; we will email {PII['EMAIL']}. [UNKNOWN_9] stays."
    )


FRAGMENTS = [
    *PII.values(),
    "Rs. 1,800,000.00",
    "Rs. 123456789012.50",
    "6%",
    "YA 2025/26",
    "[SEED-PERSONAL_RELIEF-YA2025_26]",
    "EPF employee contribution: Rs. 120,000.00",
    "EPF No:",
]


@given(st.lists(st.sampled_from(FRAGMENTS), max_size=12), st.sampled_from([" ", ", ", "\n"]))
def test_round_trip_property(parts, sep):
    text = sep.join(parts)
    r = redact(text, known_names=[NAME])
    assert restore(r.text, r.mapping) == text


@pytest.mark.parametrize(
    "text",
    [
        "Rs. 1,800,000.00",
        "Rs. 123456789012.50",  # 12 digits, but part of a larger number
        "Rs. 1,234,567,890.00",
        "Tax at 18% on taxable income from 1000000 to 1500000",
        "EPF employee contribution: Rs. 120,000.00",
        "YA 2025/26, also written 2025-2026",
        "[SEED-PERSONAL_RELIEF-YA2025_26] [IRA-S52-0123456789012]",
    ],
)
def test_figures_citations_and_year_are_not_redacted(text):
    assert redact(text, known_names=[NAME]).text == text


@pytest.mark.parametrize("ya", ["2025/26", "2024/25"])
@pytest.mark.parametrize("income, apit", [(3000000, 50000), (1500000, 0), (9876543, 1234567)])
def test_real_prompts_pass_through_unchanged(ya, income, apit):
    """The prompt the explainer sends today: engine figures, rule tags, the YA; no PII."""
    rules = load_seed_rules(ya)
    steps = calculate(D(income), D(apit), rules).steps
    refs = {r.rule_id: r.citation for r in _refs(rules)}
    prompt = f"Year of assessment {ya}.\n" + build_prompt(steps, template_lines(steps, refs))

    r = redact(prompt, known_names=[NAME])
    assert r.text == prompt and r.count == 0
    assert numbers_in(r.text) == numbers_in(prompt)
    assert TAG.findall(r.text) == TAG.findall(prompt)


def test_pii_next_to_figures_leaves_the_figures_alone():
    line = f"{PLANTED} Relief Rs. 1,800,000.00 [SEED-PERSONAL_RELIEF-YA2025_26] for YA 2025/26."
    r = redact(line, known_names=[NAME])
    tail = " Relief Rs. 1,800,000.00 [SEED-PERSONAL_RELIEF-YA2025_26] for YA 2025/26."
    assert r.text.endswith(tail)
    assert TAG.sub(" ", r.text).count("1,800,000.00") == 1


def test_no_names_and_blank_names_are_ignored():
    assert redact("Nimal Perera", known_names=["", "  "]).text == "Nimal Perera"
    assert restore("unchanged [NAME_1]", {}) == "unchanged [NAME_1]"


# --- around the explainer's LLM call -------------------------------------------------------

RULES = load_seed_rules("2025/26")
RULES_USED = {r.rule_id: r.citation for r in [RULES.personal_relief.ref, RULES.tax_bands.ref]}
STEPS = calculate(D(3000000), D(50000), RULES).steps  # GS-001


class EchoLLM:
    """Records every prompt and answers with the step lines it was given, unchanged: an
    honest rewrite that keeps the placeholders, as the prompt asks."""

    model = "fake-model"
    temperature = 0.0

    def __init__(self, fail: bool = False):
        self.prompts: list[str] = []
        self.fail = fail

    def generate(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        if self.fail:
            raise TimeoutError
        steps = json.loads(prompt.split("Steps:\n", 1)[1].rsplit("\n\nAnswer with", 1)[0])
        return json.dumps({"lines": steps}), "fake-model-001"


def planted_steps():
    """GS-001 with every identifier planted in the first step's description."""
    first = STEPS[0].model_copy(update={"description": PLANTED.removesuffix(".")})
    return (first, *STEPS[1:])


def test_planted_pii_never_reaches_the_llm_or_the_trace():
    llm = EchoLLM()
    outcome = explain(planted_steps(), RULES_USED, llm, "2025/26", known_names=[NAME])

    (sent,) = llm.prompts
    for value in PII.values():
        assert value not in sent
        assert value not in outcome.llm_call.output  # what the trace keeps
    assert "[NIC_1]" in sent and "[NAME_1]" in sent
    assert outcome.pii_redacted == 6
    # restored before the post-check, which then passes as it would without redaction
    assert outcome.explanation.rewrite_status == "accepted"
    assert outcome.explanation.lines[0].startswith(PLANTED.removesuffix("."))


def test_post_check_sees_the_restored_text():
    """Same steps, no LLM vs. the echo LLM: identical lines, so redaction changed nothing."""
    plain = explain(planted_steps(), RULES_USED, None, "2025/26")
    echoed = explain(planted_steps(), RULES_USED, EchoLLM(), "2025/26", known_names=[NAME])
    assert echoed.explanation.lines == plain.explanation.lines


def test_a_failed_call_still_reports_what_was_redacted():
    outcome = explain(planted_steps(), RULES_USED, EchoLLM(fail=True), known_names=[NAME])
    assert outcome.explanation.rewrite_status == "failed"
    assert outcome.pii_redacted == 6


def test_without_an_llm_nothing_is_redacted():
    assert explain(STEPS, RULES_USED).pii_redacted == 0


def test_the_pipeline_reports_the_count_and_sends_no_pii():
    llm = EchoLLM()
    d = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient(), explainer_llm=llm)
    request = TaxRequest(
        request_id="req-0001",
        year_of_assessment="2025/26",
        employment_income=D(3000000),
        apit_paid=D(50000),
    )
    result = run(request, d)
    assert result.explanation.rewrite_status == "accepted"
    assert result.pii_redacted == 0  # today's prompt holds engine figures only
    assert len(llm.prompts) == 1


def _refs(rules):
    return [rules.personal_relief.ref, rules.tax_bands.ref]
