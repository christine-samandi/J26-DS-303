"""Explanation agent with fake LLMs: honest rewrites are used, cheating ones are rejected."""

import ast
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

import explanation_agent
from explanation_agent.explainer import PROMPT_VERSION, explain, template_lines
from explanation_agent.post_check import numbers_in
from tax_calculation_agent.engine import calculate
from tax_calculation_agent.ruleset import load_seed_rules

RULES = load_seed_rules("2025/26")
RELIEF = RULES.personal_relief.ref
BANDS = RULES.tax_bands.ref
RULES_USED = {RELIEF.rule_id: RELIEF.citation, BANDS.rule_id: BANDS.citation}
# GS-001: 3,000,000 income, 50,000 APIT -> taxable 1,200,000, tax 96,000, payable 46,000
STEPS = calculate(D(3000000), D(50000), RULES).steps
IDS = [s.step_id for s in STEPS]

HONEST = {
    "S1_RELIEF": f"Everyone resident gets Rs. 1,800,000 tax-free [{RELIEF.rule_id}].",
    "S2_TAXABLE_INCOME": (
        "Your income of Rs. 3,000,000 minus the Rs. 1,800,000 relief leaves "
        f"Rs. 1,200,000 to be taxed [{RELIEF.rule_id}]."
    ),
    "S3_BAND_1": f"The first Rs. 1,000,000 is taxed at 6%, which is Rs. 60,000 [{BANDS.rule_id}].",
    "S3_BAND_2": f"The next Rs. 200,000 is taxed at 18%: Rs. 36,000 [{BANDS.rule_id}].",
    "S4_TOTAL_TAX": f"Adding the bands, your tax is Rs. 96,000.00 [{BANDS.rule_id}].",
    "S5_APIT_CREDIT": "Your employer already took Rs. 50000 of APIT from your pay.",
    "S6_PAYABLE": "So you still owe Rs. 46,000.",
}


class FakeLLM:
    """Returns a fixed reply (or raises it) and keeps every prompt it was given."""

    model = "fake-model"
    temperature = 0.0

    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, "fake-model-001"


def reply(**changes) -> str:
    """The honest rewrite as JSON, with some lines replaced."""
    lines = {**HONEST, **changes}
    return json.dumps({"lines": [{"step_id": i, "text": lines[i]} for i in IDS]})


def run(text):
    llm = FakeLLM(text)
    return explain(STEPS, RULES_USED, llm), llm


def assert_template_shown(outcome):
    expected = [
        line.replace(f"[{RELIEF.rule_id}]", f"[{RELIEF.citation}]").replace(
            f"[{BANDS.rule_id}]", f"[{BANDS.citation}]"
        )
        for line in template_lines(STEPS, RULES_USED)
    ]
    assert list(outcome.explanation.lines) == expected


# --- no LLM / honest LLM ------------------------------------------------------------------


def test_without_llm_the_template_is_the_explanation():
    outcome = explain(STEPS, RULES_USED)
    assert outcome.llm_call is None and outcome.warnings == ()
    assert outcome.explanation.lines[1] == (
        "Taxable income: employment income less personal relief (not below zero): "
        f"Rs. 1,200,000.00 [{RELIEF.citation}]"
    )
    assert not outcome.explanation.used_fallback
    assert_template_shown(outcome)


def test_honest_rewrite_is_used_with_citations_rendered():
    outcome, llm = run(reply())
    e = outcome.explanation
    assert not e.used_fallback and not e.hallucination_flag and outcome.warnings == ()
    assert e.lines[0] == f"Everyone resident gets Rs. 1,800,000 tax-free [{RELIEF.citation}]."
    assert [c.step_id for c in e.claims] == IDS
    assert [c.text for c in e.claims] == list(e.lines)
    assert e.claims[2].citation_refs == (BANDS.rule_id,)
    assert len(llm.prompts) == 1


def test_llm_call_is_recorded():
    outcome, _ = run(reply())
    call = outcome.llm_call
    assert (call.stage, call.model, call.prompt_version) == (
        "explain",
        "fake-model-001",
        PROMPT_VERSION,
    )
    assert call.temperature == 0 and call.output == reply()


def test_prompt_carries_the_template_lines_and_ids():
    _, llm = run(reply())
    for step_id, line in zip(IDS, template_lines(STEPS, RULES_USED), strict=True):
        assert step_id in llm.prompts[0]
        assert json.dumps(line) in llm.prompts[0]


@pytest.mark.parametrize(
    "text",
    [
        f"Taxable: Rs 1200000, i.e. 3000000 less 1800000 [{RELIEF.rule_id}]",
        f"Rs. 1,200,000.00 is taxed [{RELIEF.rule_id}]",
    ],
)
def test_number_formatting_may_change_but_not_the_value(text):
    outcome, _ = run(reply(S2_TAXABLE_INCOME=text))
    assert not outcome.explanation.used_fallback


def test_fenced_json_is_accepted():
    outcome, _ = run(f"```json\n{reply()}\n```")
    assert not outcome.explanation.used_fallback


# --- cheating LLMs: rejected, template shown, flagged ----------------------------------------

CHEATS = {
    "changed number": (
        "S2_TAXABLE_INCOME",
        f"Rs. 3,000,000 less Rs. 1,800,000 leaves Rs. 1,300,000 [{RELIEF.rule_id}].",
        "1300000 is not one of the engine's figures",
    ),
    "added number": (
        "S4_TOTAL_TAX",
        f"Your tax is Rs. 96,000, about 3.2% of your income [{BANDS.rule_id}].",
        "3.2 is not one of the engine's figures",
    ),
    "number from another step": (
        "S6_PAYABLE",
        "You owe Rs. 46,000 on your Rs. 3,000,000 income.",  # 3,000,000 belongs to S2
        "3000000 is not one of the engine's figures",
    ),
    "dropped result": (
        "S3_BAND_2",
        f"The next part is taxed at 18% [{BANDS.rule_id}].",
        "the result 36000.00 is missing",
    ),
    "number in words": (
        "S2_TAXABLE_INCOME",
        f"Rs. 1,200,000 (1.2 million) is taxed [{RELIEF.rule_id}].",
        "number written in words",
    ),
    "invented citation": (
        "S1_RELIEF",
        f"You get Rs. 1,800,000 tax-free [{RELIEF.rule_id}] [IRA s.52].",
        "cites 'IRA s.52', which is not in rules_used",
    ),
    "dropped citation": (
        "S1_RELIEF",
        "You get Rs. 1,800,000 tax-free.",
        "citations [] != ['SEED-PERSONAL_RELIEF-YA2025_26']",
    ),
    "citation moved to the wrong step": (
        "S3_BAND_1",
        f"The first Rs. 1,000,000 is taxed at 6%: Rs. 60,000 [{RELIEF.rule_id}].",
        "citations ['SEED-PERSONAL_RELIEF-YA2025_26'] != ['SEED-INDIVIDUAL_SLABS-YA2025_26']",
    ),
}


@pytest.mark.parametrize("step_id,text,problem", CHEATS.values(), ids=CHEATS.keys())
def test_cheating_rewrite_falls_back_to_the_template(step_id, text, problem):
    outcome, _ = run(reply(**{step_id: text}))
    e = outcome.explanation
    assert e.used_fallback and e.hallucination_flag
    assert_template_shown(outcome)
    assert any(step_id in w and problem in w for w in outcome.warnings), outcome.warnings
    assert outcome.llm_call.output == reply(**{step_id: text})  # the cheat is kept for audit


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"text": "everything is fine"}),
        json.dumps({"lines": [{"step_id": i, "text": HONEST[i]} for i in IDS[:-1]]}),
        json.dumps({"lines": [{"step_id": i, "text": HONEST[i]} for i in reversed(IDS)]}),
        json.dumps({"lines": [{"step_id": i, "text": ""} for i in IDS]}),
    ],
    ids=["not-json", "no-lines", "missing-step", "reordered", "empty-text"],
)
def test_malformed_reply_falls_back_without_hallucination_flag(bad):
    outcome, _ = run(bad)
    assert outcome.explanation.used_fallback and not outcome.explanation.hallucination_flag
    assert_template_shown(outcome)
    assert len(outcome.warnings) == 1


def test_llm_error_shows_the_template_without_a_penalty():
    outcome, _ = run(TimeoutError("quota exhausted for req-0001"))
    assert outcome.llm_call is None
    assert not outcome.explanation.used_fallback
    assert_template_shown(outcome)
    assert outcome.warnings == ("explanation LLM call failed (TimeoutError); template text shown",)


# --- template, determinism, imports ----------------------------------------------------------


def test_template_refuses_a_citation_outside_rules_used():
    with pytest.raises(ValueError, match="S1_RELIEF cites rules C3 did not return"):
        template_lines(STEPS, {BANDS.rule_id: BANDS.citation})


def test_rule_ids_are_not_read_as_numbers():
    assert numbers_in(f"Rs. 1,800,000 [{RELIEF.rule_id}], 6%.") == [D(1800000), D(6)]


def test_same_reply_gives_identical_outcome():
    cheat = reply(S6_PAYABLE="You owe Rs. 45,000.")
    assert run(cheat)[0] == run(cheat)[0]
    assert run(reply())[0] == run(reply())[0]


def test_explanation_agent_imports_no_llm_sdk():
    banned = {"google", "openai", "anthropic", "langchain", "langgraph"}
    found = []
    for path in Path(explanation_agent.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            found += [f"{path.name}: {n}" for n in names if n.split(".")[0] in banned]
    assert not found, f"LLM SDK imports in explanation_agent: {found}"
