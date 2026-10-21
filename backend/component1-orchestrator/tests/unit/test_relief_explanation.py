"""The solar scenario through the explanation post-check and into C4's payload.

Solar invoice Rs. 800,000, confirmed by the user, YA 2025/26 (figures as in
test_relief_discovery): relief min(600,000, 800,000) = 600,000 -> tax 36,000, refund 14,000.
The rewrite must keep every figure, the solar citation and the fact that the amount was
confirmed by the user; C4 is told where the amount came from, by neutral document id only.
"""

import json
from datetime import date
from decimal import Decimal as D

import pytest

from explanation_agent.demo_tamper import TamperedRewriteLLM
from explanation_agent.explainer import template_lines
from orchestrator.clients import (
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
    c4_payload,
)
from orchestrator.graph import run
from orchestrator.models import DocumentInput, TaxRequest
from orchestrator.stages import Deps

SOLAR = "RULE-SOLAR_RELIEF-V1"
YES = {"resident": "yes", "on_premises": "yes", "national_grid": "yes"}


def solar_request(*inputs, documents=("solar_invoice_2025",)):
    inputs = inputs or (
        DocumentInput(
            document="doc-1",
            amount_paid=D("800000.00"),
            document_date=date(2025, 8, 15),
            answers=YES,
        ),
    )
    return TaxRequest(
        request_id="req-0001",
        year_of_assessment="2025/26",
        employment_income=D(3000000),
        apit_paid=D(50000),
        documents=documents,
        document_inputs=inputs,
    )


class RewriteLLM:
    """Echoes the template lines it was sent, with some lines replaced."""

    model = "fake-gemini"
    temperature = 0.0

    def __init__(self, **changes):
        self.changes = changes
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        steps = json.loads(prompt.split("Steps:\n", 1)[1].rsplit("\n\nAnswer with", 1)[0])
        lines = [{**s, "text": self.changes.get(s["step_id"], s["text"])} for s in steps]
        return json.dumps({"lines": lines}), "fake-gemini-001"


def solve(llm=None, request=None):
    deps = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient(), explainer_llm=llm)
    return run(request or solar_request(), deps), deps


def solar_line(result) -> str:
    (i,) = [
        i for i, c in enumerate(result.explanation.claims) if c.step_id == "S1_SOLAR_PANEL_RELIEF"
    ]
    return result.explanation.lines[i]


# --- wording ---------------------------------------------------------------------------------


def test_template_states_limit_amount_paid_and_that_the_user_confirmed_it():
    result, _ = solve()
    line = solar_line(result)
    assert line.startswith(
        "Solar panel relief: the lower of the yearly limit and the amount paid (from the "
        "uploaded invoice, amount confirmed by the user): the yearly limit is Rs. 600,000.00 "
        "and the amount paid is Rs. 800,000.00: Rs. 600,000.00 ["
    )
    assert "No. 10 of 2021" in line  # the solar tag rendered as C3's citation
    assert result.explanation.rewrite_status == "not_requested"


def test_only_a_capped_relief_line_gets_the_two_figures():
    result, _ = solve()
    rules = {SOLAR: "x", **{r: "x" for s in result.calc.steps for r in s.citation_refs}}
    ids = [s.step_id for s in result.calc.steps]
    lines = dict(zip(ids, template_lines(result.calc.steps, rules), strict=True))
    assert lines["S1_RELIEF"].count("Rs.") == 1
    assert lines["S1_SOLAR_PANEL_RELIEF"].count("Rs.") == 3


def test_prompt_has_the_figures_and_no_issuer_or_file_name():
    llm = RewriteLLM()
    solve(llm)
    (prompt,) = llm.prompts
    assert "amount confirmed by the user" in prompt and "800,000.00" in prompt
    assert f"[{SOLAR}]" in prompt
    for leak in ("Sunshine", "solar_invoice_2025"):
        assert leak not in prompt


# --- post-check on the solar line --------------------------------------------------------


HONEST = (
    "You paid Rs. 800,000.00 for solar panels, an amount you confirmed. The relief is capped "
    f"at Rs. 600,000.00 a year, so you get Rs. 600,000.00 [{SOLAR}]"
)


def test_honest_solar_rewrite_is_accepted():
    result, _ = solve(RewriteLLM(S1_SOLAR_PANEL_RELIEF=HONEST))
    assert result.explanation.rewrite_status == "accepted"
    assert "an amount you confirmed" in solar_line(result)
    assert result.calc.total_tax == D(36000)


@pytest.mark.parametrize(
    "text, problem",
    [
        (
            "You paid Rs. 800,000.00; the relief is capped at Rs. 600,000.00: "
            f"Rs. 600,000.00 [{SOLAR}]",
            "drops that the amount was confirmed by the user",
        ),
        (
            "We read Rs. 800,000.00 from your invoice by OCR, you confirmed it; capped at "
            f"Rs. 600,000.00: Rs. 600,000.00 [{SOLAR}]",
            "says an amount was read from a document",
        ),
        (
            f"You confirmed Rs. 600,000.00 against the cap, so you get Rs. 800,000.00 [{SOLAR}]",
            "states 800000.00 as the result",
        ),
        (
            "You confirmed Rs. 800,000.00; Rs. 200,000 carries to later years, so you get "
            f"Rs. 600,000.00 [{SOLAR}]",
            "200000 is not one of the engine's figures",
        ),
        (
            "You confirmed Rs. 800,000.00, capped at Rs. 600,000.00: Rs. 600,000.00",
            "citations [] != ['RULE-SOLAR_RELIEF-V1']",
        ),
        (
            "You confirmed Rs. 800,000.00, capped at Rs. 600,000.00 (Act of 2017-18): "
            f"Rs. 600,000.00 [{SOLAR}]",
            "mentions YA 2017-18",
        ),
    ],
    ids=["no confirm", "claims OCR", "wrong result", "invented carry-over", "no citation", "year"],
)
def test_cheating_solar_rewrite_falls_back_to_the_template(text, problem):
    result, _ = solve(RewriteLLM(S1_SOLAR_PANEL_RELIEF=text))
    exp = result.explanation
    assert (exp.rewrite_status, exp.used_fallback, exp.hallucination_flag) == (
        "rejected",
        True,
        True,
    )
    assert any(problem in w for w in result.warnings), result.warnings
    assert "amount confirmed by the user" in solar_line(result)  # the template is shown
    assert result.calc.total_tax == D(36000)  # figures are the engine's either way


def test_taxable_income_line_that_drops_the_solar_citation_is_rejected():
    s2 = (
        "Your income of Rs. 3,000,000.00 less reliefs Rs. 1,800,000.00 and Rs. 600,000.00 "
        "leaves Rs. 600,000.00 [RULE-PERSONAL_RELIEF-V4]"
    )
    result, _ = solve(RewriteLLM(S2_TAXABLE_INCOME=s2))
    assert result.explanation.rewrite_status == "rejected"


def test_demo_tamper_is_rejected_offline_on_the_solar_scenario():
    llm = TamperedRewriteLLM()
    result, _ = solve(llm)
    assert llm.last is not None and llm.last.figure_step == "S6_REFUND"
    assert result.explanation.rewrite_status == "rejected"
    assert result.explanation.hallucination_flag
    assert "amount confirmed by the user" in solar_line(result)


def test_same_solar_run_gives_an_identical_explanation():
    first = solve()[0].explanation
    assert all(solve()[0].explanation == first for _ in range(3))


# --- C4 payload ------------------------------------------------------------------------------


def payload(request=None):
    _, deps = solve(request=request)
    (trace,) = deps.assurance.traces
    return trace, *c4_payload(trace)


def test_c4_gets_the_solar_rule_and_the_confirmed_amount_as_sources():
    trace, path, body = payload()
    assert path == "/assurance/combined-check"
    (relief,) = trace.reliefs
    assert (relief.base, relief.base_source, relief.document_ids) == (
        D("800000.00"),
        "user",
        ("doc-1",),
    )
    sources = {s["source_id"]: s["text"] for s in body["sources"]}
    assert sources["doc-1:amount_paid"] == (
        "Amount paid 800000.00 on the invoice doc-1, confirmed by the user"
    )
    assert "national grid" in sources[SOLAR]  # C3's evidence text
    (cl,) = [c for c in trace.claims if c.step_id == "S1_SOLAR_PANEL_RELIEF"]
    claims = {c["claim_id"]: c for c in body["claims"]}
    assert claims[f"{cl.claim_id}/doc-1:amount_paid"]["excerpt"] == sources["doc-1:amount_paid"]
    assert claims[f"{cl.claim_id}/{SOLAR}"]["source_id"] == SOLAR
    assert body["reported_total"] == "36000.00"


def test_c4_payload_fits_c4s_request_models_and_names_no_file_or_issuer():
    _, _, body = payload()
    assert set(body) == {
        "reported_total",
        "line_items",
        "regeneration_attempts",
        "claims",
        "sources",
    }
    assert all(set(c) == {"claim_id", "source_id", "excerpt"} for c in body["claims"])
    assert all(set(s) == {"source_id", "text"} for s in body["sources"])
    cited = {c["source_id"] for c in body["claims"]}
    assert cited == {s["source_id"] for s in body["sources"]}
    text = json.dumps(body)
    for leak in ("solar_invoice_2025", "Sunshine"):
        assert leak not in text


def test_two_invoices_give_c4_one_combined_amount_source():
    def doc(n):
        return DocumentInput(
            document=f"doc-{n}",
            amount_paid=D("400000.00"),
            document_date=date(2025, 8, 15),
            answers=YES,
        )

    request = solar_request(doc(1), doc(2), documents=("solar_invoice_2025", "solar_invoice_2025"))
    _, _, body = payload(request)
    sources = {s["source_id"]: s["text"] for s in body["sources"]}
    assert sources["doc-1+doc-2:amount_paid"] == (
        "Amount paid 800000.00 on the invoice doc-1, doc-2, confirmed by the user"
    )


def test_without_reliefs_the_c4_payload_is_unchanged():
    request = solar_request(documents=())
    _, path, body = payload(request.model_copy(update={"document_inputs": ()}))
    assert not any(s["source_id"].endswith(":amount_paid") for s in body.get("sources", []))
    assert path == "/assurance/combined-check"
