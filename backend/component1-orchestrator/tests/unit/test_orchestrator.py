"""The C1 pipeline end to end on mock C2/C3/C4, through the LangGraph wiring."""

import ast
import json
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

import orchestrator
from orchestrator.clients import (
    MOCK_DIR,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
    mock_retrieval_file,
)
from orchestrator.confidence import c1_confidence
from orchestrator.graph import build_graph, run
from orchestrator.models import AssuranceVerdict, FailedClaim, TaxRequest
from orchestrator.stages import Deps

LINEAR = ["extract", "normalise", "retrieve", "calculate", "explain", "score", "assure"]


def request(**kw) -> TaxRequest:
    return TaxRequest(**{"request_id": "req-0001", "year_of_assessment": "2025/26", **kw})


def typed(**kw) -> TaxRequest:  # GS-001: 3,000,000 income, 50,000 APIT -> 96,000 tax
    return request(employment_income=D(3000000), apit_paid=D(50000), **kw)


def deps(script=(), retrieval_path=MOCK_DIR / "retrieval_ya2025_26.json") -> Deps:
    return Deps(MockOcrClient(), MockRetrievalClient(retrieval_path), MockAssuranceClient(script))


def test_typed_input_is_delivered_with_engine_figures():
    result = run(typed(), deps())
    assert result.status == "delivered" and result.review_reasons == ()
    assert (result.calc.taxable_income, result.calc.total_tax) == (D(1200000), D(96000))
    assert (result.calc.tax_payable, result.calc.refund) == (D(46000), D(0))
    assert result.attempt == 1
    assert list(result.stages_run) == [*LINEAR, "deliver"]
    assert result.confidence.score == D("0.9")  # lowest C3 rule confidence


def test_clean_payslip_goes_through_c2():
    d = deps()
    result = run(request(documents=("payslip_clean",)), d)
    assert result.status == "delivered"
    assert result.calc.total_tax == D(96000)
    assert d.assurance.traces[0].profile.employment_income.source == "ocr"


def test_low_ocr_confidence_is_escalated_even_when_c4_passes():
    result = run(request(documents=("payslip_noisy_low_confidence",)), deps())
    assert result.status == "needs_review"
    assert result.confidence.ocr == D("0.55")
    assert result.review_reasons == ("C1 confidence 0.55 is below the review threshold 0.7",)
    assert result.calc.total_tax == D(96000)  # the figures are still shown, flagged


def test_regenerate_reruns_from_the_named_stage_then_passes():
    verdict = AssuranceVerdict(
        request_id="req-0001",
        attempt=1,
        decision="REGENERATE",
        rerun_from="retrieve",
        failed_claims=(FailedClaim(step_id="S3_BAND_2", reason="wrong band rate"),),
    )
    d = deps(script=[verdict, "PASS"])
    result = run(typed(), d)
    assert result.status == "delivered" and result.attempt == 2
    assert [t.attempt for t in d.assurance.traces] == [1, 2]
    assert [v.decision for v in result.verdicts] == ["REGENERATE", "PASS"]
    assert list(result.stages_run) == [*LINEAR, "regenerate", *LINEAR[2:], "deliver"]
    assert len(d.retrieval.requests) == 2


def test_second_regenerate_is_not_retried():
    d = deps(script=["REGENERATE", "REGENERATE"])
    result = run(typed(), d)
    assert result.status == "needs_review" and result.attempt == 2
    assert len(d.assurance.traces) == 2
    assert result.review_reasons[0] == "C4 still found problems after regenerating once"


def test_human_review_from_c4():
    result = run(typed(), deps(script=["HUMAN_REVIEW"]))
    assert result.status == "needs_review"
    assert result.review_reasons == ("C4 asked for human review", "mock C4 scripted HUMAN_REVIEW")


def test_seed_and_c3_conflict_is_surfaced_and_penalised(tmp_path):
    data = json.loads((MOCK_DIR / "retrieval_ya2025_26.json").read_text(encoding="utf-8"))
    data["results"][0]["values"]["amount"]["amount"] = 1200000
    path = tmp_path / "c3.json"
    path.write_text(json.dumps(data))
    result = run(typed(), deps(retrieval_path=path))
    assert result.status == "needs_review"
    assert result.conflicts == ("personal relief: seed 1800000, c3 1200000",)
    assert result.confidence.conflict_penalty == D("0.6")
    assert "seed rules and C3 disagree on personal relief: seed 1800000, c3 1200000" in (
        result.review_reasons
    )


def test_no_match_from_c3_fails_cleanly():
    path = (
        Path(orchestrator.__file__).resolve().parents[3]
        / "contracts/mock_data/retrieval_output_examples/04_no_match.json"
    )
    d = deps(retrieval_path=path)
    result = run(typed(), d)
    assert result.status == "failed" and result.calc is None and result.rules is None
    assert result.review_reasons[0].startswith("retrieve: rules for YA 2025/26 could not be")
    assert list(result.stages_run) == ["extract", "normalise", "retrieve", "deliver"]
    assert d.assurance.traces == []


def test_missing_income_fails_before_calling_c3():
    d = deps()
    result = run(request(), d)
    assert result.status == "failed"
    assert result.review_reasons == (
        "normalise: no employment income: type it in or upload a payslip",
    )
    assert d.retrieval.requests == []


def test_missing_apit_is_taken_as_zero_with_a_warning():
    result = run(request(employment_income=D(3000000)), deps())
    assert result.calc.tax_payable == D(96000)
    assert result.warnings == ("APIT paid was not given, so it is taken as 0",)


def test_c3_is_asked_for_the_rules_at_the_start_of_the_ya():
    d = deps()
    run(typed(), d)
    (asked,) = d.retrieval.requests
    assert str(asked.as_of_date) == "2025-04-01"
    assert asked.request_id == "req-0001" and asked.caller == "c1-orchestrator"


def test_every_step_has_a_claim_and_citations_come_from_c3():
    result = run(typed(), deps())
    steps = [s.step_id for s in result.calc.steps]
    assert [c.step_id for c in result.explanation.claims] == steps
    cited = {ref for c in result.explanation.claims for ref in c.citation_refs}
    assert cited == {"RULE-PERSONAL_RELIEF-V4", "RULE-INDIVIDUAL_SLABS-V4"}


def test_same_request_gives_identical_result():
    first = run(typed(), deps()).model_dump_json()
    assert all(run(typed(), deps()).model_dump_json() == first for _ in range(20))


def test_graph_can_be_drawn_for_the_slides():
    mermaid = build_graph(deps()).get_graph().draw_mermaid()
    for node in [*LINEAR, "regenerate", "deliver"]:
        assert node in mermaid


def test_confidence_formula():
    b = c1_confidence(
        ocr=[D("0.97"), D("0.95")],
        retrieval=[D("0.93"), D("0.9")],
        interpretation=[],
        explanation_fell_back=True,
        has_conflict=True,
    )
    assert (b.ocr, b.retrieval, b.interpretation, b.explanation) == (
        D("0.95"),
        D("0.9"),
        D(1),
        D("0.5"),
    )
    assert b.score == D("0.30")


def test_only_graph_module_imports_langgraph():
    package = Path(orchestrator.__file__).parent
    importers = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            if any(n.split(".")[0] == "langgraph" for n in names):
                importers.append(path.name)
    assert importers == ["graph.py"]


@pytest.mark.parametrize("bad_id", ["0001", "REQ-1", "req-"])
def test_request_id_must_be_req_prefixed(bad_id):
    with pytest.raises(ValueError):
        request(request_id=bad_id)


# --- explanation LLM (fake: no real Gemini calls) ---------------------------------------------


class FakeExplainer:
    """Rewords whatever steps the prompt holds; `cheat` edits a step's text first."""

    model = "fake-explainer"
    temperature = 0.0

    def __init__(self, cheat=None, error=None):
        self.cheat = cheat or {}
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        steps = json.loads(prompt.split("Steps:\n", 1)[1].split("\n\nAnswer with JSON", 1)[0])
        reworded = {s["step_id"]: f"In short: {s['text']}" for s in steps}
        lines = [{"step_id": k, "text": self.cheat.get(k, v)} for k, v in reworded.items()]
        return json.dumps({"lines": lines}), "fake-explainer-001"


def llm_deps(llm, script=()) -> Deps:
    d = deps(script)
    d.explainer_llm = llm
    return d


def test_every_cited_rule_is_in_the_result_with_its_evidence():
    result = run(typed(), deps())
    refs = {r.rule_id: r for r in (result.rules.personal_relief.ref, result.rules.tax_bands.ref)}
    cited = {ref for s in result.calc.steps for ref in s.citation_refs}
    assert cited and cited <= refs.keys()
    assert all(refs[ref].evidence and refs[ref].evidence_chunk_ids for ref in cited)


def test_honest_explainer_rewrite_is_delivered_and_logged_in_the_trace():
    d = llm_deps(FakeExplainer())
    result = run(typed(), d)
    assert result.status == "delivered" and result.confidence.explanation == D(1)
    assert all(line.startswith("In short: ") for line in result.explanation.lines)
    assert (
        "[Inland Revenue (Amendment) Act No. 2 of 2025 [C1 MOCK], s.5, inserting item (v) in "
        "Fifth Schedule para 2(a) of Act No. 24 of 2017]" in (result.explanation.lines[0])
    )
    (trace,) = d.assurance.traces
    (call,) = trace.llm_calls
    assert (call.stage, call.model, call.prompt_version, call.temperature) == (
        "explain",
        "fake-explainer-001",
        "explain-v2",
        0,
    )
    assert trace.explanation == result.explanation
    assert [c.text for c in trace.claims] == list(result.explanation.lines)


def test_cheating_explainer_is_caught_flagged_and_sent_for_review():
    cheat = "Rs. 3,000,000 less Rs. 1,800,000 leaves Rs. 1,100,000 [RULE-PERSONAL_RELIEF-V4]."
    d = llm_deps(FakeExplainer(cheat={"S2_TAXABLE_INCOME": cheat}))
    result = run(typed(), d)
    e = result.explanation
    assert e.used_fallback and e.hallucination_flag
    assert "1,100,000" not in " ".join(e.lines)
    assert e.lines[1].startswith("Taxable income: employment income less personal relief")
    assert result.calc.taxable_income == D(1200000)  # the engine's figure is untouched
    assert result.confidence.explanation == D("0.5")
    assert result.status == "needs_review"
    assert any("S2_TAXABLE_INCOME: 1100000 is not one" in w for w in result.warnings)
    (trace,) = d.assurance.traces
    assert trace.explanation.hallucination_flag
    assert cheat in trace.llm_calls[0].output  # the rejected reply is kept for C4 and audit


def test_explainer_citing_an_unretrieved_rule_is_caught():
    cheat = "Everyone gets Rs. 1,800,000 tax-free [RULE-PERSONAL_RELIEF-V3]."  # superseded
    result = run(typed(), llm_deps(FakeExplainer(cheat={"S1_RELIEF": cheat})))
    assert result.explanation.hallucination_flag
    assert any("cites 'RULE-PERSONAL_RELIEF-V3'" in w for w in result.warnings)


def test_explainer_outage_shows_the_template_without_review():
    d = llm_deps(FakeExplainer(error=ConnectionError("no route")))
    result = run(typed(), d)
    assert result.status == "delivered" and result.confidence.explanation == D(1)
    assert not result.explanation.used_fallback
    assert d.assurance.traces[0].llm_calls == ()
    assert result.warnings == (
        "explanation LLM call failed (ConnectionError); template text shown",
    )


def test_regenerate_from_explain_calls_the_llm_again_and_traces_each_attempt():
    verdict = AssuranceVerdict(
        request_id="req-0001", attempt=1, decision="REGENERATE", rerun_from="explain"
    )
    llm = FakeExplainer(cheat={"S6_PAYABLE": "You owe Rs. 45,000."})
    d = llm_deps(llm, script=[verdict, "PASS"])
    result = run(typed(), d)
    assert len(llm.prompts) == 2
    assert [len(t.llm_calls) for t in d.assurance.traces] == [1, 1]
    assert result.attempt == 2 and result.explanation.hallucination_flag


def test_without_an_explainer_the_trace_has_no_llm_calls():
    d = deps()
    run(typed(), d)
    assert d.assurance.traces[0].llm_calls == ()
    assert not d.assurance.traces[0].explanation.used_fallback


# --- C1 mock C3 per year of assessment ---------------------------------------------------


@pytest.mark.parametrize(
    ("day", "name"),
    [
        (date(2024, 4, 1), "retrieval_ya2024_25.json"),
        (date(2025, 3, 31), "retrieval_ya2024_25.json"),
        (date(2025, 4, 1), "retrieval_ya2025_26.json"),
    ],
)
def test_mock_c3_file_is_chosen_by_year_of_assessment(day, name):
    assert mock_retrieval_file(day) == MOCK_DIR / name


def test_ya_2024_25_uses_that_years_law_and_matches_its_seed():
    """GS-012's inputs: the mock C3 answer for 2024/25 gives last year's tax, cross-checked."""
    result = run(typed(year_of_assessment="2024/25"), Deps(
        MockOcrClient(), MockRetrievalClient(), MockAssuranceClient()
    ))  # fmt: skip
    assert result.status == "delivered"
    assert (result.calc.taxable_income, result.calc.total_tax, result.calc.tax_payable) == (
        D(1800000),
        D(252000),
        D(202000),
    )
    assert result.conflicts == ()
    assert not [w for w in result.warnings if "not cross-checked" in w]
    assert "Act No. 45 of 2022 [C1 MOCK], s.36(2)(a)" in result.rules.personal_relief.ref.citation
    assert "Act No. 45 of 2022 [C1 MOCK], s.33(1)" in result.rules.tax_bands.ref.citation


def test_year_without_a_mock_file_fails_with_the_reason():
    result = run(typed(year_of_assessment="2023/24"), Deps(
        MockOcrClient(), MockRetrievalClient(), MockAssuranceClient()
    ))  # fmt: skip
    assert result.status == "failed" and result.calc is None
    assert "C1 mock C3 has no rules for 2023-04-01" in result.review_reasons[0]
