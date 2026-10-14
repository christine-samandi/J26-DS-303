"""The C1 pipeline end to end on mock C2/C3/C4, through the LangGraph wiring."""

import ast
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

import orchestrator
from orchestrator.clients import MOCK_DIR, MockAssuranceClient, MockOcrClient, MockRetrievalClient
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
    assert result.status == "failed" and result.calc is None
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
