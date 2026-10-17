"""Demo mode: the simulated tampered rewrite is always caught by the real post-check."""

from decimal import Decimal as D

import pytest

from explanation_agent.demo_tamper import MODEL, TamperedRewriteLLM, tamper_lines
from orchestrator.clients import MockAssuranceClient, MockOcrClient, MockRetrievalClient
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps

LINES = [
    ("S4_TOTAL_TAX", "Total tax: Rs. 96,000.00 [RULE-INDIVIDUAL_SLABS-V4]"),
    ("S5_APIT_CREDIT", "APIT already paid: Rs. 50,000"),
    ("S6_PAYABLE", "Tax payable: Rs. 46,000.00"),
]


def typed(income, apit) -> TaxRequest:
    return TaxRequest(
        request_id="req-0001",
        year_of_assessment="2025/26",
        employment_income=D(income),
        apit_paid=D(apit),
    )


def demo_deps() -> Deps:
    d = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient([]))
    d.explainer_llm = TamperedRewriteLLM()
    return d


def test_tamper_changes_one_figure_and_moves_one_citation():
    t = tamper_lines(LINES)
    assert t.lines == (
        ("S4_TOTAL_TAX", "Total tax: Rs. 96,000.00"),
        LINES[1],
        ("S6_PAYABLE", "Tax payable: Rs. 56,000.00 [RULE-INDIVIDUAL_SLABS-V4]"),
    )
    assert (t.figure_step, t.old_figure, t.new_figure) == ("S6_PAYABLE", "46,000.00", "56,000.00")
    assert (t.citation, t.citation_from, t.citation_to) == (
        "RULE-INDIVIDUAL_SLABS-V4",
        "S4_TOTAL_TAX",
        "S6_PAYABLE",
    )


def test_citation_on_the_last_line_is_dropped():
    t = tamper_lines([("S6_PAYABLE", "Tax payable: Rs. 46,000.00 [RULE-X]")])
    assert t.lines == (("S6_PAYABLE", "Tax payable: Rs. 56,000.00"),)


def test_a_figure_inside_a_citation_tag_is_not_the_one_changed():
    t = tamper_lines([("S6", "Due: Rs. 5 [Rs. 9]")])
    assert (t.old_figure, t.lines[-1][1]) == ("5", "Due: Rs. 10,005.00")


@pytest.mark.parametrize(
    "lines, reason",
    [
        ([], "no lines"),
        ([("S1", "Relief [RULE-A]"), ("S6", "Tax payable: nothing")], "no Rs. figure"),
        ([("S1", "Relief: Rs. 1,800,000"), ("S6", "Due: Rs. 46,000.00")], "no citation"),
    ],
)
def test_nothing_to_tamper_with_raises(lines, reason):
    with pytest.raises(ValueError, match=reason):
        tamper_lines(lines)


@pytest.mark.parametrize(
    "income, apit, last_step, figure",
    [
        (3_000_000, 50_000, "S6_PAYABLE", D(46000)),  # GS-001
        (2_000_000, 50_000, "S6_REFUND", D(38000)),  # 12,000 tax, 50,000 APIT
    ],
)
def test_tampered_rewrite_is_rejected_flagged_and_sent_for_review(income, apit, last_step, figure):
    d = demo_deps()
    result = run(typed(income, apit), d)
    tampering = d.explainer_llm.last
    e = result.explanation

    assert tampering.figure_step == last_step
    assert e.used_fallback and e.hallucination_flag
    assert tampering.new_figure not in " ".join(e.lines)  # template shown, not the lie
    assert result.calc.steps[-1].step_id == last_step
    assert result.calc.steps[-1].result == figure  # the engine's figure is untouched
    assert result.confidence.explanation == D("0.5")
    assert result.status == "needs_review"
    bumped = tampering.new_figure.replace(",", "")
    assert any(f"{last_step}: {bumped} is not one" in w for w in result.warnings)
    assert any(f"{tampering.citation_from}: citations" in w for w in result.warnings)

    (trace,) = d.assurance.traces
    (call,) = trace.llm_calls
    assert call.model == MODEL
    assert tampering.new_figure in call.output  # the rejected reply is kept for C4 and audit
    assert trace.explanation.hallucination_flag
