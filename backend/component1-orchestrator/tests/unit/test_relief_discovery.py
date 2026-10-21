"""Multi-document requests and relief discovery, end to end on mock C2/C3/C4.

Invoice/receipt -> C3 is asked with the issuer (+ the user's "what was this for?") -> each
relief C3 returns is applied only if the user confirmed the amount and date and answered
every condition question "yes". Figures checked by hand (YA 2025/26 unless stated):
    3,000,000 income, 800,000 solar invoice: relief min(600,000, 800,000) = 600,000,
    taxable 3,000,000 - 1,800,000 - 600,000 = 600,000, tax 6% = 36,000,
    APIT 50,000 -> refund 14,000; 200,000 remaining claimable in later years.
    Without solar: taxable 1,200,000, tax 60,000 + 200,000 @ 18% = 96,000.
"""

import json
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

from orchestrator.clients import (
    MOCK_DIR,
    ComponentUnavailable,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
)
from orchestrator.graph import run
from orchestrator.models import AssuranceVerdict, DocumentInput, FailedClaim, TaxRequest
from orchestrator.reliefs import relief_query
from orchestrator.stages import Deps
from tax_calculation_agent.ruleset import load_seed_reliefs

INVOICE = "solar_invoice_2025"
YES = {"resident": "yes", "on_premises": "yes", "national_grid": "yes"}
SNAPSHOT = Path(__file__).resolve().parent / "data/no_extra_documents_snapshot.json"


def confirmed(document=INVOICE, amount="800000.00", day=date(2025, 8, 15), answers=YES, **kw):
    return DocumentInput(
        document=document,
        amount_paid=None if amount is None else D(amount),
        document_date=day,
        answers=answers,
        **kw,
    )


def request(documents=(INVOICE,), inputs=None, ya="2025/26", income=3000000, apit=50000):
    return TaxRequest(
        request_id="req-0001",
        year_of_assessment=ya,
        employment_income=None if income is None else D(income),
        apit_paid=D(apit),
        documents=documents,
        document_inputs=(confirmed(),) if inputs is None else inputs,
    )


def deps(ocr_dir=MOCK_DIR / "ocr", retrieval=None, script=(), **kw) -> Deps:
    return Deps(
        MockOcrClient(ocr_dir),
        retrieval or MockRetrievalClient(),
        MockAssuranceClient(script),
        **kw,
    )


def step(result, step_id):
    (match,) = [s for s in result.calc.steps if s.step_id == step_id]
    return match


def finding(result, document=INVOICE):
    (match,) = [f for f in result.relief_findings if f.document_id == document]
    return match


def ocr_dir(tmp_path, **docs) -> Path:
    """A mock C2 directory: the shipped mocks plus `name=OcrResult-json` overrides."""
    for p in (MOCK_DIR / "ocr").glob("*.json"):
        (tmp_path / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    for name, data in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def invoice(issuer="Sunshine Solar Energy (Pvt) Ltd", total="400000.00", kind="invoice"):
    return {
        "document_id": f"doc-{issuer[:4]}",
        "document_type": kind,
        "fields": [],
        "issuer": issuer,
        "document_date": "2025-08-15",
        "total_amount": total,
    }


# --- the brief's scenario -------------------------------------------------------------------


def test_solar_invoice_applies_600000_and_gives_refund_14000():
    d = deps()
    result = run(request(), d)
    assert result.status == "delivered" and result.review_reasons == ()
    assert step(result, "S1_SOLAR_PANEL_RELIEF").result == D(600000)
    assert result.calc.taxable_income == D(600000)
    assert result.calc.total_tax == D(36000)
    assert (result.calc.tax_payable, result.calc.refund) == (D(0), D(14000))
    f = finding(result)
    assert f.status == "applied"
    assert f.rule_id == "RULE-SOLAR_RELIEF-V1"
    assert "paragraph 2(g) of Act No. 24 of 2017" in f.citation
    assert "national grid" in f.rule_text  # C3's conditions, shown as is
    assert f.remaining_claimable == D(200000)
    assert [(q.question_id, q.answer) for q in f.questions] == list(YES.items())


def test_c3_is_asked_with_the_issuer_for_reliefs_at_the_ya_start():
    d = deps()
    run(request(), d)
    relief_queries = [q for q in d.retrieval.requests if q.rule_types == ["relief"]]
    (q,) = relief_queries
    assert q.query_text == "Sunshine Solar Energy (Pvt) Ltd"
    assert q.as_of_date == date(2025, 4, 1)
    assert (q.taxpayer_category, q.caller) == ("resident_individual", "c1-orchestrator")


def test_relief_stage_runs_only_with_an_invoice_and_its_citation_is_explained():
    result = run(request(), deps())
    assert list(result.stages_run) == [
        "extract",
        "normalise",
        "retrieve",
        "discover_reliefs",
        "calculate",
        "explain",
        "score",
        "assure",
        "deliver",
    ]
    (claim,) = [c for c in result.explanation.claims if c.step_id == "S1_SOLAR_PANEL_RELIEF"]
    assert claim.citation_refs == ("RULE-SOLAR_RELIEF-V1",)
    assert result.rules is not None


def test_applied_relief_confidence_counts_in_c1_confidence():
    mock = json.loads((MOCK_DIR / "retrieval_reliefs_ya2025_26.json").read_text(encoding="utf-8"))
    (solar,) = [r for r in mock["results"] if r["rule_key"] == "solar_panel_relief"]
    score = D(str(solar["confidence"]["score"]))
    assert score < D("0.9")  # below the personal relief / band rules, so it is the minimum
    result = run(request(), deps())
    assert result.confidence.retrieval == score


# --- conditions: any "No" -> not applied, unanswered -> needs review --------------------------


def test_grid_answer_missing_needs_review_and_relief_is_not_applied():
    answers = {"resident": "yes", "on_premises": "yes"}
    result = run(request(inputs=(confirmed(answers=answers),)), deps())
    assert result.status == "needs_review"
    assert result.calc.total_tax == D(96000)
    assert "S1_SOLAR_PANEL_RELIEF" not in [s.step_id for s in result.calc.steps]
    f = finding(result)
    assert f.status == "needs_review"
    assert f.reasons == ("answer: Are the solar panels connected to the national grid?",)
    assert result.review_reasons == (
        "Solar panel relief (solar_invoice_2025): answer: Are the solar panels connected to "
        "the national grid?",
    )
    assert [q.answer for q in f.questions] == ["yes", "yes", None]


def test_grid_answer_no_is_not_applied_with_the_reason():
    answers = YES | {"national_grid": "no"}
    result = run(request(inputs=(confirmed(answers=answers),)), deps())
    assert result.status == "delivered"  # a clear "No" needs no review
    assert result.calc.total_tax == D(96000)
    f = finding(result)
    assert f.status == "not_applied"
    assert f.reasons == ("the solar panels must be connected to the national grid",)
    assert f.remaining_claimable is None


def test_no_answers_at_all_needs_review_for_every_question():
    result = run(request(inputs=(confirmed(answers={}),)), deps())
    assert finding(result).status == "needs_review"
    assert len(finding(result).reasons) == 3


@pytest.mark.parametrize("day", [date(2025, 3, 31), date(2026, 4, 1), date(2023, 6, 1)])
def test_invoice_dated_outside_the_ya_is_not_applied(day):
    result = run(request(inputs=(confirmed(day=day),)), deps())
    assert result.calc.total_tax == D(96000)
    f = finding(result)
    assert f.status == "not_applied"
    assert f.reasons == (
        f"the invoice is dated {day}, outside this year of assessment (2025-04-01 to 2026-03-31)",
    )


@pytest.mark.parametrize("day", [date(2025, 4, 1), date(2026, 3, 31)])
def test_invoice_dated_on_the_ya_edges_is_applied(day):
    assert finding(run(request(inputs=(confirmed(day=day),)), deps())).status == "applied"


def test_unconfirmed_amount_and_date_need_review_even_though_c2_read_them():
    result = run(request(inputs=(confirmed(amount=None, day=None),)), deps())
    assert result.status == "needs_review"
    assert result.calc.total_tax == D(96000)
    assert finding(result).reasons == (
        "confirm the date on the invoice",
        "confirm the amount paid on the invoice",
    )


def test_invoice_without_any_user_input_needs_review():
    result = run(request(inputs=()), deps())
    assert result.status == "needs_review"
    assert finding(result).status == "needs_review"
    assert result.calc.total_tax == D(96000)


# --- documents: no invoice, bank statements, unclear issuers --------------------------------


def test_no_invoice_leaves_the_tax_unchanged_at_96000():
    result = run(request(documents=(), inputs=()), deps())
    assert result.status == "delivered"
    assert result.calc.total_tax == D(96000)
    assert result.relief_findings == () and result.documents == ()
    assert "discover_reliefs" not in result.stages_run


def test_bank_statement_is_listed_and_ignored():
    result = run(
        request(documents=("payslip_clean", "bank_statement_2025"), inputs=(), income=None), deps()
    )
    assert result.status == "delivered"
    assert result.calc.total_tax == D(96000)  # from the payslip
    assert [(x.document_id, x.role) for x in result.documents] == [
        ("payslip_clean", "payslip"),
        ("bank_statement_2025", "not_analysed"),
    ]
    assert result.documents[1].note == (
        "not analysed yet: C1 does not use bank_statement documents"
    )
    assert "discover_reliefs" not in result.stages_run
    assert result.relief_findings == ()


def test_payslip_bank_statement_and_invoice_together():
    docs = ("payslip_clean", "bank_statement_2025", INVOICE)
    result = run(request(documents=docs, income=None), deps())
    assert [x.role for x in result.documents] == ["payslip", "not_analysed", "relief_candidate"]
    assert result.calc.total_tax == D(36000)
    assert result.calc.refund == D(14000)  # APIT 50,000 from the payslip


def test_two_payslips_are_still_refused():
    result = run(
        request(documents=("payslip_clean", "payslip_refund"), inputs=(), income=None), deps()
    )
    assert result.status == "failed"
    assert "more than one payslip" in result.review_reasons[0]


def test_unreadable_issuer_asks_what_the_payment_was_for():
    docs = ("receipt_unreadable_issuer",)
    result = run(request(documents=docs, inputs=()), deps())
    assert result.status == "needs_review"
    f = finding(result, "receipt_unreadable_issuer")
    assert f.status == "needs_review"
    assert f.reasons == (
        "the issuer of this receipt could not be read: say what the payment was for",
    )


def test_users_answer_finds_the_relief_when_the_issuer_is_unclear():
    docs = ("receipt_unreadable_issuer",)
    inputs = (
        confirmed(
            "receipt_unreadable_issuer",
            amount="12500.00",
            payment_for="rooftop solar panel installation",
        ),
    )
    result = run(request(documents=docs, inputs=inputs), deps())
    f = finding(result, "receipt_unreadable_issuer")
    assert (f.status, f.query) == ("applied", "rooftop solar panel installation")
    assert step(result, "S1_SOLAR_PANEL_RELIEF").result == D("12500.00")


def test_issuer_with_no_relief_asks_what_it_was_for(tmp_path):
    d = ocr_dir(tmp_path, grocery=invoice("Keells Super (Pvt) Ltd", kind="receipt"))
    result = run(request(documents=("grocery",), inputs=()), deps(d))
    f = finding(result, "grocery")
    assert f.status == "needs_review"
    assert f.reasons == ("C3 found no relief for the issuer: say what the payment was for",)


def test_payment_with_no_relief_is_simply_listed(tmp_path):
    d = ocr_dir(tmp_path, grocery=invoice("Keells Super (Pvt) Ltd", kind="receipt"))
    inputs = (confirmed("grocery", payment_for="weekly groceries"),)
    result = run(request(documents=("grocery",), inputs=inputs), deps(d))
    assert result.status == "delivered"
    assert finding(result, "grocery").status == "no_relief"
    assert result.calc.total_tax == D(96000)


def test_relief_on_an_income_c1_does_not_calculate_is_found_but_not_applied(tmp_path):
    d = ocr_dir(tmp_path, lease=invoice("Lanka Property Agents", kind="receipt"))
    inputs = (confirmed("lease", payment_for="house rent received"),)
    result = run(request(documents=("lease",), inputs=inputs), deps(d))
    f = finding(result, "lease")
    assert (f.rule_key, f.status) == ("rental_relief", "not_supported")
    assert f.reasons == ("needs investment income, which C1 does not calculate yet",)
    assert result.status == "delivered"
    assert result.calc.total_tax == D(96000)


def test_one_payment_matching_two_reliefs_needs_review(tmp_path):
    d = ocr_dir(tmp_path, mixed=invoice("Sunshine Solar Energy (Pvt) Ltd"))
    inputs = (confirmed("mixed", payment_for="solar panels and house rent"),)
    result = run(request(documents=("mixed",), inputs=inputs), deps(d))
    by_key = {f.rule_key: f for f in result.relief_findings}
    assert by_key["rental_relief"].status == "not_supported"
    # rent is not supported, so only one relief could apply: solar is applied
    assert by_key["solar_panel_relief"].status == "applied"


def test_two_invoices_for_the_same_relief_share_one_yearly_cap(tmp_path):
    d = ocr_dir(tmp_path, solar_a=invoice(), solar_b=invoice())
    inputs = (confirmed("solar_a", amount="400000.00"), confirmed("solar_b", amount="400000.00"))
    result = run(request(documents=("solar_a", "solar_b"), inputs=inputs), deps(d))
    s = step(result, "S1_SOLAR_PANEL_RELIEF")
    assert s.inputs == {"cap": D(600000), "amount_paid": D(800000)}
    assert s.result == D(600000)
    assert result.calc.total_tax == D(36000)
    assert [f.remaining_claimable for f in result.relief_findings] == [D(200000), D(200000)]
    assert "combined with 1 other document" in result.relief_findings[0].reasons[0]


# --- PII, C3 outages, conflicts, other years, regenerate ------------------------------------


def test_issuer_query_keeps_the_business_name_and_strips_contact_ids():
    q = relief_query(
        "Sunshine Solar Energy (Pvt) Ltd  info@sunshine.lk  +94 77 123 4567  NIC 199012345678",
        "panels for 200123456V's house",
    )
    assert q == "Sunshine Solar Energy (Pvt) Ltd NIC panels for 's house"
    assert relief_query(None, "  ") is None


def test_relief_query_never_adds_the_generic_word_relief():
    assert "relief" not in relief_query("Sunshine Solar Energy (Pvt) Ltd", None).lower()


class ReliefsDown(MockRetrievalClient):
    def retrieve(self, request):
        if request.rule_types == ["relief"]:
            raise ComponentUnavailable("C3 did not answer")
        return super().retrieve(request)


def test_c3_down_for_reliefs_needs_review_but_still_calculates():
    result = run(request(), deps(retrieval=ReliefsDown()))
    assert result.status == "needs_review"
    assert result.calc.total_tax == D(96000)
    assert finding(result).reasons == ("reliefs could not be checked: C3 did not answer",)


def test_seed_and_c3_disagree_on_solar_is_surfaced():
    (solar,) = load_seed_reliefs("2025/26")
    seed = (solar.model_copy(update={"value": D(500000)}),)
    result = run(request(), deps(seed_reliefs=lambda ya: seed))
    assert result.status == "needs_review"
    assert "solar_panel_relief: seed cap 500000, c3 cap 600000" in result.conflicts
    assert result.confidence.conflict_penalty == D("0.6")
    assert result.calc.total_tax == D(36000)  # C3's value is used, and flagged


def test_solar_applies_in_ya_2024_25_too():
    inputs = (confirmed(day=date(2024, 9, 1)),)
    result = run(request(ya="2024/25", inputs=inputs, apit=0), deps())
    assert finding(result).status == "applied"
    # 3,000,000 - 1,200,000 - 600,000 = 1,200,000 -> 30,000 + 60,000 + 36,000 = 126,000
    assert result.calc.total_tax == D(126000)
    assert result.conflicts == ()  # the 2024/25 seed has solar too


def test_regenerate_from_retrieve_rediscovers_the_reliefs():
    verdict = AssuranceVerdict(
        request_id="req-0001",
        attempt=1,
        decision="REGENERATE",
        rerun_from="retrieve",
        failed_claims=(FailedClaim(step_id="S3_BAND_1", reason="x"),),
    )
    d = deps(script=[verdict, "PASS"])
    result = run(request(), d)
    assert result.attempt == 2 and result.status == "delivered"
    assert result.stages_run.count("discover_reliefs") == 2
    assert result.calc.total_tax == D(36000)
    assert [len(t.calc_steps) for t in d.assurance.traces] == [7, 7]


def test_same_request_gives_identical_result():
    first = run(request(), deps()).model_dump_json()
    assert all(run(request(), deps()).model_dump_json() == first for _ in range(5))


# --- mock C3 relief answers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, keys",
    [
        ("Sunshine Solar Energy (Pvt) Ltd", ["solar_panel_relief"]),
        ("house rent", ["rental_relief"]),
        ("Current Holdings Ltd", []),  # "rent" only as a whole word
        ("Keells Super", []),
    ],
)
def test_mock_c3_returns_the_reliefs_the_query_names(text, keys):
    from j26_contracts.retrieval import RetrievalRequest

    out = MockRetrievalClient().retrieve(
        RetrievalRequest(
            schema_version="1.0.0",
            request_id="req-0001",
            query_text=text,
            as_of_date=date(2025, 4, 1),
            rule_types=["relief"],
        )
    )
    assert [r.rule_key for r in out.results] == keys
    assert [r.rank for r in out.results] == list(range(1, len(keys) + 1))
    assert out.warnings == ([] if keys else ["NO_MATCH"])


# --- byte-identical without extra documents -------------------------------------------------


def _no_extra_documents_runs() -> dict:
    """The flows without invoices, as captured before relief discovery existed (7ff4906)."""

    def req(ya="2025/26", **kw):
        return TaxRequest(request_id="req-0001", year_of_assessment=ya, **kw)

    typed = {"employment_income": D(3000000), "apit_paid": D(50000)}
    regen = AssuranceVerdict(
        request_id="req-0001",
        attempt=1,
        decision="REGENERATE",
        rerun_from="retrieve",
        failed_claims=(FailedClaim(step_id="S3_BAND_2", reason="wrong band rate"),),
    )
    cases = {
        "typed": (req(**typed), []),
        "typed_2024_25": (req("2024/25", **typed), []),
        "payslip_clean": (req(documents=("payslip_clean",)), []),
        "payslip_noisy": (req(documents=("payslip_noisy_low_confidence",)), []),
        "regenerate_then_pass": (req(**typed), [regen, "PASS"]),
    }
    out = {}
    for name, (r, script) in cases.items():
        d = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient(script))
        res = run(r, d)
        out[name] = {
            "result": res.model_dump(mode="json", exclude={"documents", "relief_findings"}),
            "traces": [t.model_dump(mode="json") for t in d.assurance.traces],
            "c3_requests": [q.model_dump(mode="json") for q in d.retrieval.requests],
        }
    return out


def test_without_extra_documents_results_and_traces_are_byte_identical():
    now = json.dumps(_no_extra_documents_runs(), indent=1, sort_keys=True)
    assert now == SNAPSHOT.read_text(encoding="utf-8")
