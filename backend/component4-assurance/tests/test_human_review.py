
from datetime import timezone
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_combined
from citation_verifier.verifier import CitationClaim, EvidenceSource
from decision_engine.engine import AssuranceAction
from human_review.models import ReviewCaseStatus
from human_review.service import create_review_case


SOURCES = [
    EvidenceSource(
        source_id="DOC-001",
        text="Example relief may apply subject to conditions.",
    )
]


def get_assurance_result(
    reported_total=Decimal("150000"),
    missing_citation=False,
    regeneration_attempts=0,
):
    claim = CitationClaim(
        claim_id="CL-001",
        source_id=None if missing_citation else "DOC-001",
        excerpt=None if missing_citation else "relief may apply",
    )

    return assure_combined(
        reported_total=reported_total,
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
        claims=[claim],
        sources=SOURCES,
        regeneration_attempts=regeneration_attempts,
    )


def test_critical_failure_creates_pending_review_case():
    result = get_assurance_result(
        reported_total=Decimal("180000"),
    )

    case = create_review_case(result, request_id="req-001")

    assert case.status == ReviewCaseStatus.PENDING
    assert case.decision.action == AssuranceAction.HUMAN_REVIEW
    assert case.request_id == "req-001"
    assert case.case_id.startswith("REV-")


def test_review_case_preserves_original_findings():
    result = get_assurance_result(
        reported_total=Decimal("180000"),
    )

    case = create_review_case(result)

    assert case.findings == result.findings
    assert case.decision == result.decision
    assert case.findings[0].passed is False


def test_exhausted_retries_can_create_review_case():
    result = get_assurance_result(
        missing_citation=True,
        regeneration_attempts=2,
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW

    case = create_review_case(result)

    assert case.status == ReviewCaseStatus.PENDING


def test_pass_result_cannot_create_review_case():
    result = get_assurance_result()

    assert result.decision.action == AssuranceAction.PASS

    with pytest.raises(ValueError, match="HUMAN_REVIEW"):
        create_review_case(result)


def test_regenerate_result_cannot_create_review_case():
    result = get_assurance_result(missing_citation=True)

    assert result.decision.action == AssuranceAction.REGENERATE

    with pytest.raises(ValueError, match="HUMAN_REVIEW"):
        create_review_case(result)


def test_each_review_case_has_unique_id():
    result = get_assurance_result(
        reported_total=Decimal("180000"),
    )

    first_case = create_review_case(result)
    second_case = create_review_case(result)

    assert first_case.case_id != second_case.case_id


def test_review_case_timestamp_is_utc():
    result = get_assurance_result(
        reported_total=Decimal("180000"),
    )

    case = create_review_case(result)

    assert case.created_at.tzinfo is not None
    assert case.created_at.utcoffset() == timezone.utc.utcoffset(None)


def test_blank_request_id_is_rejected():
    result = get_assurance_result(
        reported_total=Decimal("180000"),
    )

    with pytest.raises(ValueError, match="Request ID"):
        create_review_case(result, request_id="   ")
