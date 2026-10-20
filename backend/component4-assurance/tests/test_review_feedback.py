
from datetime import timezone
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_combined
from citation_verifier.verifier import CitationClaim, EvidenceSource
from human_review.models import (
    ReviewCaseStatus,
    ReviewerDecision,
)
from human_review.repository import (
    ReviewCaseRepository,
    ReviewCaseNotFoundError,
)
from human_review.service import create_review_case


def create_repository_with_case(start_review=True):
    """Create a repository containing one human review case."""

    result = assure_combined(
        reported_total=Decimal("180000"),
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
        claims=[
            CitationClaim(
                claim_id="CL-001",
                source_id="DOC-001",
                excerpt="example relief",
            )
        ],
        sources=[
            EvidenceSource(
                source_id="DOC-001",
                text="The example relief is available.",
            )
        ],
    )

    case = create_review_case(result, request_id="req-001")

    repository = ReviewCaseRepository()
    repository.add_case(case)

    if start_review:
        repository.start_review(case.case_id)

    return repository, case


@pytest.mark.parametrize(
    "reviewer_decision",
    [
        ReviewerDecision.APPROVE,
        ReviewerDecision.REJECT,
        ReviewerDecision.REQUEST_CORRECTION,
    ],
)
def test_reviewer_decisions_are_recorded(reviewer_decision):
    repository, case = create_repository_with_case()

    updated = repository.complete_review(
        case_id=case.case_id,
        reviewer_id="reviewer-001",
        decision=reviewer_decision,
        comment="Reviewed the verification findings.",
    )

    assert updated.status == ReviewCaseStatus.COMPLETED
    assert updated.feedback is not None
    assert updated.feedback.decision == reviewer_decision
    assert updated.feedback.reviewer_id == "reviewer-001"


def test_feedback_preserves_original_findings():
    repository, case = create_repository_with_case()

    updated = repository.complete_review(
        case.case_id,
        "reviewer-001",
        ReviewerDecision.REJECT,
        "The calculation is incorrect.",
    )

    assert updated.findings == case.findings
    assert updated.decision == case.decision


def test_pending_case_cannot_be_completed():
    repository, case = create_repository_with_case(
        start_review=False
    )

    with pytest.raises(ValueError, match="IN_REVIEW"):
        repository.complete_review(
            case.case_id,
            "reviewer-001",
            ReviewerDecision.APPROVE,
            "Reviewed the case.",
        )


def test_completed_case_cannot_be_completed_again():
    repository, case = create_repository_with_case()

    repository.complete_review(
        case.case_id,
        "reviewer-001",
        ReviewerDecision.REJECT,
        "Incorrect calculation.",
    )

    with pytest.raises(ValueError, match="IN_REVIEW"):
        repository.complete_review(
            case.case_id,
            "reviewer-002",
            ReviewerDecision.APPROVE,
            "Second decision.",
        )


def test_blank_reviewer_id_rejected():
    repository, case = create_repository_with_case()

    with pytest.raises(ValueError, match="Reviewer ID"):
        repository.complete_review(
            case.case_id,
            "   ",
            ReviewerDecision.APPROVE,
            "Reviewed the case.",
        )


def test_blank_comment_rejected():
    repository, case = create_repository_with_case()

    with pytest.raises(ValueError, match="comment"):
        repository.complete_review(
            case.case_id,
            "reviewer-001",
            ReviewerDecision.REJECT,
            "   ",
        )


def test_invalid_decision_rejected():
    repository, case = create_repository_with_case()

    with pytest.raises(ValueError, match="Invalid reviewer decision"):
        repository.complete_review(
            case.case_id,
            "reviewer-001",
            "UNKNOWN_DECISION",
            "Reviewed the case.",
        )


def test_feedback_timestamp_is_utc():
    repository, case = create_repository_with_case()

    updated = repository.complete_review(
        case.case_id,
        "reviewer-001",
        ReviewerDecision.APPROVE,
        "Reviewed and approved.",
    )

    assert updated.feedback is not None
    assert updated.feedback.reviewed_at.tzinfo is not None
    assert updated.feedback.reviewed_at.utcoffset() == timezone.utc.utcoffset(None)


def test_completed_case_can_be_retrieved():
    repository, case = create_repository_with_case()

    repository.complete_review(
        case.case_id,
        "reviewer-001",
        ReviewerDecision.REQUEST_CORRECTION,
        "Please correct the calculation.",
    )

    saved = repository.get_case(case.case_id)

    assert saved.status == ReviewCaseStatus.COMPLETED
    assert saved.feedback.decision == ReviewerDecision.REQUEST_CORRECTION


def test_unknown_case_cannot_be_completed():
    repository = ReviewCaseRepository()

    with pytest.raises(ReviewCaseNotFoundError):
        repository.complete_review(
            "REV-UNKNOWN",
            "reviewer-001",
            ReviewerDecision.REJECT,
            "Unknown case.",
        )
