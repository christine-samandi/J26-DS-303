
from decimal import Decimal

import mongomock
import pytest

from assurance_pipeline.pipeline import assure_combined
from citation_verifier.verifier import CitationClaim, EvidenceSource

from human_review.models import (
    ReviewCaseStatus,
    ReviewerDecision,
)
from human_review.mongo_repository import MongoReviewCaseRepository
from human_review.repository import (
    DuplicateReviewCaseError,
    ReviewCaseNotFoundError,
)
from human_review.service import create_review_case


@pytest.fixture
def repository():
    client = mongomock.MongoClient()
    collection = client["c4_assurance"]["review_cases"]
    return MongoReviewCaseRepository(collection)


def make_case():
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

    return create_review_case(
        result,
        request_id="req-001",
    )


def test_add_and_get_case(repository):
    case = make_case()
    repository.add_case(case)

    saved = repository.get_case(case.case_id)

    assert saved.case_id == case.case_id
    assert saved.status == ReviewCaseStatus.PENDING
    assert saved.request_id == "req-001"


def test_findings_are_preserved(repository):
    case = make_case()
    repository.add_case(case)

    saved = repository.get_case(case.case_id)

    assert saved.findings == case.findings
    assert saved.decision == case.decision


def test_list_pending_cases(repository):
    repository.add_case(make_case())
    repository.add_case(make_case())

    pending = repository.list_cases(
        status=ReviewCaseStatus.PENDING
    )

    assert len(pending) == 2


def test_start_review(repository):
    case = repository.add_case(make_case())

    updated = repository.start_review(case.case_id)

    assert updated.status == ReviewCaseStatus.IN_REVIEW
    assert repository.get_case(case.case_id).status == ReviewCaseStatus.IN_REVIEW


@pytest.mark.parametrize(
    "decision",
    [
        ReviewerDecision.APPROVE,
        ReviewerDecision.REJECT,
        ReviewerDecision.REQUEST_CORRECTION,
    ],
)
def test_complete_review(repository, decision):
    case = repository.add_case(make_case())
    repository.start_review(case.case_id)

    updated = repository.complete_review(
        case_id=case.case_id,
        reviewer_id="reviewer-001",
        decision=decision,
        comment="Verified the calculation.",
    )

    assert updated.status == ReviewCaseStatus.COMPLETED
    assert updated.feedback.decision == decision
    assert updated.feedback.reviewer_id == "reviewer-001"


def test_duplicate_case_rejected(repository):
    case = repository.add_case(make_case())

    with pytest.raises(DuplicateReviewCaseError):
        repository.add_case(case)


def test_unknown_case_rejected(repository):
    with pytest.raises(ReviewCaseNotFoundError):
        repository.get_case("REV-UNKNOWN")


def test_pending_case_cannot_be_completed(repository):
    case = repository.add_case(make_case())

    with pytest.raises(ValueError, match="IN_REVIEW"):
        repository.complete_review(
            case.case_id,
            "reviewer-001",
            ReviewerDecision.APPROVE,
            "Reviewed.",
        )


def test_review_cannot_start_twice(repository):
    case = repository.add_case(make_case())
    repository.start_review(case.case_id)

    with pytest.raises(ValueError, match="PENDING"):
        repository.start_review(case.case_id)


def test_completed_feedback_can_be_retrieved(repository):
    case = repository.add_case(make_case())
    repository.start_review(case.case_id)

    repository.complete_review(
        case.case_id,
        "reviewer-001",
        ReviewerDecision.REQUEST_CORRECTION,
        "Incorrect total.",
    )

    saved = repository.get_case(case.case_id)

    assert saved.status == ReviewCaseStatus.COMPLETED
    assert saved.feedback.comment == "Incorrect total."
