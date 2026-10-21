
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_combined
from citation_verifier.verifier import CitationClaim, EvidenceSource
from human_review.models import ReviewCaseStatus
from human_review.repository import (
    DuplicateReviewCaseError,
    ReviewCaseNotFoundError,
    ReviewCaseRepository,
)
from human_review.service import create_review_case


def make_review_case():
    """Generate a review case from a failed calculation."""

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

    return create_review_case(result)


def test_add_review_case():
    repository = ReviewCaseRepository()
    case = make_review_case()

    saved_case = repository.add_case(case)

    assert saved_case.case_id == case.case_id
    assert saved_case.status == ReviewCaseStatus.PENDING


def test_get_review_case():
    repository = ReviewCaseRepository()
    case = make_review_case()

    repository.add_case(case)

    retrieved_case = repository.get_case(case.case_id)

    assert retrieved_case == case


def test_list_review_cases():
    repository = ReviewCaseRepository()

    repository.add_case(make_review_case())
    repository.add_case(make_review_case())

    cases = repository.list_cases()

    assert len(cases) == 2


def test_filter_pending_cases():
    repository = ReviewCaseRepository()

    first_case = make_review_case()
    second_case = make_review_case()

    repository.add_case(first_case)
    repository.add_case(second_case)

    repository.start_review(first_case.case_id)

    pending_cases = repository.list_cases(
        status=ReviewCaseStatus.PENDING
    )

    assert len(pending_cases) == 1
    assert pending_cases[0].case_id == second_case.case_id


def test_start_review_updates_status():
    repository = ReviewCaseRepository()
    case = make_review_case()

    repository.add_case(case)

    updated = repository.start_review(case.case_id)

    assert updated.status == ReviewCaseStatus.IN_REVIEW
    assert updated.findings == case.findings
    assert updated.decision == case.decision


def test_duplicate_case_rejected():
    repository = ReviewCaseRepository()
    case = make_review_case()

    repository.add_case(case)

    with pytest.raises(DuplicateReviewCaseError):
        repository.add_case(case)


def test_unknown_case_rejected():
    repository = ReviewCaseRepository()

    with pytest.raises(ReviewCaseNotFoundError):
        repository.get_case("REV-UNKNOWN")


def test_review_cannot_start_twice():
    repository = ReviewCaseRepository()
    case = make_review_case()

    repository.add_case(case)
    repository.start_review(case.case_id)

    with pytest.raises(ValueError, match="PENDING"):
        repository.start_review(case.case_id)
