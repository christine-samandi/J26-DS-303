from dataclasses import replace
from datetime import datetime, timezone
from human_review.models import (
    ReviewCase, 
    ReviewCaseStatus,
    ReviewFeedback,
    ReviewerDecision,
)


class ReviewCaseNotFoundError(Exception):
    """Raised when a review case does not exist."""


class DuplicateReviewCaseError(Exception):
    """Raised when a case with the same ID already exists."""


class ReviewCaseRepository:
    """In-memory repository for human review cases."""

    def __init__(self):
        self._cases: dict[str, ReviewCase] = {}

    def add_case(self, case: ReviewCase) -> ReviewCase:
        """Store a newly created review case."""

        if case.case_id in self._cases:
            raise DuplicateReviewCaseError(
                f"Review case {case.case_id} already exists."
            )

        self._cases[case.case_id] = case
        return case

    def get_case(self, case_id: str) -> ReviewCase:
        """Retrieve a review case by its unique ID."""

        if case_id not in self._cases:
            raise ReviewCaseNotFoundError(
                f"Review case {case_id} was not found."
            )

        return self._cases[case_id]

    def list_cases(
        self,
        status: ReviewCaseStatus | None = None,
    ) -> tuple[ReviewCase, ...]:
        """List all review cases or filter by status."""

        cases = tuple(self._cases.values())

        if status is None:
            return cases

        return tuple(
            case for case in cases
            if case.status == status
        )

    def start_review(self, case_id: str) -> ReviewCase:
        """Move a pending case to IN_REVIEW."""

        case = self.get_case(case_id)

        if case.status != ReviewCaseStatus.PENDING:
            raise ValueError(
                "Only PENDING review cases can enter IN_REVIEW."
            )

        updated_case = replace(
            case,
            status=ReviewCaseStatus.IN_REVIEW,
        )

        self._cases[case_id] = updated_case

        return updated_case
    
    def complete_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: ReviewerDecision,
        comment: str,
    ) -> ReviewCase:
        """Complete a review case and store the reviewer's feedback."""

        case = self.get_case(case_id)

        # A reviewer must start the case before completing it.
        if case.status != ReviewCaseStatus.IN_REVIEW:
            raise ValueError(
                "Only IN_REVIEW cases can be completed."
            )

        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("Reviewer ID cannot be empty.")

        if not isinstance(decision, ReviewerDecision):
            raise ValueError("Invalid reviewer decision.")

        if not comment or not comment.strip():
            raise ValueError("Review comment cannot be empty.")

        # Create the reviewer feedback record.
        feedback = ReviewFeedback(
            reviewer_id=reviewer_id.strip(),
            decision=decision,
            comment=comment.strip(),
            reviewed_at=datetime.now(timezone.utc),
        )

        # Preserve the original findings and assurance decision.
        updated_case = replace(
            case,
            status=ReviewCaseStatus.COMPLETED,
            feedback=feedback,
        )

        # Save the updated review case.
        self._cases[case_id] = updated_case

        return updated_case
