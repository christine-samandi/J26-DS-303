
from datetime import datetime, timezone

from pydantic import TypeAdapter
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from human_review.models import (
    ReviewCase,
    ReviewCaseStatus,
    ReviewFeedback,
    ReviewerDecision,
)
from human_review.repository import (
    DuplicateReviewCaseError,
    ReviewCaseNotFoundError,
)


class MongoReviewCaseRepository:
    """Store human review cases in MongoDB."""

    def __init__(self, collection):
        self.collection = collection
        self.case_adapter = TypeAdapter(ReviewCase)
        self.feedback_adapter = TypeAdapter(ReviewFeedback)

    def _to_document(self, case: ReviewCase) -> dict:
        """Convert a ReviewCase into a MongoDB-compatible document."""

        document = self.case_adapter.dump_python(
            case,
            mode="json",
        )

        # Use the review case ID as MongoDB's unique primary key.
        document["_id"] = case.case_id
        return document

    def _to_case(self, document: dict) -> ReviewCase:
        """Convert a MongoDB document back into a ReviewCase."""

        data = dict(document)
        data.pop("_id", None)

        return self.case_adapter.validate_python(data)

    def add_case(self, case: ReviewCase) -> ReviewCase:
        """Save a newly created human review case."""

        try:
            self.collection.insert_one(
                self._to_document(case)
            )
        except DuplicateKeyError:
            raise DuplicateReviewCaseError(
                f"Review case {case.case_id} already exists."
            )

        return case

    def get_case(self, case_id: str) -> ReviewCase:
        """Retrieve a review case using its case ID."""

        document = self.collection.find_one(
            {"_id": case_id}
        )

        if document is None:
            raise ReviewCaseNotFoundError(
                f"Review case {case_id} was not found."
            )

        return self._to_case(document)

    def list_cases(
        self,
        status: ReviewCaseStatus | None = None,
    ) -> tuple[ReviewCase, ...]:
        """List review cases, optionally filtering by status."""

        query = {}

        if status is not None:
            query["status"] = status.value

        documents = self.collection.find(query)

        return tuple(
            self._to_case(document)
            for document in documents
        )

    def start_review(self, case_id: str) -> ReviewCase:
        """Move a pending case into review."""

        document = self.collection.find_one_and_update(
            {
                "_id": case_id,
                "status": ReviewCaseStatus.PENDING.value,
            },
            {
                "$set": {
                    "status": ReviewCaseStatus.IN_REVIEW.value
                }
            },
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            self.get_case(case_id)
            raise ValueError(
                "Only PENDING review cases can enter IN_REVIEW."
            )

        return self._to_case(document)

    def complete_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: ReviewerDecision,
        comment: str,
    ) -> ReviewCase:
        """Complete a case and store the reviewer's feedback."""

        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("Reviewer ID cannot be empty.")

        if not isinstance(decision, ReviewerDecision):
            raise ValueError("Invalid reviewer decision.")

        if not comment or not comment.strip():
            raise ValueError("Review comment cannot be empty.")

        feedback = ReviewFeedback(
            reviewer_id=reviewer_id.strip(),
            decision=decision,
            comment=comment.strip(),
            reviewed_at=datetime.now(timezone.utc),
        )

        feedback_document = self.feedback_adapter.dump_python(
            feedback,
            mode="json",
        )

        document = self.collection.find_one_and_update(
            {
                "_id": case_id,
                "status": ReviewCaseStatus.IN_REVIEW.value,
            },
            {
                "$set": {
                    "status": ReviewCaseStatus.COMPLETED.value,
                    "feedback": feedback_document,
                }
            },
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            self.get_case(case_id)
            raise ValueError(
                "Only IN_REVIEW cases can be completed."
            )

        return self._to_case(document)
