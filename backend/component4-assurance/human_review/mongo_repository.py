
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
from human_review.audit import (
    AuditEventType,
    ReviewAuditEvent,
    create_audit_event,
)


class MongoReviewCaseRepository:
    """Store human review cases in MongoDB."""

    def __init__(self, collection):
        self.collection = collection
        self.case_adapter = TypeAdapter(ReviewCase)
        self.feedback_adapter = TypeAdapter(ReviewFeedback)
        self.audit_adapter = TypeAdapter(ReviewAuditEvent)


    def _to_document(self, case: ReviewCase) -> dict:
        """Convert a new review case into a MongoDB document."""

        document = self.case_adapter.dump_python(
            case,
            mode="json",
        )

        document["_id"] = case.case_id

        # Record the initial event when the case is created.
        created_event = create_audit_event(
            case_id=case.case_id,
            event_type=AuditEventType.CASE_CREATED,
        )

        document["events"] = [
            self.audit_adapter.dump_python(
                created_event,
                mode="json",
            )
        ]

        return document


    def _to_case(self, document: dict) -> ReviewCase:
        """Convert MongoDB data into the review-case model."""

        data = dict(document)

        data.pop("_id", None)
        data.pop("events", None)

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
        """Start a review and record the transition event."""

        event = create_audit_event(
            case_id=case_id,
            event_type=AuditEventType.REVIEW_STARTED,
        )

        event_document = self.audit_adapter.dump_python(
            event,
            mode="json",
        )

        document = self.collection.find_one_and_update(
            {
                "_id": case_id,
                "status": ReviewCaseStatus.PENDING.value,
            },
            {
                "$set": {
                    "status": ReviewCaseStatus.IN_REVIEW.value,
                },
                "$push": {
                    "events": event_document,
                },
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

        # Record the reviewer's final decision as an audit event.
        event = create_audit_event(
            case_id=case_id,
            event_type=AuditEventType.REVIEW_COMPLETED,
            actor_id=feedback.reviewer_id,
            decision=feedback.decision,
        )

        event_document = self.audit_adapter.dump_python(
            event,
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
                },
                "$push": {
                    "events": event_document,
                },
            },
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            self.get_case(case_id)
            raise ValueError(
                "Only IN_REVIEW cases can be completed."
            )

        return self._to_case(document)

    def get_history(
        self,
        case_id: str,
    ) -> tuple[ReviewAuditEvent, ...]:
        """Retrieve the stored event history of a review case."""

        document = self.collection.find_one(
            {"_id": case_id},
            {"events": 1},
        )

        if document is None:
            raise ReviewCaseNotFoundError(
                f"Review case {case_id} was not found."
            )

        events = document.get("events", [])

        return tuple(
            self.audit_adapter.validate_python(event)
            for event in events
        )
