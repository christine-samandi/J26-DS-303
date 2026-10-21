
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from human_review.models import ReviewerDecision


class AuditEventType(str, Enum):
    """Events in the human review lifecycle."""

    CASE_CREATED = "CASE_CREATED"
    REVIEW_STARTED = "REVIEW_STARTED"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"


@dataclass(frozen=True)
class ReviewAuditEvent:
    """Record describing an event in a review case."""

    event_id: str
    case_id: str
    event_type: AuditEventType
    occurred_at: datetime
    actor_id: str | None = None
    decision: ReviewerDecision | None = None


def create_audit_event(
    case_id: str,
    event_type: AuditEventType,
    actor_id: str | None = None,
    decision: ReviewerDecision | None = None,
) -> ReviewAuditEvent:
    """Create a validated, UTC-timestamped review event."""

    if not case_id or not case_id.strip():
        raise ValueError("Case ID cannot be blank.")

    if not isinstance(event_type, AuditEventType):
        raise ValueError("Invalid audit event type.")

    if actor_id is not None and not actor_id.strip():
        raise ValueError("Actor ID cannot be blank.")

    if event_type == AuditEventType.REVIEW_COMPLETED:
        if not actor_id:
            raise ValueError(
                "Completed review events require a reviewer ID."
            )

        if not isinstance(decision, ReviewerDecision):
            raise ValueError(
                "Completed review events require a valid decision."
            )

    elif decision is not None:
        raise ValueError(
            "Only completed review events can contain a decision."
        )

    return ReviewAuditEvent(
        event_id=f"AUD-{uuid4().hex.upper()}",
        case_id=case_id.strip(),
        event_type=event_type,
        occurred_at=datetime.now(timezone.utc),
        actor_id=actor_id.strip() if actor_id else None,
        decision=decision,
    )
