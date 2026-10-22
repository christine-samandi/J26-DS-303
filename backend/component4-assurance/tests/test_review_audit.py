
from datetime import timezone

import pytest

from human_review.audit import (
    AuditEventType,
    create_audit_event,
)
from human_review.models import ReviewerDecision


@pytest.mark.parametrize(
    "event_type",
    [
        AuditEventType.CASE_CREATED,
        AuditEventType.REVIEW_STARTED,
        AuditEventType.REVIEW_COMPLETED,
    ],
)
def test_audit_event_types(event_type):
    actor_id = None
    decision = None

    if event_type == AuditEventType.REVIEW_COMPLETED:
        actor_id = "reviewer-001"
        decision = ReviewerDecision.REJECT

    event = create_audit_event(
        case_id="REV-001",
        event_type=event_type,
        actor_id=actor_id,
        decision=decision,
    )

    assert event.case_id == "REV-001"
    assert event.event_type == event_type
    assert event.event_id.startswith("AUD-")


def test_audit_event_ids_are_unique():
    first = create_audit_event(
        "REV-001",
        AuditEventType.CASE_CREATED,
    )

    second = create_audit_event(
        "REV-001",
        AuditEventType.CASE_CREATED,
    )

    assert first.event_id != second.event_id


def test_audit_timestamp_is_utc():
    event = create_audit_event(
        "REV-001",
        AuditEventType.REVIEW_STARTED,
    )

    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() == timezone.utc.utcoffset(None)


def test_blank_case_id_rejected():
    with pytest.raises(ValueError, match="Case ID"):
        create_audit_event(
            "   ",
            AuditEventType.CASE_CREATED,
        )


@pytest.mark.parametrize(
    "actor_id,decision",
    [
        (None, ReviewerDecision.REJECT),
        ("reviewer-001", None),
    ],
)
def test_completed_event_requires_reviewer_and_decision(
    actor_id,
    decision,
):
    with pytest.raises(ValueError):
        create_audit_event(
            "REV-001",
            AuditEventType.REVIEW_COMPLETED,
            actor_id=actor_id,
            decision=decision,
        )


def test_start_event_cannot_contain_review_decision():
    with pytest.raises(ValueError, match="Only completed"):
        create_audit_event(
            "REV-001",
            AuditEventType.REVIEW_STARTED,
            decision=ReviewerDecision.APPROVE,
        )
