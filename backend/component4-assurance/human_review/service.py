
from datetime import datetime, timezone
from uuid import uuid4

from assurance_pipeline.pipeline import AssuranceResult
from decision_engine.engine import AssuranceAction
from human_review.models import ReviewCase, ReviewCaseStatus


def create_review_case(
    assurance_result: AssuranceResult,
    request_id: str | None = None,
) -> ReviewCase:
    """Create a pending review case for a HUMAN_REVIEW result."""

    # Only human-review decisions can create review cases.
    if assurance_result.decision.action != AssuranceAction.HUMAN_REVIEW:
        raise ValueError(
            "Review cases can only be created for HUMAN_REVIEW decisions."
        )

    if request_id is not None and not request_id.strip():
        raise ValueError("Request ID cannot be blank.")

    # Generate a unique case identifier.
    case_id = f"REV-{uuid4().hex.upper()}"

    # Record the creation time in UTC.
    created_at = datetime.now(timezone.utc)

    return ReviewCase(
        case_id=case_id,
        request_id=request_id,
        status=ReviewCaseStatus.PENDING,
        created_at=created_at,
        findings=assurance_result.findings,
        decision=assurance_result.decision,
    )
