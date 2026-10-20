
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from consistency_checker.checker import VerificationFinding
from decision_engine.engine import AssuranceDecision


class ReviewCaseStatus(str, Enum):
    """Current status of a human review case."""

    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    COMPLETED = "COMPLETED"


class ReviewerDecision(str, Enum):
    """Possible decisions submitted by a human reviewer."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CORRECTION = "REQUEST_CORRECTION"


@dataclass(frozen=True)
class ReviewFeedback:
    """Feedback submitted by a human reviewer."""

    reviewer_id: str
    decision: ReviewerDecision
    comment: str
    reviewed_at: datetime


@dataclass(frozen=True)
class ReviewCase:
    """A tax-assurance result escalated for human review."""

    case_id: str
    request_id: str | None
    status: ReviewCaseStatus
    created_at: datetime
    findings: tuple[VerificationFinding, ...]
    decision: AssuranceDecision

    # Empty until a human reviewer completes the case.
    feedback: ReviewFeedback | None = None
