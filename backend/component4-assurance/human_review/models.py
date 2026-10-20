
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


@dataclass(frozen=True)
class ReviewCase:
    """A tax-assurance result escalated for human review."""

    case_id: str
    request_id: str | None
    status: ReviewCaseStatus
    created_at: datetime
    findings: tuple[VerificationFinding, ...]
    decision: AssuranceDecision
