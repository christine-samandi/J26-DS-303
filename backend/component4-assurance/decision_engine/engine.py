
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from consistency_checker.checker import Severity, VerificationFinding


class AssuranceAction(str, Enum):
    PASS = "PASS"
    REGENERATE = "REGENERATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass(frozen=True)
class AssuranceDecision:
    action: AssuranceAction
    reason: str
    failed_checks: int


def decide_assurance_action(
    findings: Sequence[VerificationFinding],
    regeneration_attempts: int = 0,
    max_regenerations: int = 2,
) -> AssuranceDecision:
    """Select an assurance action from verification findings."""

    if regeneration_attempts < 0:
        raise ValueError("regeneration_attempts cannot be negative")

    if max_regenerations < 0:
        raise ValueError("max_regenerations cannot be negative")

    if not findings:
        return AssuranceDecision(
            action=AssuranceAction.HUMAN_REVIEW,
            reason="No verification findings available.",
            failed_checks=0,
        )

    failures = [finding for finding in findings if not finding.passed]

    if not failures:
        return AssuranceDecision(
            action=AssuranceAction.PASS,
            reason="All available verification checks passed.",
            failed_checks=0,
        )

    critical_failures = [
        finding
        for finding in failures
        if finding.severity == Severity.CRITICAL
    ]

    if critical_failures:
        return AssuranceDecision(
            action=AssuranceAction.HUMAN_REVIEW,
            reason="Critical verification failure detected.",
            failed_checks=len(failures),
        )

    if regeneration_attempts >= max_regenerations:
        return AssuranceDecision(
            action=AssuranceAction.HUMAN_REVIEW,
            reason="Regeneration limit reached with unresolved failures.",
            failed_checks=len(failures),
        )

    return AssuranceDecision(
        action=AssuranceAction.REGENERATE,
        reason="Non-critical verification failure; regeneration permitted.",
        failed_checks=len(failures),
    )
