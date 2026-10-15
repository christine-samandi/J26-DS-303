
from dataclasses import dataclass
from decimal import Decimal

from consistency_checker.checker import (
    VerificationFinding,
    check_total_consistency,
)
from decision_engine.engine import (
    AssuranceDecision,
    decide_assurance_action,
)


@dataclass(frozen=True)
class AssuranceResult:
    """Combined findings and decision from Component 4."""

    findings: tuple[VerificationFinding, ...]
    decision: AssuranceDecision


def assure_tax_total(
    reported_total: Decimal,
    calculated_total: Decimal,
    regeneration_attempts: int = 0,
) -> AssuranceResult:
    """Run the initial deterministic assurance pipeline."""

    finding = check_total_consistency(
        reported_total=reported_total,
        calculated_total=calculated_total,
    )

    findings = (finding,)

    decision = decide_assurance_action(
        findings=findings,
        regeneration_attempts=regeneration_attempts,
    )

    return AssuranceResult(
        findings=findings,
        decision=decision,
    )
