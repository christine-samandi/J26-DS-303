
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from consistency_checker.checker import (
    VerificationFinding,
    check_line_item_sum,
    check_total_consistency,
)
from decision_engine.engine import (
    AssuranceDecision,
    decide_assurance_action,
)
from citation_verifier.verifier import (
    CitationClaim,
    EvidenceSource,
    verify_citation,
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

def assure_line_item_total(
    reported_total: Decimal,
    line_items: Sequence[Decimal],
    regeneration_attempts: int = 0,
) -> AssuranceResult:
    """Verify a reported total against independently summed line items."""

    finding = check_line_item_sum(
        reported_total=reported_total,
        line_items=line_items,
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

def assure_citation(
    claim: CitationClaim,
    sources: Sequence[EvidenceSource],
    regeneration_attempts: int = 0,
) -> AssuranceResult:
    """Run citation traceability verification and risk-based decision-making."""

    finding = verify_citation(
        claim=claim,
        sources=sources,
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
