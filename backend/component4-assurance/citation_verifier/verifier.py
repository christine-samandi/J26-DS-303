
import re
from dataclasses import dataclass
from typing import Sequence

from consistency_checker.checker import (
    ErrorCode,
    Severity,
    VerificationFinding,
)


@dataclass(frozen=True)
class EvidenceSource:
    """A source document or retrieved evidence passage."""

    source_id: str
    text: str


@dataclass(frozen=True)
class CitationClaim:
    """A generated claim with its cited evidence."""

    claim_id: str
    source_id: str | None
    excerpt: str | None


def normalize_whitespace(text: str) -> str:
    """Normalize spacing without changing the actual words."""
    return re.sub(r"\s+", " ", text).strip()


def verify_citation(
    claim: CitationClaim,
    sources: Sequence[EvidenceSource],
) -> VerificationFinding:
    """Check whether a citation points to traceable evidence."""

    if not claim.claim_id.strip():
        raise ValueError("Claim ID cannot be empty.")

    source_lookup = {}

    for source in sources:
        if not source.source_id.strip():
            raise ValueError("Source ID cannot be empty.")

        if source.source_id in source_lookup:
            raise ValueError("Duplicate source ID detected.")

        source_lookup[source.source_id] = source

    if not claim.source_id or not claim.source_id.strip():
        return VerificationFinding(
            check_name="citation_traceability",
            passed=False,
            message=f"Claim {claim.claim_id} has no citation.",
            severity=Severity.WARNING,
            error_code=ErrorCode.MISSING_CITATION,
        )

    if claim.source_id not in source_lookup:
        return VerificationFinding(
            check_name="citation_traceability",
            passed=False,
            message=(
                f"Claim {claim.claim_id} cites an unknown source: "
                f"{claim.source_id}."
            ),
            severity=Severity.CRITICAL,
            error_code=ErrorCode.UNKNOWN_SOURCE,
        )

    if not claim.excerpt or not claim.excerpt.strip():
        return VerificationFinding(
            check_name="citation_traceability",
            passed=False,
            message=f"Claim {claim.claim_id} has no evidence excerpt.",
            severity=Severity.WARNING,
            error_code=ErrorCode.MISSING_EVIDENCE,
        )

    source_text = normalize_whitespace(
        source_lookup[claim.source_id].text
    )

    evidence_excerpt = normalize_whitespace(claim.excerpt)

    if evidence_excerpt not in source_text:
        return VerificationFinding(
            check_name="citation_traceability",
            passed=False,
            message=(
                f"Evidence excerpt for claim {claim.claim_id} "
                f"was not found in source {claim.source_id}."
            ),
            severity=Severity.CRITICAL,
            error_code=ErrorCode.EVIDENCE_NOT_FOUND,
        )

    return VerificationFinding(
        check_name="citation_traceability",
        passed=True,
        message=(
            f"Claim {claim.claim_id} has traceable evidence "
            f"in source {claim.source_id}."
        ),
        severity=Severity.INFO,
        error_code=ErrorCode.NONE,
    )
