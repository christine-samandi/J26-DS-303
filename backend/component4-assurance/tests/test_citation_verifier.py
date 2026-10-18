
import pytest

from citation_verifier.verifier import (
    CitationClaim,
    EvidenceSource,
    verify_citation,
)
from consistency_checker.checker import ErrorCode, Severity


SOURCES = [
    EvidenceSource(
        source_id="TAX-DOC-001",
        text=(
            "The example deduction is available "
            "subject to the stated conditions."
        ),
    )
]


def test_valid_citation_passes():
    claim = CitationClaim(
        claim_id="CLAIM-001",
        source_id="TAX-DOC-001",
        excerpt="available subject to the stated conditions",
    )

    result = verify_citation(claim, SOURCES)

    assert result.passed is True
    assert result.error_code == ErrorCode.NONE


def test_whitespace_differences_are_normalized():
    claim = CitationClaim(
        claim_id="CLAIM-002",
        source_id="TAX-DOC-001",
        excerpt="available   subject to the stated conditions",
    )

    result = verify_citation(claim, SOURCES)

    assert result.passed is True


def test_missing_citation_detected():
    claim = CitationClaim(
        claim_id="CLAIM-003",
        source_id=None,
        excerpt=None,
    )

    result = verify_citation(claim, SOURCES)

    assert result.passed is False
    assert result.severity == Severity.WARNING
    assert result.error_code == ErrorCode.MISSING_CITATION


def test_unknown_source_detected():
    claim = CitationClaim(
        claim_id="CLAIM-004",
        source_id="UNKNOWN-DOC",
        excerpt="example deduction",
    )

    result = verify_citation(claim, SOURCES)

    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.error_code == ErrorCode.UNKNOWN_SOURCE


def test_missing_evidence_detected():
    claim = CitationClaim(
        claim_id="CLAIM-005",
        source_id="TAX-DOC-001",
        excerpt=None,
    )

    result = verify_citation(claim, SOURCES)

    assert result.error_code == ErrorCode.MISSING_EVIDENCE


def test_evidence_not_found_detected():
    claim = CitationClaim(
        claim_id="CLAIM-006",
        source_id="TAX-DOC-001",
        excerpt="This completely different passage is not present.",
    )

    result = verify_citation(claim, SOURCES)

    assert result.passed is False
    assert result.error_code == ErrorCode.EVIDENCE_NOT_FOUND


def test_duplicate_source_ids_rejected():
    claim = CitationClaim(
        claim_id="CLAIM-007",
        source_id="TAX-DOC-001",
        excerpt="example deduction",
    )

    duplicate_sources = [SOURCES[0], SOURCES[0]]

    with pytest.raises(ValueError, match="Duplicate"):
        verify_citation(claim, duplicate_sources)


def test_empty_claim_id_rejected():
    claim = CitationClaim(
        claim_id="",
        source_id="TAX-DOC-001",
        excerpt="example deduction",
    )

    with pytest.raises(ValueError, match="Claim ID"):
        verify_citation(claim, SOURCES)
