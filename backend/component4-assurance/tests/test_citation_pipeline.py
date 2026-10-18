
import pytest

from assurance_pipeline.pipeline import assure_citation
from citation_verifier.verifier import (
    CitationClaim,
    EvidenceSource,
)
from consistency_checker.checker import ErrorCode
from decision_engine.engine import AssuranceAction


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

    result = assure_citation(claim, SOURCES)

    assert result.decision.action == AssuranceAction.PASS
    assert result.decision.failed_checks == 0
    assert result.findings[0].passed is True


def test_missing_citation_triggers_regeneration():
    claim = CitationClaim(
        claim_id="CLAIM-002",
        source_id=None,
        excerpt=None,
    )

    result = assure_citation(claim, SOURCES)

    assert result.decision.action == AssuranceAction.REGENERATE
    assert result.findings[0].error_code == ErrorCode.MISSING_CITATION


def test_unknown_source_triggers_human_review():
    claim = CitationClaim(
        claim_id="CLAIM-003",
        source_id="UNKNOWN-DOC",
        excerpt="example deduction",
    )

    result = assure_citation(claim, SOURCES)

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.findings[0].error_code == ErrorCode.UNKNOWN_SOURCE


def test_missing_excerpt_triggers_regeneration():
    claim = CitationClaim(
        claim_id="CLAIM-004",
        source_id="TAX-DOC-001",
        excerpt=None,
    )

    result = assure_citation(claim, SOURCES)

    assert result.decision.action == AssuranceAction.REGENERATE
    assert result.findings[0].error_code == ErrorCode.MISSING_EVIDENCE


def test_unmatched_excerpt_triggers_human_review():
    claim = CitationClaim(
        claim_id="CLAIM-005",
        source_id="TAX-DOC-001",
        excerpt="A completely different passage.",
    )

    result = assure_citation(claim, SOURCES)

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.findings[0].error_code == ErrorCode.EVIDENCE_NOT_FOUND


def test_retry_limit_escalates_missing_citation():
    claim = CitationClaim(
        claim_id="CLAIM-006",
        source_id=None,
        excerpt=None,
    )

    result = assure_citation(
        claim,
        SOURCES,
        regeneration_attempts=2,
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.decision.failed_checks == 1


def test_negative_retry_count_rejected():
    claim = CitationClaim(
        claim_id="CLAIM-007",
        source_id="TAX-DOC-001",
        excerpt="example deduction",
    )

    with pytest.raises(ValueError, match="negative"):
        assure_citation(
            claim,
            SOURCES,
            regeneration_attempts=-1,
        )
