
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_combined
from citation_verifier.verifier import CitationClaim, EvidenceSource
from decision_engine.engine import AssuranceAction


SOURCES = [
    EvidenceSource(
        source_id="DOC-001",
        text="Example relief may apply subject to conditions.",
    ),
    EvidenceSource(
        source_id="DOC-002",
        text="Example tax calculation uses the applicable rate.",
    ),
]


def valid_claims():
    return [
        CitationClaim(
            claim_id="CL-001",
            source_id="DOC-001",
            excerpt="relief may apply subject to conditions",
        ),
        CitationClaim(
            claim_id="CL-002",
            source_id="DOC-002",
            excerpt="uses the applicable rate",
        ),
    ]


def run_assurance(
    reported_total=Decimal("150000"),
    claims=None,
    regeneration_attempts=0,
):
    return assure_combined(
        reported_total=reported_total,
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
        claims=valid_claims() if claims is None else claims,
        sources=SOURCES,
        regeneration_attempts=regeneration_attempts,
    )


def test_all_checks_pass():
    result = run_assurance()

    assert result.decision.action == AssuranceAction.PASS
    assert len(result.findings) == 3
    assert all(finding.passed for finding in result.findings)


def test_missing_citation_triggers_regeneration():
    claims = valid_claims()
    claims[1] = CitationClaim(
        claim_id="CL-002",
        source_id=None,
        excerpt=None,
    )

    result = run_assurance(claims=claims)

    assert result.decision.action == AssuranceAction.REGENERATE
    assert result.decision.failed_checks == 1


def test_unknown_source_triggers_human_review():
    claims = valid_claims()
    claims[1] = CitationClaim(
        claim_id="CL-002",
        source_id="UNKNOWN-DOC",
        excerpt="example text",
    )

    result = run_assurance(claims=claims)

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW


def test_wrong_total_triggers_human_review():
    result = run_assurance(reported_total=Decimal("180000"))

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.findings[0].passed is False


def test_multiple_missing_citations():
    claims = [
        CitationClaim("CL-001", None, None),
        CitationClaim("CL-002", None, None),
    ]

    result = run_assurance(claims=claims)

    assert result.decision.action == AssuranceAction.REGENERATE
    assert result.decision.failed_checks == 2


def test_retry_limit_escalates_to_human_review():
    claims = [CitationClaim("CL-001", None, None)]

    result = run_assurance(
        claims=claims,
        regeneration_attempts=2,
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW


def test_empty_claims_rejected():
    with pytest.raises(ValueError, match="At least one citation"):
        run_assurance(claims=[])
