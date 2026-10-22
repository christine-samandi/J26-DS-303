
import json

import pytest
from pydantic import ValidationError

from llm_judge.models import (
    JudgeEvidence,
    JudgeRequest,
    JudgeVerdictType,
    parse_judge_response,
)
from llm_judge.prompting import build_judge_messages


def make_request():
    return JudgeRequest(
        claim_id="CL-001",
        claim_text="Every employee is exempt from income tax.",
        assessment_date="2025-04-01",
        evidence=[
            JudgeEvidence(
                source_id="LAW-001",
                text=(
                    "A personal relief allowance is available "
                    "to eligible resident individuals."
                ),
                legal_reference="Synthetic legal excerpt",
            )
        ],
    )


def make_response(
    verdict="SUPPORTED",
    evidence_source_ids=None,
    confidence=0.9,
    claim_id="CL-001",
):
    if evidence_source_ids is None:
        evidence_source_ids = ["LAW-001"]

    return json.dumps({
        "claim_id": claim_id,
        "verdict": verdict,
        "confidence": confidence,
        "reason": "Evidence was evaluated.",
        "evidence_source_ids": evidence_source_ids,
    })


def test_request_contains_claim_and_evidence():
    request = make_request()

    assert request.claim_id == "CL-001"
    assert len(request.evidence) == 1


def test_blank_claim_is_rejected():
    with pytest.raises(ValidationError):
        JudgeRequest(
            claim_id="CL-001",
            claim_text="   ",
            assessment_date="2025-04-01",
        )


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValidationError):
        JudgeRequest(
            claim_id="CL-001",
            claim_text="Example claim",
            assessment_date="2025-04-01",
            evidence=[
                JudgeEvidence(source_id="LAW-001", text="Evidence A"),
                JudgeEvidence(source_id="LAW-001", text="Evidence B"),
            ],
        )


def test_prompt_contains_claim_date_and_evidence():
    messages = build_judge_messages(make_request())

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "2025-04-01" in messages[1]["content"]
    assert "LAW-001" in messages[1]["content"]
    assert "Every employee" in messages[1]["content"]


def test_prompt_treats_evidence_as_untrusted():
    messages = build_judge_messages(make_request())

    assert "untrusted data" in messages[0]["content"]


def test_supported_response_is_parsed():
    result = parse_judge_response(
        make_response(),
        make_request(),
    )

    assert result.verdict == JudgeVerdictType.SUPPORTED
    assert result.confidence == 0.9


def test_contradicted_response_is_parsed():
    result = parse_judge_response(
        make_response(verdict="CONTRADICTED"),
        make_request(),
    )

    assert result.verdict == JudgeVerdictType.CONTRADICTED


def test_insufficient_evidence_response_is_parsed():
    result = parse_judge_response(
        make_response(
            verdict="INSUFFICIENT_EVIDENCE",
            evidence_source_ids=[],
        ),
        make_request(),
    )

    assert result.verdict == JudgeVerdictType.INSUFFICIENT_EVIDENCE


def test_unknown_evidence_reference_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        parse_judge_response(
            make_response(evidence_source_ids=["LAW-UNKNOWN"]),
            make_request(),
        )


def test_mismatched_claim_id_is_rejected():
    with pytest.raises(ValueError, match="claim ID"):
        parse_judge_response(
            make_response(claim_id="CL-WRONG"),
            make_request(),
        )


def test_supported_verdict_requires_evidence():
    with pytest.raises(ValidationError):
        parse_judge_response(
            make_response(evidence_source_ids=[]),
            make_request(),
        )


def test_confidence_outside_range_is_rejected():
    with pytest.raises(ValidationError):
        parse_judge_response(
            make_response(confidence=1.5),
            make_request(),
        )


def test_invalid_json_is_rejected():
    with pytest.raises(ValidationError):
        parse_judge_response(
            "not valid json",
            make_request(),
        )
