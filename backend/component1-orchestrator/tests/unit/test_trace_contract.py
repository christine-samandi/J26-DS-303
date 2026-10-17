"""The proposed pipeline_trace contract: mocks are valid, and the extra checks bite."""

import copy
import json
from decimal import Decimal as D
from pathlib import Path

import pytest
from j26_contracts import export
from j26_contracts.trace import PipelineTrace
from pydantic import ValidationError

MOCK_DIR = Path(__file__).resolve().parents[4] / "contracts/mock_data/pipeline_trace_examples"
MOCKS = sorted(MOCK_DIR.glob("*.json"))


def load(name: str) -> dict:
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def test_schema_files_are_up_to_date():
    assert export.stale_files() == []


@pytest.mark.parametrize("path", MOCKS, ids=lambda p: p.name)
def test_mock_is_valid_and_round_trips(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = PipelineTrace.model_validate(data)
    assert trace.model_dump(mode="json", exclude_none=True) == data


def test_money_is_exact_decimal():
    trace = PipelineTrace.model_validate(load("trace_correct.json"))
    assert (trace.result.total_tax, trace.result.tax_payable) == (D(96000), D(46000))
    assert isinstance(trace.steps[0].result, D)


def test_wrong_band_mock_disagrees_only_with_its_rule():
    wrong = PipelineTrace.model_validate(load("trace_injected_error_wrong_band.json"))
    band2 = next(s for s in wrong.steps if s.step_id == "S3_BAND_2")
    assert band2.inputs["rate_percent"] == 12  # the seed/C3 rule says 18
    assert band2.result == band2.inputs["amount_in_band"] * band2.inputs["rate_percent"] / 100
    assert wrong.result.total_tax == D(84000)


DELETE = object()


def broken(**changes):
    data = copy.deepcopy(load("trace_correct.json"))
    for path, value in changes.items():
        *keys, last = path.split(".")
        target = data
        for k in keys:
            target = target[int(k)] if k.isdigit() else target[k]
        if value is DELETE:
            del target[int(last) if last.isdigit() else last]
        else:
            target[int(last) if last.isdigit() else last] = value
    return data


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"steps.0.citation_refs": ["RULE-INVENTED-V1"]}, "citations not in rules_used"),
        ({"claims.0.step_id": "S9_UNKNOWN"}, "claims about unknown steps"),
        ({"claims.0": DELETE}, "steps with no claim: \\['S1_RELIEF'\\]"),
        ({"regeneration": DELETE}, "regeneration is required on attempt 2"),
        ({"attempt": 3}, "less than or equal to 2"),
        ({"result.refund": 10}, "cannot both be above 0"),
        ({"inputs.0.document_id": DELETE}, "document_id is required"),
        ({"result.total_tax": 96000.123}, "decimal places"),
        ({"request_id": "0001"}, "String should match pattern"),
        ({"inputs.0.employee_name": "A. Perera"}, "Extra inputs are not permitted"),
        ({"explanation.rewrite_status": DELETE}, "rewrite_status\n  Field required"),
        ({"explanation.rewrite_status": "skipped"}, "Input should be 'not_requested'"),
        ({"explanation.rewrite_status": "rejected"}, "used_fallback must be true exactly"),
        (
            {"explanation.used_fallback": True, "explanation.rewrite_status": "failed"},
            "used_fallback must be true exactly",
        ),
        (
            {"explanation.hallucination_flag": True, "explanation.rewrite_status": "accepted"},
            "hallucination_flag needs rewrite_status 'rejected'",
        ),
    ],
)
def test_extra_checks(changes, message):
    with pytest.raises(ValidationError, match=message):
        PipelineTrace.model_validate(broken(**changes))


@pytest.mark.parametrize("path", MOCKS, ids=lambda p: p.name)
def test_mocks_carry_rewrite_status(path):
    explanation = json.loads(path.read_text(encoding="utf-8"))["explanation"]
    assert explanation["rewrite_status"] == "not_requested"  # no mock has an explain LLM call


@pytest.mark.parametrize(
    ("status", "used_fallback", "hallucination_flag"),
    [
        ("not_requested", False, False),
        ("failed", False, False),  # the LLM call raised: template shown, nothing rejected
        ("rejected", True, False),  # malformed rewrite
        ("rejected", True, True),  # rewrite changed a number or citation
        ("accepted", False, False),
    ],
)
def test_every_rewrite_status_is_accepted_with_consistent_flags(
    status, used_fallback, hallucination_flag
):
    data = broken(
        **{
            "explanation.rewrite_status": status,
            "explanation.used_fallback": used_fallback,
            "explanation.hallucination_flag": hallucination_flag,
        }
    )
    assert PipelineTrace.model_validate(data).explanation.rewrite_status == status
