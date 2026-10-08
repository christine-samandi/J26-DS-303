"""The shared Pydantic contract models (contracts/j26_contracts/retrieval.py), as used by C3."""

import copy
import json

import pytest
from pydantic import ValidationError

from c3_common.config import get_settings
from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json

MOCK_DIR = get_settings().contracts_dir / "mock_data" / "retrieval_output_examples"


def _mocks(prefix):
    return sorted(p for p in MOCK_DIR.glob("*.json") if p.name.startswith(prefix))


def _load(name):
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _mocks("0"), ids=lambda p: p.name)
def test_output_mocks_round_trip(path):
    """Loading a mock into Pydantic and writing it back gives the identical JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert to_contract_json(RetrievalOutput.model_validate(data)) == data


@pytest.mark.parametrize("path", _mocks("request_"), ids=lambda p: p.name)
def test_request_mocks_round_trip(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert to_contract_json(RetrievalRequest.model_validate(data)) == data


def test_typed_access():
    out = RetrievalOutput.model_validate(_load("01_current_personal_relief.json"))
    rule = out.results[0]
    assert rule.values.amount.currency == "LKR"
    assert rule.validity.in_force_on(out.query.as_of_date)


# --- Bad data is rejected with a clear message ---------------------------------------------


def test_rejects_replaced_without_successor():
    bad = _load("02_point_in_time_with_history.json")
    bad["results"][0]["validity"]["superseded_by"] = None
    with pytest.raises(ValidationError, match="superseded_by"):
        RetrievalOutput.model_validate(bad)


def test_rejects_active_with_end_date():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["validity"]["valid_to"] = "2030-01-01"
    with pytest.raises(ValidationError, match="active rule"):
        RetrievalOutput.model_validate(bad)


def test_rejects_confidence_out_of_range():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["confidence"]["score"] = 1.5
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        RetrievalOutput.model_validate(bad)


def test_rejects_unknown_field():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["surprise"] = 1
    with pytest.raises(ValidationError, match="Extra inputs"):
        RetrievalOutput.model_validate(bad)


def test_rejects_unknown_enum_value():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["rule_type"] = "discount"
    with pytest.raises(ValidationError, match="rule_type"):
        RetrievalOutput.model_validate(bad)


def test_rejects_request_without_query():
    req = copy.deepcopy(_load("request_01_current_personal_relief.json"))
    for key in ("query_text", "rule_types"):
        req.pop(key)
    with pytest.raises(ValidationError, match="at least one"):
        RetrievalRequest.model_validate(req)


# --- Extra checks Pydantic adds on top of the JSON Schema -----------------------------------


def test_extra_check_in_force_flag_must_match_dates():
    bad = _load("02_point_in_time_with_history.json")
    bad["results"][1]["validity"]["in_force_on_as_of_date"] = True  # 2023 rule, 2021 query
    with pytest.raises(ValidationError, match="its dates say False"):
        RetrievalOutput.model_validate(bad)


def test_extra_check_end_before_start():
    bad = _load("02_point_in_time_with_history.json")
    bad["results"][0]["validity"]["valid_to"] = "2019-01-01"
    with pytest.raises(ValidationError, match="before valid_from"):
        RetrievalOutput.model_validate(bad)


def test_extra_check_ranks_in_order():
    bad = _load("02_point_in_time_with_history.json")
    bad["results"][0]["rank"] = 3
    with pytest.raises(ValidationError, match="ranked"):
        RetrievalOutput.model_validate(bad)


def test_extra_check_no_stale_rules_without_history():
    bad = _load("02_point_in_time_with_history.json")
    bad["query"]["include_history"] = False
    with pytest.raises(ValidationError, match="not in force"):
        RetrievalOutput.model_validate(bad)
