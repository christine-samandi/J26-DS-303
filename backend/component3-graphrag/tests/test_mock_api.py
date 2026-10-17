"""The mock /retrieve endpoint accepts contract requests and returns contract responses."""

import json

import pytest
from fastapi.testclient import TestClient

from c3_common.config import get_settings
from c3_common.contracts import OUTPUT_SCHEMA, contract_errors
from retrieval_api.app import app

client = TestClient(app)
MOCK_DIR = get_settings().contracts_dir / "mock_data" / "retrieval_output_examples"


def _request(name):
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def test_health():
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("name", sorted(p.name for p in MOCK_DIR.glob("request_*.json")))
def test_retrieve_returns_valid_contract(name):
    req = _request(name)
    res = client.post("/retrieve", json=req)
    assert res.status_code == 200, res.text
    body = res.json()
    assert contract_errors(body, OUTPUT_SCHEMA) == []
    assert body["request_id"] == req["request_id"]
    assert body["query"]["as_of_date"] == req["as_of_date"]


def test_point_in_time_returns_rule_in_force_then():
    req = _request("request_02_point_in_time_with_history.json")
    body = client.post("/retrieve", json=req).json()
    top = body["results"][0]
    assert top["validity"]["in_force_on_as_of_date"] is True
    assert top["validity"]["status"] == "replaced"


def test_without_history_only_in_force_rules_returned():
    req = _request("request_02_point_in_time_with_history.json")
    req["include_history"] = False
    body = client.post("/retrieve", json=req).json()
    assert body["results"]
    assert all(r["validity"]["in_force_on_as_of_date"] for r in body["results"])


def test_unknown_topic_returns_no_match():
    req = {
        "schema_version": "1.0.0",
        "request_id": "t-1",
        "query_text": "customs duty on cars",
        "as_of_date": "2026-10-08",
    }
    body = client.post("/retrieve", json=req).json()
    assert body["results"] == [] and "NO_MATCH" in body["warnings"]


def test_invalid_request_rejected():
    res = client.post("/retrieve", json={"request_id": "t-2"})
    assert res.status_code == 422


def _relief(as_of_date, include_history=False):
    req = {
        "schema_version": "1.0.0",
        "request_id": "t-relief",
        "query_text": "personal relief",
        "as_of_date": as_of_date,
        "include_history": include_history,
    }
    return client.post("/retrieve", json=req).json()


@pytest.mark.parametrize(
    "as_of_date,expected_rule",
    [
        ("2021-06-01", "RULE-PERSONAL_RELIEF-V2"),
        ("2024-06-01", "RULE-PERSONAL_RELIEF-V3"),
        ("2026-10-08", "RULE-PERSONAL_RELIEF-V4"),
    ],
)
def test_point_in_time_picks_version_in_force(as_of_date, expected_rule):
    body = _relief(as_of_date)
    assert [r["rule_id"] for r in body["results"]] == [expected_rule]
    assert body["results"][0]["validity"]["in_force_on_as_of_date"] is True


def test_date_before_any_version_warns():
    body = _relief("2019-01-01")
    assert body["results"] == []
    assert "NO_RULE_IN_FORCE_ON_DATE" in body["warnings"]
    assert "AS_OF_DATE_BEFORE_CORPUS" in body["warnings"]


def test_invalid_request_explains_which_field():
    res = client.post(
        "/retrieve",
        json={
            "schema_version": "1.0.0",
            "request_id": "t-3",
            "query_text": "x",
            "as_of_date": "2026-10-08",
            "top_k": 50,
        },
    )
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-1] == "top_k"
