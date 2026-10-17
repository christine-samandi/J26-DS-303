"""The generated JSON Schema files: up to date, every mock matches, bad data is rejected."""

import copy
import json

import pytest

from c3_common.config import get_settings
from c3_common.contracts import OUTPUT_SCHEMA, REQUEST_SCHEMA, contract_errors
from j26_contracts.export import stale_files

MOCK_DIR = get_settings().contracts_dir / "mock_data" / "retrieval_output_examples"
MOCK_FILES = sorted(MOCK_DIR.glob("*.json"))


def _load(name):
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def test_mocks_exist():
    assert len(MOCK_FILES) >= 4


@pytest.mark.parametrize("path", MOCK_FILES, ids=lambda p: p.name)
def test_mock_matches_contract(path):
    schema = REQUEST_SCHEMA if path.name.startswith("request_") else OUTPUT_SCHEMA
    assert contract_errors(json.loads(path.read_text(encoding="utf-8")), schema) == []


def test_schema_files_generated_from_current_models():
    """contracts/schemas/*.json must be regenerated whenever the Pydantic models change."""
    stale = stale_files()
    assert stale == [], f"out of date: {stale} — run: python -m j26_contracts.export"


def test_replaced_rule_must_name_successor():
    bad = _load("02_point_in_time_with_history.json")
    bad["results"][0]["validity"]["superseded_by"] = None
    assert contract_errors(bad, OUTPUT_SCHEMA)


def test_active_rule_cannot_have_end_date():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["validity"]["valid_to"] = "2030-01-01"
    assert contract_errors(bad, OUTPUT_SCHEMA)


def test_confidence_out_of_range_rejected():
    bad = _load("01_current_personal_relief.json")
    bad["results"][0]["confidence"]["score"] = 1.5
    assert contract_errors(bad, OUTPUT_SCHEMA)


def test_request_needs_some_query():
    req = copy.deepcopy(_load("request_01_current_personal_relief.json"))
    for key in ("query_text", "rule_types"):
        req.pop(key)
    assert contract_errors(req, REQUEST_SCHEMA)
