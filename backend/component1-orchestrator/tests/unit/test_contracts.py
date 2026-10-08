"""C1 can import the shared contract models and read every C3 retrieval mock."""

import json
from pathlib import Path

import pytest
from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest

MOCK_DIR = Path(__file__).resolve().parents[4] / "contracts/mock_data/retrieval_output_examples"
REQUEST_MOCKS = sorted(MOCK_DIR.glob("request_*.json"))
OUTPUT_MOCKS = sorted(p for p in MOCK_DIR.glob("*.json") if not p.name.startswith("request_"))


def test_mocks_found():
    assert REQUEST_MOCKS and OUTPUT_MOCKS


@pytest.mark.parametrize("path", REQUEST_MOCKS, ids=lambda p: p.name)
def test_retrieval_request_mock_is_valid(path):
    RetrievalRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", OUTPUT_MOCKS, ids=lambda p: p.name)
def test_retrieval_output_mock_is_valid(path):
    RetrievalOutput.model_validate(json.loads(path.read_text(encoding="utf-8")))
