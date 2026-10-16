"""Why: the contract is shared by 4 components - this test fails the moment code or mocks drift from it."""
import json
from pathlib import Path

import jsonschema

from api.schema import OCROutput

REPO = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO / "contracts/schemas/ocr_output_schema.json").read_text(encoding="utf-8"))
MOCKS = sorted((REPO / "contracts/mock_data/ocr_output_examples").glob("*.json"))


def test_mocks_exist():
    assert MOCKS, "add at least one mock payload"


def test_mocks_match_json_schema_and_pydantic():
    for m in MOCKS:
        data = json.loads(m.read_text(encoding="utf-8"))
        jsonschema.validate(data, SCHEMA)   # what Components 1, 3, 4 rely on
        OCROutput.model_validate(data)      # what our FastAPI will enforce
