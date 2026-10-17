"""Load the retrieval contract schemas and validate payloads against them.

The schemas live in the repo-level `contracts/schemas/` folder and are shared with C1 and C4.
Every response C3 sends must pass `validate_output`.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from c3_common.config import get_settings

REQUEST_SCHEMA = "retrieval_request.schema.json"
OUTPUT_SCHEMA = "retrieval_output.schema.json"


@cache
def load_schema(name: str) -> dict[str, Any]:
    path = get_settings().contracts_dir / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def contract_errors(payload: dict[str, Any], schema_name: str) -> list[str]:
    """Return a list of readable validation errors (empty list = valid)."""
    errors = sorted(_validator(schema_name).iter_errors(payload), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def validate_request(payload: dict[str, Any]) -> None:
    errors = contract_errors(payload, REQUEST_SCHEMA)
    if errors:
        raise ValueError("Invalid retrieval request: " + "; ".join(errors))


def validate_output(payload: dict[str, Any]) -> None:
    errors = contract_errors(payload, OUTPUT_SCHEMA)
    if errors:
        raise ValueError("Invalid retrieval output: " + "; ".join(errors))
