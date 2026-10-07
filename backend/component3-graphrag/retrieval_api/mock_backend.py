"""Mock retrieval backend that serves the contract examples.

Lets C1 and C4 call a real HTTP endpoint before the knowledge graph exists. It picks the
closest mock response for the request, then rewrites the echo fields (request_id, query,
generated_at) so the response is consistent with what was asked. Values are placeholders.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from c3_common.config import get_settings

MOCK_DIR_NAME = "retrieval_output_examples"
_SLAB_WORDS = ("slab", "rate", "band", "bracket")


@lru_cache(maxsize=1)
def _mocks() -> dict[str, dict[str, Any]]:
    folder = get_settings().contracts_dir / "mock_data" / MOCK_DIR_NAME
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("0*.json"))
    }


def _all_mock_rules() -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for mock in _mocks().values():
        for rule in mock["results"]:
            seen.setdefault(rule["rule_id"], rule)
    return list(seen.values())


def _pick_template(req: dict[str, Any]) -> dict[str, Any]:
    mocks = _mocks()
    text = req.get("query_text", "").lower()
    types = set(req.get("rule_types", []))
    if "tax_slab" in types or "tax_rate" in types or any(w in text for w in _SLAB_WORDS):
        return mocks["03_tax_slabs_low_confidence"]
    if "relief" in types or "relief" in text:
        if req["as_of_date"] < "2025-04-01":
            return mocks["02_point_in_time_with_history"]
        return mocks["01_current_personal_relief"]
    return mocks["04_no_match"]


def retrieve(req: dict[str, Any]) -> dict[str, Any]:
    """Build a contract-shaped response for an already-validated request."""
    started = datetime.now(UTC)
    if "rule_ids" in req:
        wanted = set(req["rule_ids"])
        results = [copy.deepcopy(r) for r in _all_mock_rules() if r["rule_id"] in wanted]
        out = copy.deepcopy(_mocks()["01_current_personal_relief"])
        out["results"] = results
        out["warnings"] = [] if results else ["NO_MATCH"]
    else:
        out = copy.deepcopy(_pick_template(req))

    if not req.get("include_history", False):
        out["results"] = [r for r in out["results"] if r["validity"]["in_force_on_as_of_date"]]
    top_k = req.get("top_k", 5)
    out["results"] = out["results"][:top_k]
    for i, rule in enumerate(out["results"], start=1):
        rule["rank"] = i
    if not out["results"] and "NO_MATCH" not in out["warnings"]:
        out["warnings"] = ["NO_MATCH", *out["warnings"]]

    query_echo = {
        k: req[k]
        for k in ("query_text", "income_types", "taxpayer_category", "rule_types", "rule_ids")
        if k in req
    }
    query_echo.update(
        as_of_date=req["as_of_date"],
        top_k=top_k,
        include_history=req.get("include_history", False),
    )
    out["request_id"] = req["request_id"]
    out["query"] = query_echo
    out["generated_at"] = datetime.now(UTC).isoformat()
    out["metadata"]["latency_ms"] = round((datetime.now(UTC) - started).total_seconds() * 1000, 2)
    return out
