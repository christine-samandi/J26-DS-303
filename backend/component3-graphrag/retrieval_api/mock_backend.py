"""Mock retrieval backend built from the contract examples.

Lets C1 and C4 call a real HTTP endpoint before the knowledge graph exists. The mock rules
(personal relief versions, a slab table) come from contracts/mock_data; their values are
placeholders. The point-in-time logic is real: for whatever `as_of_date` is asked, each rule's
`in_force_on_as_of_date` is recomputed from its valid_from / valid_to dates, so asking about
2021, 2024 or today returns different versions, just like the real system will.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from functools import cache
from typing import Any

from c3_common.config import get_settings
from j26_contracts.retrieval import (
    SCHEMA_VERSION,
    Metadata,
    QueryEcho,
    RetrievalOutput,
    RetrievalRequest,
    RetrievedRule,
)

MOCK_DIR_NAME = "retrieval_output_examples"
SLAB_TEMPLATE = "03_tax_slabs_low_confidence"
_SLAB_WORDS = ("slab", "rate", "band", "bracket")


@cache
def _raw_mocks() -> dict[str, dict[str, Any]]:
    folder = get_settings().contracts_dir / "mock_data" / MOCK_DIR_NAME
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("0*.json"))
    }


def _all_rules() -> list[RetrievedRule]:
    """Every distinct rule in the mock files, as fresh model objects (safe to modify)."""
    seen: dict[str, dict[str, Any]] = {}
    for mock in _raw_mocks().values():
        for rule in mock["results"]:
            seen.setdefault(rule["rule_id"], rule)
    return [RetrievedRule.model_validate(r) for r in seen.values()]


def _candidates(req: RetrievalRequest) -> tuple[list[RetrievedRule], list[str]]:
    """Pick candidate rules for the request, plus any warnings that come with them."""
    if req.rule_ids:
        return [r for r in _all_rules() if r.rule_id in req.rule_ids], []

    text = (req.query_text or "").lower()
    types = set(req.rule_types or [])
    if types & {"tax_slab", "tax_rate"} or any(w in text for w in _SLAB_WORDS):
        template = _raw_mocks()[SLAB_TEMPLATE]
        rules = [RetrievedRule.model_validate(r) for r in template["results"]]
        return rules, [w for w in template["warnings"] if w != "LOW_CONFIDENCE"]
    if "relief" in types or "relief" in text:
        return [r for r in _all_rules() if r.rule_key == "personal_relief"], []
    return [], ["NO_MATCH", "PARTIAL_CORPUS_COVERAGE"]


def _apply_date(rule: RetrievedRule, day: date) -> None:
    """Recompute whether the rule was in force on `day`, and adjust its mock confidence."""
    validity, confidence = rule.validity, rule.confidence
    was_in_force = validity.in_force_on_as_of_date
    now_in_force = validity.in_force_on(day)
    validity.in_force_on_as_of_date = now_in_force
    confidence.components.temporal_match = 1.0 if now_in_force else 0.0
    if now_in_force and not was_in_force:
        confidence.score, confidence.level = 0.9, "high"
    elif was_in_force and not now_in_force:
        confidence.score, confidence.level = min(confidence.score, 0.4), "low"


def retrieve(req: RetrievalRequest) -> RetrievalOutput:
    started = time.perf_counter()
    pool, warnings = _candidates(req)
    day = req.as_of_date

    for rule in pool:
        _apply_date(rule, day)
    # Rules in force on the date first, then older/newer versions in date order.
    pool.sort(key=lambda r: (not r.validity.in_force_on_as_of_date, r.validity.valid_from))

    results = (
        pool if req.include_history else [r for r in pool if r.validity.in_force_on_as_of_date]
    )
    results = results[: req.top_k]
    for rank, rule in enumerate(results, start=1):
        rule.rank = rank

    if pool and not any(r.validity.in_force_on_as_of_date for r in pool):
        warnings.append("NO_RULE_IN_FORCE_ON_DATE")
        if day < min(r.validity.valid_from for r in pool):
            warnings.append("AS_OF_DATE_BEFORE_CORPUS")
    if not pool and req.rule_ids:
        warnings.append("NO_MATCH")
    if any(r.validity.in_force_on_as_of_date and r.confidence.level == "low" for r in results):
        warnings.append("LOW_CONFIDENCE")

    echo = req.model_dump(
        include={"query_text", "income_types", "taxpayer_category", "rule_types", "rule_ids"},
        exclude_unset=True,
    )
    return RetrievalOutput(
        schema_version=SCHEMA_VERSION,
        request_id=req.request_id,
        generated_at=datetime.now(UTC),
        query=QueryEcho(
            **echo, as_of_date=day, top_k=req.top_k, include_history=req.include_history
        ),
        results=results,
        warnings=list(dict.fromkeys(warnings)),  # remove duplicates, keep order
        metadata=Metadata(
            retrieval_mode="mock",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            corpus_version="mock-2026-10-08",
            embedding_model="none (mock)",
            candidates_considered=len(pool),
            confidence_method="mock",
        ),
    )
