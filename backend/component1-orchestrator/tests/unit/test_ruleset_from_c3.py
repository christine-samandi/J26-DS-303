"""Building a RuleSet from C3's RetrievalOutput, and cross-checking it against the seed."""

import json
from decimal import Decimal as D

import pytest
from j26_contracts.retrieval import RetrievalOutput

from orchestrator.clients import MOCK_DIR
from tax_calculation_agent.ruleset import (
    RuleSetError,
    load_seed_rules,
    mid_year_changes,
    rule_conflicts,
    rules_from_retrieval,
)

SEED = load_seed_rules("2025/26")


def c3_data() -> dict:
    return json.loads((MOCK_DIR / "retrieval_ya2025_26.json").read_text(encoding="utf-8"))


def build(data: dict, ya: str = "2025/26"):
    return rules_from_retrieval(RetrievalOutput.model_validate(data), ya)


def test_c1_mock_matches_the_seed_rules():
    rules = build(c3_data())
    assert rules.personal_relief.amount == D(1800000)
    assert rules.personal_relief.ref.origin == "c3"
    assert rules.personal_relief.ref.rule_id == "RULE-PERSONAL_RELIEF-V4"
    assert rules.tax_bands.bands[-1].upper is None
    assert rule_conflicts(SEED, rules) == []
    assert mid_year_changes(rules) == []


def test_every_disagreement_is_reported():
    data = c3_data()
    data["results"][0]["values"]["amount"]["amount"] = 1200000
    data["results"][1]["values"]["bands"][1]["rate_percent"] = 12
    assert rule_conflicts(SEED, build(data)) == [
        "personal relief: seed 1800000, c3 1200000",
        "tax band 2: seed 1000000-1500000 @ 18%, c3 1000000-1500000 @ 12%",
    ]


def test_old_six_band_table_conflicts_on_band_count():
    six = [
        {"order": i + 1, "lower": lo, "upper": hi, "rate_percent": r}
        for i, (lo, hi, r) in enumerate(
            [
                (0, 500000, 6),
                (500000, 1000000, 12),
                (1000000, 1500000, 18),
                (1500000, 2000000, 24),
                (2000000, 2500000, 30),
                (2500000, None, 36),
            ]
        )
    ]
    data = c3_data()
    data["results"][1]["values"]["bands"] = six
    assert rule_conflicts(SEED, build(data)) == ["tax bands: seed has 5 bands, c3 has 6"]


def test_missing_rule_is_an_error():
    data = c3_data()
    del data["results"][1]
    with pytest.raises(RuleSetError, match="individual_income_tax_slabs.*found 0"):
        build(data)


def test_rules_not_in_force_on_the_date_are_ignored():
    data = c3_data()
    data["query"]["include_history"] = True
    old = json.loads(json.dumps(data["results"][0]))
    old.update(rank=3, rule_id="RULE-PERSONAL_RELIEF-V3")
    old["values"]["amount"]["amount"] = 1200000
    old["validity"].update(
        valid_from="2024-04-01",
        valid_to="2025-03-31",
        status="replaced",
        in_force_on_as_of_date=False,
        superseded_by="RULE-PERSONAL_RELIEF-V4",
    )
    data["results"].append(old)
    assert build(data).personal_relief.amount == D(1800000)


def test_query_date_outside_the_ya_is_an_error():
    with pytest.raises(RuleSetError, match="outside YA 2026/27"):
        build(c3_data(), "2026/27")


def test_rule_ending_inside_the_ya_is_flagged():
    data = c3_data()
    data["results"][0]["validity"].update(
        valid_to="2025-12-31", status="replaced", superseded_by="RULE-PERSONAL_RELIEF-V5"
    )
    assert mid_year_changes(build(data)) == [
        "RULE-PERSONAL_RELIEF-V4 ends on 2025-12-31, before the YA ends on 2026-03-31"
    ]


def test_invalid_bands_from_c3_are_an_error():
    data = c3_data()
    data["results"][1]["values"]["bands"][2]["lower"] = 1400000  # overlaps band 2
    with pytest.raises(RuleSetError, match="gap or overlap"):
        build(data)
