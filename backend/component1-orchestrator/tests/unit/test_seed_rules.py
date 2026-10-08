"""The YA 2025/26 seed rule file loads and its values are internally consistent."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

SEED_FILE = (
    Path(__file__).resolve().parents[2] / "tax_calculation_agent/rules/ya_2025_26.yaml"
)


@pytest.fixture(scope="module")
def seed():
    return yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))


def rule(seed, key):
    (match,) = [r for r in seed["rules"] if r["rule_key"] == key]
    return match


def test_period_is_one_year_of_assessment(seed):
    assert seed["year_of_assessment"] == "2025/26"
    assert seed["period"] == {"start": date(2025, 4, 1), "end": date(2026, 3, 31)}


def test_rule_ids_unique_and_effective_from_ya_start(seed):
    ids = [r["rule_id"] for r in seed["rules"]]
    assert len(ids) == len(set(ids))
    for r in seed["rules"]:
        assert r["validity"]["valid_from"] == seed["period"]["start"]


def test_every_rule_cites_a_listed_source(seed):
    known = {s["id"] for s in seed["sources"]}
    for r in seed["rules"]:
        assert r["source"]["citation"]
        assert set(r["source"]["source_ids"]) <= known


def test_personal_relief(seed):
    amount = rule(seed, "personal_relief")["values"]["amount"]
    assert amount == {"amount": 1800000, "currency": "LKR", "period": "year_of_assessment"}


def test_bands_are_continuous_and_progressive(seed):
    bands = rule(seed, "individual_income_tax_slabs")["values"]["bands"]
    assert [b["order"] for b in bands] == list(range(1, len(bands) + 1))
    assert bands[0]["lower"] == 0
    assert bands[-1]["upper"] is None
    for prev, nxt in zip(bands, bands[1:], strict=False):
        assert nxt["lower"] == prev["upper"]
        assert nxt["rate_percent"] > prev["rate_percent"]


def test_bands_match_sanity_check_from_brief(seed):
    # CLAUDE.md §5: taxable income 2,500,000 -> 60,000 + 90,000 + 120,000 + 150,000 = 420,000
    bands = rule(seed, "individual_income_tax_slabs")["values"]["bands"]
    taxable = Decimal(2500000)
    per_band = [
        (min(taxable, Decimal(b["upper"])) - Decimal(b["lower"])) * Decimal(b["rate_percent"]) / 100
        for b in bands
        if b["upper"] is not None and taxable > b["lower"]
    ]
    assert per_band == [60000, 90000, 120000, 150000]
    assert sum(per_band) == 420000
