"""Every seed rule file matches C3's hand-verified ground truth (component3-graphrag/ground_truth).

The CSVs are compiled from the official Acts; the seed files must not drift from them.
"""

import csv
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

from tax_calculation_agent.ruleset import load_seed_rules

GROUND_TRUTH = Path(__file__).resolve().parents[3] / "component3-graphrag/ground_truth"
# seed YA -> (personal relief version, tax band version) in the ground truth
VERSIONS = {"2024/25": ("V3", "B3"), "2025/26": ("V4", "B4")}


def rows(name: str) -> list[dict]:
    with (GROUND_TRUTH / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def day(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


@pytest.mark.parametrize("ya", VERSIONS)
def test_seed_values_and_dates_match_ground_truth(ya):
    relief_version, band_version = VERSIONS[ya]
    rs = load_seed_rules(ya)

    (relief,) = [r for r in rows("personal_relief_versions.csv") if r["version"] == relief_version]
    assert relief["verified"] == "yes"
    assert rs.personal_relief.amount == D(relief["amount_lkr"])
    assert rs.personal_relief.ref.valid_from == day(relief["valid_from"])
    assert rs.personal_relief.ref.valid_to == day(relief["valid_to"])

    bands = [r for r in rows("individual_tax_bands.csv") if r["version"] == band_version]
    assert all(b["verified"] == "yes" for b in bands)
    assert [(b.order, b.lower, b.upper, b.rate_percent) for b in rs.tax_bands.bands] == [
        (
            int(b["band_order"]),
            D(b["lower_lkr"]),
            D(b["upper_lkr"]) if b["upper_lkr"] else None,
            D(b["rate_percent"]),
        )
        for b in bands
    ]
    assert rs.tax_bands.ref.valid_from == day(bands[0]["valid_from"])
    assert rs.tax_bands.ref.valid_to == day(bands[0]["valid_to"])


def test_ya_2024_25_bands_sanity_check():
    # taxable income 2,500,000 -> 30,000 + 60,000 + 90,000 + 120,000 + 150,000 = 450,000
    bands = load_seed_rules("2024/25").tax_bands.bands
    taxable = D(2500000)
    per_band = [
        (min(taxable, b.upper) - b.lower) * b.rate_percent / 100
        for b in bands
        if b.upper is not None and taxable > b.lower
    ]
    assert per_band == [30000, 60000, 90000, 120000, 150000]
    assert sum(per_band) == 450000
