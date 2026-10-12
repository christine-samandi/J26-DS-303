"""The hand-built ground truth must be internally consistent before anything is scored on it."""

import csv
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path

import pytest

from data_collection.sources import load_registry

GT = Path(__file__).resolve().parents[1] / "ground_truth"


def _rows(name):
    with (GT / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _d(value):
    return date.fromisoformat(value) if value else None


RELIEF = _rows("personal_relief_versions.csv")
BANDS = _rows("individual_tax_bands.csv")
BAND_VERSIONS = [
    (version, list(rows)) for version, rows in groupby(BANDS, key=lambda r: r["version"])
]


@pytest.mark.parametrize(
    "periods",
    [
        [(_d(r["valid_from"]), _d(r["valid_to"])) for r in RELIEF],
        [(_d(rows[0]["valid_from"]), _d(rows[0]["valid_to"])) for _, rows in BAND_VERSIONS],
    ],
    ids=["relief", "bands"],
)
def test_versions_are_contiguous_without_gaps_or_overlaps(periods):
    for (_, end), (start, _) in zip(periods, periods[1:], strict=False):
        assert end is not None and start == end + timedelta(days=1)
    assert periods[-1][1] is None  # the latest version is still in force


def test_every_source_is_in_the_corpus_registry():
    known = {d.document_id for d in load_registry().documents}
    assert {r["source_document_id"] for r in RELIEF + BANDS} <= known


def test_relief_amounts():
    assert [int(r["amount_lkr"]) for r in RELIEF] == [500_000, 3_000_000, 1_200_000, 1_800_000]


@pytest.mark.parametrize("version,rows", BAND_VERSIONS, ids=[v for v, _ in BAND_VERSIONS])
def test_bands_are_contiguous_and_open_ended(version, rows):
    assert [int(r["band_order"]) for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["lower_lkr"] == "0"
    for a, b in zip(rows, rows[1:], strict=False):
        assert a["upper_lkr"] == b["lower_lkr"]
    assert rows[-1]["upper_lkr"] == ""
    assert len({(r["valid_from"], r["valid_to"]) for r in rows}) == 1


def test_bands_2025_drop_the_12_percent_step():
    b4 = dict(BAND_VERSIONS)["B4"]
    assert [int(r["rate_percent"]) for r in b4] == [6, 18, 24, 30, 36]


def test_every_row_is_verified_against_the_acts():
    pending = {r["version"] for r in RELIEF + BANDS if r["verified"] != "yes"}
    assert pending == set()


def _fixed_amounts(rows):
    """Tax on the full width of each lower band: the "Rs. X plus" figures printed in the Act."""
    total, out = 0, []
    for r in rows[:-1]:
        total += (int(r["upper_lkr"]) - int(r["lower_lkr"])) * int(r["rate_percent"]) // 100
        out.append(total)
    return out


def test_2017_bands_reproduce_the_fixed_amounts_printed_in_the_act():
    # Act No. 24 of 2017, First Schedule para 1(1): "Rs. 24,000 plus 8% of the amount ..."
    b1 = dict(BAND_VERSIONS)["B1"]
    assert _fixed_amounts(b1) == [24_000, 72_000, 144_000, 240_000, 360_000]
