"""Loading the seed YAML into a typed RuleSet."""

import copy
from datetime import date
from decimal import Decimal

import pytest
import yaml
from pydantic import ValidationError

from tax_calculation_agent import ruleset
from tax_calculation_agent.ruleset import RuleSetError, load_seed_rules, ya_period

YA = "2025/26"


@pytest.fixture(scope="module")
def seed_data():
    return yaml.safe_load(ruleset.seed_file(YA).read_text(encoding="utf-8"))


@pytest.fixture
def load_modified(tmp_path, monkeypatch, seed_data):
    """Write a changed copy of the seed file to a temp rules dir and load it."""

    def _load(change):
        data = copy.deepcopy(seed_data)
        change(data)
        (tmp_path / "ya_2025_26.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        monkeypatch.setattr(ruleset, "RULES_DIR", tmp_path)
        return load_seed_rules(YA)

    return _load


def slabs(data):
    return next(r for r in data["rules"] if r["rule_key"] == "individual_income_tax_slabs")


# --- the real seed file -----------------------------------------------------


def test_loads_ya_2025_26():
    rs = load_seed_rules(YA)
    assert rs.year_of_assessment == YA
    assert (rs.period_start, rs.period_end) == (date(2025, 4, 1), date(2026, 3, 31))
    assert rs.personal_relief.amount == Decimal("1800000")
    assert [(b.lower, b.upper, b.rate_percent) for b in rs.tax_bands.bands] == [
        (Decimal(0), Decimal(1000000), Decimal(6)),
        (Decimal(1000000), Decimal(1500000), Decimal(18)),
        (Decimal(1500000), Decimal(2000000), Decimal(24)),
        (Decimal(2000000), Decimal(2500000), Decimal(30)),
        (Decimal(2500000), None, Decimal(36)),
    ]


def test_all_numbers_are_decimal():
    rs = load_seed_rules(YA)
    assert type(rs.personal_relief.amount) is Decimal
    for b in rs.tax_bands.bands:
        assert type(b.lower) is Decimal
        assert b.upper is None or type(b.upper) is Decimal
        assert type(b.rate_percent) is Decimal


def test_rules_carry_citation_refs():
    rs = load_seed_rules(YA)
    for ref in (rs.personal_relief.ref, rs.tax_bands.ref):
        assert ref.origin == "seed"
        assert ref.rule_id.startswith("SEED-")
        assert ref.citation
        assert ref.valid_from == date(2025, 4, 1)


def test_ruleset_is_immutable():
    rs = load_seed_rules(YA)
    with pytest.raises(ValidationError):
        rs.personal_relief.amount = Decimal(0)


def test_same_file_loads_identically():
    assert load_seed_rules(YA) == load_seed_rules(YA)


# --- year of assessment -----------------------------------------------------


def test_ya_period():
    assert ya_period("2025/26") == (date(2025, 4, 1), date(2026, 3, 31))
    assert ya_period("1999/00") == (date(1999, 4, 1), date(2000, 3, 31))


@pytest.mark.parametrize("bad", ["2025", "2025-26", "2025/27", "25/26", ""])
def test_bad_ya_format_rejected(bad):
    with pytest.raises(RuleSetError):
        ya_period(bad)


def test_missing_seed_file():
    with pytest.raises(RuleSetError, match="no seed rules"):
        load_seed_rules("2019/20")


# --- broken data is rejected ------------------------------------------------


def test_decimal_from_text_not_float(load_modified):
    def change(d):
        slabs(d)["values"]["bands"][0]["rate_percent"] = 6.1

    assert load_modified(change).tax_bands.bands[0].rate_percent == Decimal("6.1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: slabs(d)["values"]["bands"][1].update(lower=1000001), "gap or overlap"),
        (lambda d: slabs(d)["values"]["bands"][-1].update(upper=9000000), "no upper limit"),
        (lambda d: slabs(d)["values"]["bands"][0].update(lower=1), "start at 0"),
        (lambda d: slabs(d)["values"]["bands"][0].update(order=9), "1..n"),
        (lambda d: slabs(d)["values"]["bands"][2].update(rate_percent=101), "0..100"),
        (lambda d: slabs(d)["values"].update(bands=[]), "at least one band"),
        (lambda d: d["rules"].pop(0), "exactly one 'personal_relief'"),
        (lambda d: d["rules"].append(copy.deepcopy(d["rules"][0])), "exactly one"),
        (lambda d: d["rules"][0]["values"]["amount"].update(amount=-1), "negative"),
        (lambda d: d["rules"][0]["values"]["amount"].update(amount="lots"), "expected a number"),
        (lambda d: d["rules"][0]["values"]["amount"].update(amount=True), "expected a number"),
        (lambda d: d["rules"][0]["values"]["amount"].update(amount=float("nan")), "finite"),
        (lambda d: d["rules"][0]["values"]["amount"].update(currency="USD"), "LKR"),
        (lambda d: d["rules"][0]["validity"].update(valid_from=date(2025, 6, 1)), "whole YA"),
        (lambda d: d["rules"][0]["validity"].update(valid_to=date(2025, 12, 31)), "whole YA"),
        (lambda d: d["period"].update(start=date(2025, 1, 1)), "period must be"),
        (lambda d: d.update(year_of_assessment="2024/25"), "file is for YA"),
        (lambda d: d.update(schema="other/1"), "expected schema"),
    ],
)
def test_inconsistent_seed_rejected(load_modified, change, message):
    with pytest.raises(RuleSetError, match=message):
        load_modified(change)
