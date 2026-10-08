"""Typed tax rules for one year of assessment (YA), and the loader for the local seed files.

The engine only ever sees a `RuleSet`. Today it is built from the seed YAML
(`rules/ya_YYYY_YY.yaml`); later a second builder will make one from C3's `RetrievalOutput`,
so seed and C3 values can be compared field by field.

All money and rates are `Decimal`, converted from the YAML text with `Decimal(str(value))`
so no float ever enters the engine.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

RULES_DIR = Path(__file__).resolve().parent / "rules"
SEED_SCHEMA = "c1-seed-rules/1"
_YA_PATTERN = re.compile(r"^(\d{4})/(\d{2})$")

RuleOrigin = Literal["seed", "c3"]


class RuleSetError(ValueError):
    """The rule data is missing or inconsistent. The engine must not run on it."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RuleRef(_Frozen):
    """Where a rule came from: what a CalcStep cites."""

    rule_id: str
    citation: str
    origin: RuleOrigin
    valid_from: date
    valid_to: date | None


class PersonalRelief(_Frozen):
    amount: Decimal
    ref: RuleRef

    @field_validator("amount")
    @classmethod
    def _not_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("personal relief cannot be negative")
        return v


class TaxBand(_Frozen):
    order: int
    lower: Decimal
    upper: Decimal | None  # None = no upper limit
    rate_percent: Decimal


class TaxBands(_Frozen):
    bands: tuple[TaxBand, ...]
    ref: RuleRef

    @model_validator(mode="after")
    def _continuous_and_progressive(self) -> TaxBands:
        b = self.bands
        if not b:
            raise ValueError("at least one band is required")
        if [x.order for x in b] != list(range(1, len(b) + 1)):
            raise ValueError("band order must run 1..n")
        if b[0].lower != 0:
            raise ValueError("first band must start at 0")
        if b[-1].upper is not None:
            raise ValueError("last band must have no upper limit")
        for prev, nxt in zip(b, b[1:], strict=False):
            if prev.upper is None or nxt.lower != prev.upper:
                raise ValueError(f"gap or overlap between bands {prev.order} and {nxt.order}")
        for x in b:
            if x.upper is not None and x.upper <= x.lower:
                raise ValueError(f"band {x.order} upper must be above lower")
            if not 0 <= x.rate_percent <= 100:
                raise ValueError(f"band {x.order} rate must be within 0..100")
        return self


class RuleSet(_Frozen):
    """Everything the v1 engine needs for one YA: personal relief and the rate bands."""

    year_of_assessment: str
    period_start: date
    period_end: date
    personal_relief: PersonalRelief
    tax_bands: TaxBands


def ya_period(year_of_assessment: str) -> tuple[date, date]:
    """'2025/26' -> (2025-04-01, 2026-03-31). A YA runs 1 April to 31 March."""
    m = _YA_PATTERN.match(year_of_assessment)
    if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
        raise RuleSetError(
            f"year of assessment must look like '2025/26', got {year_of_assessment!r}"
        )
    start = int(m.group(1))
    return date(start, 4, 1), date(start + 1, 3, 31)


def seed_file(year_of_assessment: str) -> Path:
    ya_period(year_of_assessment)  # validates the format
    return RULES_DIR / f"ya_{year_of_assessment.replace('/', '_')}.yaml"


def load_seed_rules(year_of_assessment: str) -> RuleSet:
    """Load and check the local seed rules for a YA. Raises RuleSetError if anything is off."""
    path = seed_file(year_of_assessment)
    if not path.is_file():
        raise RuleSetError(f"no seed rules for YA {year_of_assessment} ({path.name})")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return _build(data, year_of_assessment)
    except (KeyError, TypeError, ValueError) as e:
        raise RuleSetError(f"{path.name}: {e}") from e


def _build(data: dict, year_of_assessment: str) -> RuleSet:
    if data.get("schema") != SEED_SCHEMA:
        raise RuleSetError(f"expected schema {SEED_SCHEMA!r}, got {data.get('schema')!r}")
    if data["year_of_assessment"] != year_of_assessment:
        raise RuleSetError(f"file is for YA {data['year_of_assessment']}")
    start, end = ya_period(year_of_assessment)
    if (data["period"]["start"], data["period"]["end"]) != (start, end):
        raise RuleSetError(f"period must be {start} to {end}")

    relief = _one_rule(data, "personal_relief")
    slabs = _one_rule(data, "individual_income_tax_slabs")
    for r in (relief, slabs):
        _check_in_force_for_ya(r, start, end)

    amount = relief["values"]["amount"]
    if amount["currency"] != "LKR":
        raise RuleSetError("personal relief must be in LKR")

    return RuleSet(
        year_of_assessment=year_of_assessment,
        period_start=start,
        period_end=end,
        personal_relief=PersonalRelief(amount=_dec(amount["amount"]), ref=_ref(relief)),
        tax_bands=TaxBands(
            bands=tuple(
                TaxBand(
                    order=b["order"],
                    lower=_dec(b["lower"]),
                    upper=None if b["upper"] is None else _dec(b["upper"]),
                    rate_percent=_dec(b["rate_percent"]),
                )
                for b in slabs["values"]["bands"]
            ),
            ref=_ref(slabs),
        ),
    )


def _one_rule(data: dict, rule_key: str) -> dict:
    matches = [r for r in data["rules"] if r["rule_key"] == rule_key]
    if len(matches) != 1:
        raise RuleSetError(f"expected exactly one {rule_key!r} rule, found {len(matches)}")
    return matches[0]


def _check_in_force_for_ya(rule: dict, start: date, end: date) -> None:
    """Seed rules must cover the whole YA; a mid-year change is not supported in a seed file."""
    v = rule["validity"]
    if v["valid_from"] > start or (v["valid_to"] is not None and v["valid_to"] < end):
        raise RuleSetError(f"{rule['rule_id']} does not cover the whole YA {start} to {end}")


def _ref(rule: dict) -> RuleRef:
    return RuleRef(
        rule_id=rule["rule_id"],
        citation=rule["source"]["citation"],
        origin="seed",
        valid_from=rule["validity"]["valid_from"],
        valid_to=rule["validity"]["valid_to"],
    )


def _dec(value: object) -> Decimal:
    """YAML number -> Decimal via its text form. Text, booleans, NaN and infinity are rejected."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuleSetError(f"expected a number, got {value!r}")
    result = Decimal(str(value))
    if not result.is_finite():
        raise RuleSetError(f"expected a finite number, got {value!r}")
    return result
