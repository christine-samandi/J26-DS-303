"""Fiscal validation and self-correction of payslip amounts (Gap 7; own algorithm).

Why this module exists:
OCR confidence says how sure the engine is about the SHAPES it read, not whether the number
makes sense. A payslip, however, must obey arithmetic:
    net pay = gross pay - total deductions            (always true - a HARD rule)
    gross   = basic + allowances                      (true unless other earnings exist - SOFT)
    total deductions = EPF(employee) + APIT           (true unless other deductions exist - SOFT)
    EPF employee = 8%, EPF employer = 12% (15% at some employers), ETF = 3% of EPF-liable
    earnings (basic, basic + COLA, or gross - SOFT)
This module (1) checks every rule whose fields were found, and (2) when a rule fails, tries the
most likely OCR misreadings of each amount (one extra digit, one dropped digit, one look-alike
digit swap such as 3<->8, 1<->7) and keeps a change only if it makes the arithmetic consistent
and is the single best explanation. Corrected fields are always flagged for user review.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

TOL = 1.0   # rupees; rounding on payslips is to the cent, so 1 rupee is generous
PRECISE_TOL = 0.05   # two amounts rounded separately from the same base differ by a few cents at most

# Digit pairs that OCR commonly confuses in printed numbers.
DIGIT_SWAPS = {"1": "7", "7": "1", "3": "8", "8": "3", "5": "6", "6": "5", "0": "8", "1": "4", "4": "1",
               "5": "9", "9": "5"}


@dataclass
class Rule:
    name: str
    fields: Tuple[str, ...]
    hard: bool
    check: Callable[[Dict[str, float]], bool]
    text: str
    precise: bool = False      # matches to the cent: one repaired precise rule is enough evidence


def _eq(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def epf_bases(v: Dict[str, float]) -> List[float]:
    """Possible EPF-liable earnings. By law COLA (cost-of-living allowance) is EPF-liable, so a
    real Sinhala slip calculates EPF/ETF on basic + COLA (179,045.79 + 12,789.50)."""
    bases = [v[b] for b in ("basic_salary", "gross_income") if b in v]
    if "basic_salary" in v and "cola" in v:
        bases.append(v["basic_salary"] + v["cola"])
    return bases


def _rate_rule(f: str, *rates: float):
    """Several rates are allowed: most employers pay 12% employer EPF, some (e.g. semi-government
    bodies) pay 15%; the payslip in hand decides which one its arithmetic matches."""
    def check(v):
        return any(_eq(v[f], round(base * r, 2)) for base in epf_bases(v) for r in rates)
    return check


RULES: List[Rule] = [
    Rule("net = gross - deductions", ("net_amount", "gross_income", "total_deductions"), True,
         lambda v: _eq(v["net_amount"], v["gross_income"] - v["total_deductions"]),
         "net pay equals gross pay minus total deductions"),
    Rule("gross = basic + allowances", ("gross_income", "basic_salary", "allowances"), False,
         lambda v: _eq(v["gross_income"], v["basic_salary"] + v["allowances"]),
         "gross pay equals basic salary plus allowances"),
    Rule("deductions = EPF + APIT", ("total_deductions", "epf_employee", "apit"), False,
         lambda v: _eq(v["total_deductions"], v["epf_employee"] + v["apit"]),
         "total deductions equal employee EPF plus APIT"),
    Rule("EPF employee = 8%", ("epf_employee",), False, _rate_rule("epf_employee", 0.08),
         "employee EPF is 8% of EPF-liable earnings"),
    Rule("EPF employer = 12% or 15%", ("epf_employer",), False, _rate_rule("epf_employer", 0.12, 0.15),
         "employer EPF is 12% (or 15% at some employers) of EPF-liable earnings"),
    Rule("ETF = 3%", ("etf",), False, _rate_rule("etf", 0.03),
         "ETF is 3% of EPF-liable earnings"),
    # Same base for both, so their RATIO is fixed (3/12 or 3/15) even when basic/COLA were not
    # read: on a real slip this check alone repaired ETF read as 9,755.06 (true 5,755.06).
    Rule("ETF = EPF employer x 3/12 or 3/15", ("etf", "epf_employer"), False,
         lambda v: any(abs(v["etf"] - v["epf_employer"] * r) <= PRECISE_TOL for r in (3 / 12, 3 / 15)),
         "ETF (3%) and employer EPF (12% or 15%) are calculated on the same earnings", precise=True),
]


def _applicable(rule: Rule, v: Dict[str, float]) -> bool:
    if not all(f in v for f in rule.fields):
        return False
    if rule.name.endswith("%"):
        return any(b in v for b in ("basic_salary", "gross_income"))
    return True


@dataclass
class CheckResult:
    rule: str
    hard: bool
    passed: bool
    fields: Tuple[str, ...]
    explanation: str


@dataclass
class FiscalReport:
    values: Dict[str, float]
    checks: List[CheckResult] = field(default_factory=list)
    corrections: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # field -> (old, new)

    def verified(self, f: str) -> bool:
        return any(c.passed and f in c.fields for c in self.checks)

    def contradicted(self, f: str) -> bool:
        return any((not c.passed) and c.hard and f in c.fields for c in self.checks)


def evaluate(values: Dict[str, float]) -> List[CheckResult]:
    out = []
    for r in RULES:
        if _applicable(r, values):
            fs = r.fields + tuple(b for b in ("basic_salary", "gross_income") if r.name.endswith("%") and b in values)
            out.append(CheckResult(r.name, r.hard, bool(r.check(values)), fs, r.text))
    return out


def misreadings(value: float) -> List[Tuple[float, int]]:
    """Plausible true values for an amount OCR read as `value`: one digit removed (OCR added a
    stray character), one digit swapped for a look-alike. Returns (candidate, cost=1)."""
    whole, frac = f"{value:.2f}".split(".")
    out = set()
    for i in range(len(whole)):
        dropped = whole[:i] + whole[i + 1:]
        if dropped and not (len(dropped) > 1 and dropped[0] == "0"):
            out.add(float(f"{dropped}.{frac}"))
        if whole[i] in DIGIT_SWAPS:
            swapped = whole[:i] + DIGIT_SWAPS[whole[i]] + whole[i + 1:]
            if not (len(swapped) > 1 and swapped[0] == "0"):
                out.add(float(f"{swapped}.{frac}"))
    out.discard(value)
    return [(v, 1) for v in sorted(out)]


def _pairs(checks: List[CheckResult], base: List[CheckResult]):
    """(trial check, base check, rule) for rules evaluated in both."""
    by_name = {b.rule: b for b in base}
    rules = {r.name: r for r in RULES}
    return [(c, by_name[c.rule], rules[c.rule]) for c in checks if c.rule in by_name]


def _score(checks: List[CheckResult]) -> Tuple[int, int]:
    return (sum(c.passed for c in checks if c.hard), sum(c.passed for c in checks if not c.hard))


def validate_and_correct(values: Dict[str, float], max_changes: int = 1) -> FiscalReport:
    """Check the arithmetic; if it fails, look for the single most plausible OCR misreading
    that repairs it. A change is accepted only when it
      - repairs at least one HARD rule, or at least two SOFT rules, and
      - is the unique best explanation (no other change scores the same)."""
    base_checks = evaluate(values)
    base_hard, base_soft = _score(base_checks)
    report = FiscalReport(values=dict(values), checks=base_checks)
    if all(c.passed for c in base_checks):
        return report

    best, best_key, tie = None, None, False
    amount_fields = [f for f in values if isinstance(values[f], (int, float))]
    for k in range(1, max_changes + 1):
        for combo in itertools.combinations(amount_fields, k):
            options = [misreadings(values[f]) for f in combo]
            for picks in itertools.product(*options):
                trial = dict(values)
                for f, (v, _) in zip(combo, picks):
                    trial[f] = v
                checks = evaluate(trial)
                hard, soft = _score(checks)
                # A random wrong digit almost never matches a cent-precise rule, so repairing one
                # precise rule is enough; ordinary soft rules (1-rupee tolerance) need two.
                precise_gain = any(c.passed and not b.passed and r.precise
                                   for c, b, r in _pairs(checks, base_checks))
                gain_ok = (hard > base_hard) or (soft - base_soft >= 2) or (precise_gain and soft > base_soft)
                if not gain_ok or hard < base_hard:
                    continue
                key = (hard, soft, -k)
                if best_key is None or key > best_key:
                    best, best_key, tie = (combo, picks, trial, checks), key, False
                elif key == best_key:
                    tie = True
        if best is not None:
            break

    if best is None or tie:
        return report   # nothing convincing: keep OCR values, failed checks will lower confidence
    combo, picks, trial, checks = best
    report.values, report.checks = trial, checks
    for f, (v, _) in zip(combo, picks):
        report.corrections[f] = (values[f], v)
    return report
