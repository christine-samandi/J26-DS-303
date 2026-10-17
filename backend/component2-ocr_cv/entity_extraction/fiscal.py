"""Fiscal validation and self-correction of payslip amounts (Gap 7; own algorithm).

Why this module exists:
OCR confidence says how sure the engine is about the SHAPES it read, not whether the number
makes sense. A payslip, however, must obey arithmetic:
    net pay = gross pay - total deductions            (always true - a HARD rule)
    gross   = basic + allowances                      (true unless other earnings exist - SOFT)
    total deductions = EPF(employee) + APIT           (true unless other deductions exist - SOFT)
    EPF employee = 8%, EPF employer = 12%, ETF = 3% of EPF-liable earnings (basic or gross - SOFT)
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


def _eq(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def _rate_rule(f: str, rate: float):
    def check(v):
        bases = [v[b] for b in ("basic_salary", "gross_income") if b in v]
        return any(_eq(v[f], round(base * rate, 2)) for base in bases)
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
         "employee EPF is 8% of basic or gross"),
    Rule("EPF employer = 12%", ("epf_employer",), False, _rate_rule("epf_employer", 0.12),
         "employer EPF is 12% of basic or gross"),
    Rule("ETF = 3%", ("etf",), False, _rate_rule("etf", 0.03),
         "ETF is 3% of basic or gross"),
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
                gain_ok = (hard > base_hard) or (soft - base_soft >= 2)
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
