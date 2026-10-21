"""Arithmetic checks and self-correction for invoices, receipts and quotations (own method).

Why this module exists:
- A purchase document carries its own arithmetic: quantity x rate = line amount, the lines add up
  to the subtotal, subtotal - discount + taxes = total, and previous balance - paid + total =
  balance due. When OCR misreads one digit, an equation stops holding. So, like the payslip fiscal
  checks, the equations are used as EVIDENCE: a value confirmed by an equation is trusted more, a
  value in a broken equation is flagged, and a misread is repaired only when ONE value, changed by
  ONE digit, makes the equation hold again.
- Layouts differ: some print line amounts before tax, others (GST/VAT-inclusive quotations) after
  tax. Both readings are tried, and the one that holds is reported instead of assumed.
- Nothing here decides tax relief; Components 1 and 3 decide what a purchase means for tax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

CENT = 0.011            # rupees: rounding to cents
ROUNDING = 1.0          # a total printed without cents ('235400' for 235,399.50)
# Rates seen on Sri Lankan / regional documents; used only to say whether an implied rate is plausible.
KNOWN_RATES = {"18%": 0.18, "15%": 0.15, "12%": 0.12, "10%": 0.10, "8%": 0.08, "5%": 0.05,
               "2.5% (SSCL, grossed up 2.5/97.5)": 2.5 / 97.5, "2.5%": 0.025}
RATE_TOL = 0.0015


def close(a: float, b: float) -> bool:
    whole = abs(a - round(a)) < 1e-9 or abs(b - round(b)) < 1e-9
    return abs(a - b) <= (ROUNDING if whole and max(abs(a), abs(b)) >= 100 else CENT)


def digit_close(a: float, b: float) -> bool:
    """True when b differs from a by ONE digit (wrong, extra or missing digit) - the typical OCR
    error - so a repair never invents a value that the page does not almost show."""
    sa, sb = f"{abs(a):.2f}".replace(".", ""), f"{abs(b):.2f}".replace(".", "")
    if len(sa) == len(sb):
        return sum(x != y for x, y in zip(sa, sb)) == 1
    if abs(len(sa) - len(sb)) == 1:
        longer, shorter = (sa, sb) if len(sa) > len(sb) else (sb, sa)
        return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))
    return False


def money(x: float) -> str:
    return f"{x:,.2f}"


@dataclass
class Check:
    rule: str
    hard: bool
    passed: bool
    explanation: str
    fields: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rule": self.rule, "hard": self.hard, "passed": self.passed, "explanation": self.explanation}


@dataclass
class Equation:
    """sum(coef * value[field]) + const == 0"""
    rule: str
    terms: List[Tuple[str, float]]
    const: float
    text: str

    def residual(self, v: Dict[str, float]) -> float:
        return sum(c * v[f] for f, c in self.terms) + self.const

    def holds(self, v: Dict[str, float]) -> bool:
        lhs = sum(c * v[f] for f, c in self.terms if c > 0)
        rhs = -(sum(c * v[f] for f, c in self.terms if c < 0) + self.const)
        return close(lhs, rhs)


@dataclass
class ItemResult:
    passed: Optional[bool]       # None: the row has no quantity/rate to check
    includes_tax: Optional[bool]
    explanation: str


@dataclass
class InvoiceChecks:
    checks: List[Check]
    values: Dict[str, float]                  # after corrections
    corrections: Dict[str, Tuple[float, float]]
    items: List[ItemResult]
    verified: Set[str]
    contradicted: Set[str]


def check_item(qty: Optional[float], rate: Optional[float], amount: Optional[float],
               discount: Optional[float], tax_rate: Optional[float]) -> ItemResult:
    if qty is None or rate is None or amount is None:
        return ItemResult(None, None, "quantity, rate or amount missing; not checkable")
    base = qty * rate
    bases = [base]
    if discount:
        bases += [base - discount, base * (1 - discount / 100)]
    for b in bases:
        if close(b, amount):
            return ItemResult(True, False, f"{qty:g} x {money(rate)} = {money(amount)}")
        if tax_rate and close(b * (1 + tax_rate), amount):
            return ItemResult(True, True, f"{qty:g} x {money(rate)} + {tax_rate:.0%} tax = {money(amount)}")
    return ItemResult(False, None, f"{qty:g} x {money(rate)} = {money(base)}, but the row says {money(amount)}")


def run_checks(values: Dict[str, float], items: Sequence[dict], taxes: Dict[str, float]) -> InvoiceChecks:
    """values: amount fields (subtotal, discount, vat, sscl, total_amount, amount_paid, balance_due,
    previous_balance); items: dicts with quantity, unit_price, amount, discount, tax_rate;
    taxes: printed tax components by name (VAT, SSCL, SGST ...)."""
    v = dict(values)
    for name, amt in taxes.items():
        v[f"tax:{name}"] = amt
    checks: List[Check] = []
    item_results = [check_item(i.get("quantity"), i.get("unit_price"), i.get("amount"),
                               i.get("discount"), i.get("tax_rate")) for i in items]
    for n, r in enumerate(item_results, 1):
        if r.passed is not None:
            checks.append(Check(f"line {n}: quantity x rate = amount", False, r.passed, r.explanation))

    eqs: List[Equation] = []
    amounts = [i["amount"] for i in items if i.get("amount") is not None]
    inclusive = sum(1 for r in item_results if r.includes_tax) > sum(1 for r in item_results if r.includes_tax is False)
    if len(amounts) >= 2 or (len(amounts) == 1 and len(items) == 1):
        target = "total_amount" if inclusive else ("subtotal" if "subtotal" in v else "total_amount")
        if target == "total_amount" and not inclusive and ("vat" in v or taxes or "discount" in v):
            target = None          # lines are before tax but only an after-tax total is printed
        if target and target in v:
            eqs.append(Equation("items_sum", [(target, 1.0)], -sum(amounts),
                                f"sum of {len(amounts)} line amounts = {target.replace('_', ' ')}"))
        if inclusive and "subtotal" in v:
            pre = [i["quantity"] * i["unit_price"] for i in items
                   if i.get("quantity") is not None and i.get("unit_price") is not None]
            if len(pre) == len(amounts):
                eqs.append(Equation("items_sum_before_tax", [("subtotal", 1.0)], -sum(pre),
                                    "sum of quantity x rate (before tax) = subtotal"))

    tax_keys = [f"tax:{n}" for n in taxes] or [f for f in ("vat", "sscl") if f in v]
    if "subtotal" in v and "total_amount" in v:
        terms = [("total_amount", 1.0), ("subtotal", -1.0)] + [(k, -1.0) for k in tax_keys]
        if "discount" in v:
            terms.append(("discount", 1.0))
        eqs.append(Equation("totals_chain", terms, 0.0, "subtotal - discount + taxes = total"))
    if "balance_due" in v and "total_amount" in v and ("amount_paid" in v or "previous_balance" in v):
        terms = [("balance_due", 1.0), ("total_amount", -1.0)]
        if "amount_paid" in v:
            terms.append(("amount_paid", 1.0))
        if "previous_balance" in v:
            terms.append(("previous_balance", -1.0))
        text = ("previous balance - paid + this bill = balance due" if "previous_balance" in v
                else "total - paid = balance due")
        eqs.append(Equation("balance", terms, 0.0, text))

    # Self-correction: ONE value changed by ONE digit must make broken equations hold without breaking
    # one that held. When several values could explain one broken equation, the value that repairs
    # the MOST equations wins (a misread total breaks both 'totals' and 'balance'); a tie is left alone.
    corrections: Dict[str, Tuple[float, float]] = {}
    for _ in range(2):
        failing = [e for e in eqs if not e.holds(v)]
        if not failing:
            break
        options = {}
        for eq in failing:
            for f, c in eq.terms:
                implied = round(v[f] - eq.residual(v) / c, 2)
                if implied >= 0 and digit_close(v[f], implied):
                    options[(f, implied)] = True
        scored = []
        for f, new in options:
            trial = dict(v, **{f: new})
            if any(e.holds(v) and not e.holds(trial) for e in eqs):
                continue
            scored.append((sum(e.holds(trial) for e in eqs), f, new))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            break
        _, f, new = scored[0]
        corrections[f] = (v[f], new)
        v[f] = new

    verified, contradicted = set(), set()
    for eq in eqs:
        ok = eq.holds(v)
        fields = [f for f, _ in eq.terms]
        lhs = " + ".join(f"{'-' if c < 0 else ''}{f.replace('tax:', '')} {money(v[f])}" for f, c in eq.terms)
        fixed = [f for f in fields if f in corrections]
        note = f" (after correcting {', '.join(fixed)})" if fixed else ""
        checks.append(Check(eq.rule, True, ok, f"{eq.text}{note}: " + ("holds" if ok else
                            f"off by {money(abs(eq.residual(v)))}"), fields))
        (verified if ok else contradicted).update(fields)

    taxable = v.get("subtotal", 0.0) - v.get("discount", 0.0)
    tax_total = sum(v[k] for k in tax_keys)
    if taxable > 0 and tax_total > 0:
        implied = tax_total / taxable
        hit = next((name for name, r in KNOWN_RATES.items() if abs(implied - r) <= RATE_TOL), None)
        checks.append(Check("implied tax rate", False, hit is not None,
                            f"taxes {money(tax_total)} on {money(taxable)} = {implied:.2%}"
                            + (f", matches {hit}" if hit else ", not a standard rate")))
    contradicted -= verified
    return InvoiceChecks(checks, v, corrections, item_results, verified, contradicted)
