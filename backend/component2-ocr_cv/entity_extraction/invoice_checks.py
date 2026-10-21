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


def lost_decimal(a: float, b: float) -> bool:
    """True when a is b with its decimal point dropped: OCR read '300,000.00' as '30000000'.
    Sri Lankan money is printed with two decimals, so a = b x 100 exactly."""
    return b > 0 and abs(a - round(b * 100)) < 0.5 and abs(a - round(a)) < 1e-9


def digit_close(a: float, b: float) -> bool:
    """True when b differs from a by ONE digit (wrong, extra or missing digit) or only by a lost
    decimal point - the typical OCR errors - so a repair never invents a value that the page does
    not almost show."""
    if lost_decimal(a, b):
        return True
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
    unit_price: Optional[float] = None    # values after a lost-decimal repair (else as read)
    amount: Optional[float] = None
    repaired: bool = False


@dataclass
class InvoiceChecks:
    checks: List[Check]
    values: Dict[str, float]                  # after corrections
    corrections: Dict[str, Tuple[float, float]]
    items: List[ItemResult]
    verified: Set[str]
    contradicted: Set[str]


def _check_once(qty, rate, amount, discount, tax_rate) -> Optional[Tuple[bool, str]]:
    base = qty * rate
    bases = [(base, "")]
    if discount:
        bases += [(base - discount, f" - {money(discount)}"), (base * (1 - discount / 100), f" - {discount:g}%")]
    for b, d in bases:
        if close(b, amount):
            return False, f"{qty:g} x {money(rate)}{d} = {money(amount)}"
        if tax_rate and close(b * (1 + tax_rate), amount):
            return True, f"{qty:g} x {money(rate)}{d} + {tax_rate:.0%} tax = {money(amount)}"
    return None


def check_item(qty: Optional[float], rate: Optional[float], amount: Optional[float],
               discount: Optional[float], tax_rate: Optional[float]) -> ItemResult:
    """qty x rate (- discount) (+ tax) = amount. If it fails, the row is tried again with the
    decimal point restored in the rate and/or amount ('4150000' -> 41,500.00): OCR often drops the
    point on photos, and two printed numbers agreeing after the same repair is strong evidence."""
    if qty is None or rate is None or amount is None:
        return ItemResult(None, None, "quantity, rate or amount missing; not checkable", rate, amount)
    hit = _check_once(qty, rate, amount, discount, tax_rate)
    if hit:
        return ItemResult(True, hit[0], hit[1], rate, amount)
    for r, a in ((rate / 100, amount / 100), (rate, amount / 100), (rate / 100, amount)):
        if (r != rate and abs(rate - round(rate)) > 1e-9) or (a != amount and abs(amount - round(amount)) > 1e-9):
            continue                      # only a number printed WITHOUT decimals can have lost its point
        hit = _check_once(qty, r, a, discount, tax_rate)
        if hit:
            return ItemResult(True, hit[0], hit[1] + " (decimal point restored)", round(r, 2), round(a, 2), True)
    return ItemResult(False, None, f"{qty:g} x {money(rate)} = {money(qty * rate)}, but the row says {money(amount)}",
                      rate, amount)


def _rate_ok(v: Dict[str, float], tax_keys: Sequence[str]) -> bool:
    taxable = v.get("subtotal", 0.0) - v.get("discount", 0.0)
    tax_total = sum(v.get(k, 0.0) for k in tax_keys)
    return taxable > 0 and tax_total > 0 and \
        any(abs(tax_total / taxable - r) <= RATE_TOL for r in KNOWN_RATES.values())


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

    # Lost decimal points across the table: q x r = a also holds with r/100 and a/100, so a row
    # alone cannot tell '4150000' from 41,500.00. The lines must add up to the printed subtotal (or
    # total): if they only do after restoring the point in rows printed WITHOUT separators, restore it.
    target_value = v.get("subtotal", v.get("total_amount"))
    suspects = [k for k, (i, r) in enumerate(zip(items, item_results))
                if i.get("no_separator") and r.amount is not None and r.amount >= 100 and not r.repaired]
    if suspects and target_value is not None:
        total = sum(r.amount for r in item_results if r.amount is not None)
        fixed = total - sum(item_results[k].amount for k in suspects) * 0.99
        if not close(total, target_value) and close(fixed, target_value):
            for k in suspects:
                r = item_results[k]
                r.amount = round(r.amount / 100, 2)
                r.unit_price = round(r.unit_price / 100, 2) if r.unit_price is not None and \
                    abs(r.unit_price - round(r.unit_price)) < 1e-9 and r.unit_price >= 100 else r.unit_price
                r.repaired = True
                q = items[k].get("quantity")
                r.explanation = ((f"{q:g} x {money(r.unit_price)} = " if q is not None and r.unit_price is not None else "")
                                 + f"{money(r.amount)} (decimal point restored: the lines then add up to the subtotal)")

    # line checks are reported AFTER the table-level repair, so they show the repaired values
    for n, r in enumerate(item_results, 1):
        if r.passed is not None or r.repaired:
            checks.append(Check(f"line {n}: quantity x rate = amount", False, r.passed is not False, r.explanation))

    eqs: List[Equation] = []
    amounts = [r.amount for r in item_results if r.amount is not None]
    inclusive = sum(1 for r in item_results if r.includes_tax) > sum(1 for r in item_results if r.includes_tax is False)
    if len(amounts) >= 2 or (len(amounts) == 1 and len(items) == 1):
        target = "total_amount" if inclusive else ("subtotal" if "subtotal" in v else "total_amount")
        if target == "total_amount" and not inclusive and ("vat" in v or taxes or "discount" in v):
            target = None          # lines are before tax but only an after-tax total is printed
        if target and target in v:
            eqs.append(Equation("items_sum", [(target, 1.0)], -sum(amounts),
                                f"sum of {len(amounts)} line amounts = {target.replace('_', ' ')}"))
        if inclusive and "subtotal" in v:
            pre = [i["quantity"] * r.unit_price for i, r in zip(items, item_results)
                   if i.get("quantity") is not None and r.unit_price is not None]
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
            if _rate_ok(v, tax_keys) and not _rate_ok(trial, tax_keys):
                continue          # a tax that is exactly 18% of the subtotal is not the misread value
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
