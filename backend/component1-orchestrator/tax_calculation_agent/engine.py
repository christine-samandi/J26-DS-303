"""Deterministic tax calculation: employment income + APIT paid + RuleSet -> cited CalcSteps.

v1 scope: resident individual, employment income only, one year of assessment.

    S1_RELIEF          personal relief from the rules
    S1_<RULE_KEY>      each other relief the taxpayer claims (e.g. S1_SOLAR_PANEL_RELIEF),
                       applied by its value kind: cap -> min(cap, amount paid),
                       rate -> rate% of the limiting income, amount -> the fixed amount
    S2_TAXABLE_INCOME  max(0, employment income - personal relief - other reliefs)
    S3_BAND_n          tax on the part of taxable income inside band n (only bands reached)
    S4_TOTAL_TAX       sum of the band taxes
    S5_APIT_CREDIT     APIT already deducted by the employer, credited against the tax
    S6_PAYABLE         total tax - APIT, when the tax is at least the APIT paid
    S6_REFUND          APIT - total tax, when more APIT was paid than the tax

Every number is a Decimal. Each band's tax is rounded to the cent (ROUND_HALF_UP) and the
total is the sum of the rounded band taxes, so the steps always add up exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tax_calculation_agent.ruleset import ReliefRule, RuleSet

CENT = Decimal("0.01")
ZERO = Decimal(0)
HUNDRED = Decimal(100)

StepSource = Literal["engine", "ocr", "llm_interpretation"]
BaseSource = Literal["user", "ocr"]


class CalculationInputError(ValueError):
    """An input amount is not a valid LKR money value."""


class CalcStep(BaseModel):
    """One verifiable step. C4 can recompute `result` from `inputs` using `operation`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    description: str
    inputs: dict[str, Decimal | None]
    operation: str
    result: Decimal
    citation_refs: tuple[str, ...]  # rule_ids from the RuleSet
    source: StepSource
    # Arithmetic by the engine is certain; uncertainty in the inputs (OCR, retrieval) is
    # combined separately into C1's overall confidence.
    confidence: Decimal


class ReliefClaim(BaseModel):
    """A relief the taxpayer claims, with the amount it is based on.

    `base` is what the relief is limited by: the amount paid for a cap (e.g. the solar
    invoice total), the limiting income for a rate, None for a fixed amount. Where `base`
    came from is recorded (`base_source`, `document_ids`) so the explanation and C4 can show
    the evidence; the engine itself only does the arithmetic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: ReliefRule
    base: Decimal | None = None
    base_source: BaseSource | None = None
    document_ids: tuple[str, ...] = ()
    document_type: str | None = None  # e.g. "invoice", named in the step description


class CalcResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    year_of_assessment: str
    employment_income: Decimal
    apit_paid: Decimal
    taxable_income: Decimal
    total_tax: Decimal
    tax_payable: Decimal  # 0 when there is a refund
    refund: Decimal  # 0 when tax is payable
    steps: tuple[CalcStep, ...]


def calculate(
    employment_income: Decimal,
    apit_paid: Decimal,
    rules: RuleSet,
    reliefs: Sequence[ReliefClaim] = (),
) -> CalcResult:
    """Compute tax for one YA. Same inputs always give an identical result.

    `reliefs` are the other reliefs claimed, one per rule; without them the steps are
    exactly those of personal relief alone.
    """
    income = _money("employment_income", employment_income)
    apit = _money("apit_paid", apit_paid)
    relief_ref = rules.personal_relief.ref.rule_id
    bands_ref = rules.tax_bands.ref.rule_id
    relief = rules.personal_relief.amount
    keys = [c.rule.rule_key for c in reliefs]
    if len(keys) != len(set(keys)):
        raise CalculationInputError(f"each relief can be claimed once, got {keys}")

    steps: list[CalcStep] = [
        _step(
            "S1_RELIEF",
            "Personal relief for a resident individual",
            {"personal_relief": relief},
            "personal_relief",
            relief,
            (relief_ref,),
        )
    ]
    others = [_relief_step(claim) for claim in reliefs]
    steps += others

    if not others:
        taxable = max(ZERO, income - relief)
        steps.append(
            _step(
                "S2_TAXABLE_INCOME",
                "Taxable income: employment income less personal relief (not below zero)",
                {"employment_income": income, "personal_relief": relief},
                "max(0, employment_income - personal_relief)",
                taxable,
                (relief_ref,),
            )
        )
    else:
        other_total = sum((s.result for s in others), ZERO)
        taxable = max(ZERO, income - relief - other_total)
        names = [c.rule.rule_key for c in reliefs]
        steps.append(
            _step(
                "S2_TAXABLE_INCOME",
                "Taxable income: employment income less personal relief and the other "
                "reliefs claimed (not below zero)",
                {
                    "employment_income": income,
                    "personal_relief": relief,
                    **{name: s.result for name, s in zip(names, others, strict=True)},
                },
                "max(0, employment_income - personal_relief - " + " - ".join(names) + ")",
                taxable,
                (relief_ref, *(c.rule.ref.rule_id for c in reliefs)),
            )
        )

    band_taxes: list[Decimal] = []
    for band in rules.tax_bands.bands:
        if taxable <= band.lower:
            break
        top = taxable if band.upper is None else min(taxable, band.upper)
        in_band = top - band.lower
        tax = (in_band * band.rate_percent / HUNDRED).quantize(CENT, rounding=ROUND_HALF_UP)
        band_taxes.append(tax)
        upper_text = "and above" if band.upper is None else f"to {band.upper}"
        steps.append(
            _step(
                f"S3_BAND_{band.order}",
                f"Tax at {band.rate_percent}% on taxable income from {band.lower} {upper_text}",
                {
                    "taxable_income": taxable,
                    "band_lower": band.lower,
                    "band_upper": band.upper,
                    "amount_in_band": in_band,
                    "rate_percent": band.rate_percent,
                },
                "round_half_up(amount_in_band * rate_percent / 100, 0.01)",
                tax,
                (bands_ref,),
            )
        )

    total_tax = sum(band_taxes, ZERO)
    steps.append(
        _step(
            "S4_TOTAL_TAX",
            "Total income tax: sum of the tax in each band",
            {f"band_{i}_tax": t for i, t in enumerate(band_taxes, start=1)},
            "sum(band taxes)",
            total_tax,
            (bands_ref,),
        )
    )

    # TODO(citation): no APIT-credit rule in the RuleSet yet, so S5/S6 carry no citation.
    steps.append(
        _step(
            "S5_APIT_CREDIT",
            "APIT already deducted by the employer, credited against the tax",
            {"apit_paid": apit},
            "apit_paid",
            apit,
            (),
        )
    )

    if total_tax >= apit:
        payable, refund = total_tax - apit, ZERO
        steps.append(
            _step(
                "S6_PAYABLE",
                "Tax still payable: total tax less APIT credit",
                {"total_tax": total_tax, "apit_paid": apit},
                "total_tax - apit_paid",
                payable,
                (),
            )
        )
    else:
        payable, refund = ZERO, apit - total_tax
        steps.append(
            _step(
                "S6_REFUND",
                "Refund due: APIT credit exceeds total tax",
                {"total_tax": total_tax, "apit_paid": apit},
                "apit_paid - total_tax",
                refund,
                (),
            )
        )

    return CalcResult(
        year_of_assessment=rules.year_of_assessment,
        employment_income=income,
        apit_paid=apit,
        taxable_income=taxable,
        total_tax=total_tax,
        tax_payable=payable,
        refund=refund,
        steps=tuple(steps),
    )


def remaining_claimable(claim: ReliefClaim) -> Decimal:
    """For a cap relief: the part of the amount paid above this YA's cap (display only).

    Shown as "remaining claimable in later years"; the engine does not apply it to any
    other year. 0 for rate and fixed-amount reliefs.
    """
    if claim.rule.value_kind != "cap" or claim.base is None:
        return ZERO
    return max(ZERO, claim.base - claim.rule.value)


def _relief_step(claim: ReliefClaim) -> CalcStep:
    rule = claim.rule
    step_id = f"S1_{rule.rule_key.upper()}"
    evidence = _evidence_phrase(claim)
    if rule.value_kind == "amount":
        if claim.base is not None:
            raise CalculationInputError(f"{rule.rule_key} is a fixed amount: give no base")
        return _step(
            step_id,
            f"{rule.title}: a fixed amount",
            {rule.rule_key: rule.value},
            rule.rule_key,
            rule.value,
            (rule.ref.rule_id,),
        )
    if claim.base is None:
        raise CalculationInputError(f"{rule.rule_key} needs the amount it is based on")
    base = _money(f"{rule.rule_key} base", claim.base)
    if rule.value_kind == "cap":
        return _step(
            step_id,
            f"{rule.title}: the lower of the yearly limit and the amount paid{evidence}",
            {"cap": rule.value, "amount_paid": base},
            "min(cap, amount_paid)",
            min(rule.value, base),
            (rule.ref.rule_id,),
        )
    return _step(
        step_id,
        f"{rule.title}: {rule.value}% of the income it is limited to{evidence}",
        {"limiting_income": base, "rate_percent": rule.value},
        "round_half_up(limiting_income * rate_percent / 100, 0.01)",
        (base * rule.value / HUNDRED).quantize(CENT, rounding=ROUND_HALF_UP),
        (rule.ref.rule_id,),
    )


def _evidence_phrase(claim: ReliefClaim) -> str:
    """' (from the uploaded invoice, amount confirmed by the user)': no figures, no names."""
    parts = []
    if claim.document_ids:
        kind = claim.document_type or "document"
        many = len(claim.document_ids) > 1
        parts.append(f"from the uploaded {kind}{'s' if many else ''}")
    if claim.base_source == "user":
        parts.append("amount confirmed by the user")
    elif claim.base_source == "ocr":
        parts.append("amount read by OCR")
    return f" ({', '.join(parts)})" if parts else ""


def _step(
    step_id: str,
    description: str,
    inputs: dict[str, Decimal | None],
    operation: str,
    result: Decimal,
    citation_refs: tuple[str, ...],
) -> CalcStep:
    return CalcStep(
        step_id=step_id,
        description=description,
        inputs=inputs,
        operation=operation,
        result=result,
        citation_refs=citation_refs,
        source="engine",
        confidence=Decimal(1),
    )


def _money(name: str, value: object) -> Decimal:
    """Accept Decimal or int LKR amounts, >= 0, at most 2 decimal places. Floats are refused."""
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        raise CalculationInputError(f"{name} must be a Decimal or int, got {type(value).__name__}")
    amount = Decimal(value)
    if not amount.is_finite():
        raise CalculationInputError(f"{name} must be a finite amount, got {value}")
    if amount < 0:
        raise CalculationInputError(f"{name} cannot be negative, got {value}")
    if amount != amount.quantize(CENT):
        raise CalculationInputError(f"{name} can have at most 2 decimal places, got {value}")
    return amount
