"""Worked examples for the YA 2025/26 engine, checked by hand.

`calculate()` takes GROSS employment income. Taxable income = gross - 1,800,000 relief,
not below zero. Test names say which one a figure is:
    gross_*    the employment income passed to calculate()
    taxable_*  the income after personal relief (gross = taxable + 1,800,000)
"""

from decimal import Decimal as D

import pytest

from tax_calculation_agent.engine import CalcResult, CalculationInputError, calculate
from tax_calculation_agent.ruleset import load_seed_rules

RULES = load_seed_rules("2025/26")
RELIEF = D(1800000)
RELIEF_RULE = RULES.personal_relief.ref.rule_id
BANDS_RULE = RULES.tax_bands.ref.rule_id


def from_taxable(taxable, apit=D(0)) -> CalcResult:
    return calculate(D(taxable) + RELIEF, apit, RULES)


def step(result: CalcResult, step_id: str):
    (match,) = [s for s in result.steps if s.step_id == step_id]
    return match


def step_ids(result: CalcResult) -> list[str]:
    return [s.step_id for s in result.steps]


# --- the two figures from the brief ------------------------------------------


def test_taxable_2500000_gives_tax_420000():
    r = from_taxable(2500000)
    assert r.employment_income == D(4300000)  # gross
    assert r.taxable_income == D(2500000)
    assert [step(r, f"S3_BAND_{i}").result for i in range(1, 5)] == [
        D(60000),
        D(90000),
        D(120000),
        D(150000),
    ]
    assert "S3_BAND_5" not in step_ids(r)
    assert r.total_tax == D(420000)


def test_gross_2500000_gives_taxable_700000_and_tax_42000():
    r = calculate(D(2500000), D(0), RULES)
    assert r.taxable_income == D(700000)
    assert step(r, "S3_BAND_1").result == D(42000)
    assert r.total_tax == D(42000)


# --- every band edge, by taxable income ---------------------------------------
# Tax at each edge: 1.0M -> 60,000; +0.5M @18% -> 150,000; +0.5M @24% -> 270,000;
# +0.5M @30% -> 420,000; then 36% on the balance.

BAND_EDGES_TAXABLE = [
    (0, 0),
    (1, D("0.06")),
    (999999, D("59999.94")),
    (1000000, 60000),
    (1000001, D("60000.18")),
    (1499999, D("149999.82")),
    (1500000, 150000),
    (1500001, D("150000.24")),
    (1999999, D("269999.76")),
    (2000000, 270000),
    (2000001, D("270000.30")),
    (2499999, D("419999.70")),
    (2500000, 420000),
    (2500001, D("420000.36")),
    (3000000, 600000),
    (10000000, 3120000),
]


@pytest.mark.parametrize(
    ("taxable", "tax"),
    BAND_EDGES_TAXABLE,
    ids=[f"taxable_{t}->tax_{x}" for t, x in BAND_EDGES_TAXABLE],
)
def test_band_edges_by_taxable_income(taxable, tax):
    r = from_taxable(taxable)
    assert r.taxable_income == D(taxable)
    assert r.total_tax == D(tax)


@pytest.mark.parametrize(
    ("taxable", "bands_reached"),
    [(0, 0), (1000000, 1), (1000001, 2), (2500000, 4), (2500001, 5)],
    ids=lambda v: str(v),
)
def test_only_bands_reached_get_a_step(taxable, bands_reached):
    r = from_taxable(taxable)
    assert [s for s in step_ids(r) if s.startswith("S3_BAND_")] == [
        f"S3_BAND_{i}" for i in range(1, bands_reached + 1)
    ]


def test_taxable_3000000_top_band_is_open_ended():
    band5 = step(from_taxable(3000000), "S3_BAND_5")
    assert band5.inputs["band_upper"] is None
    assert band5.inputs["amount_in_band"] == D(500000)
    assert band5.result == D(180000)


# --- zero and below relief ----------------------------------------------------


def test_gross_0_gives_no_tax():
    r = calculate(D(0), D(0), RULES)
    assert (r.taxable_income, r.total_tax, r.tax_payable, r.refund) == (0, 0, 0, 0)
    assert step_ids(r) == [
        "S1_RELIEF",
        "S2_TAXABLE_INCOME",
        "S4_TOTAL_TAX",
        "S5_APIT_CREDIT",
        "S6_PAYABLE",
    ]


@pytest.mark.parametrize("gross", [1, 1000000, 1799999, 1800000], ids=lambda g: f"gross_{g}")
def test_gross_at_or_below_relief_gives_taxable_0_and_tax_0(gross):
    r = calculate(D(gross), D(0), RULES)
    assert r.taxable_income == 0
    assert r.total_tax == 0
    assert step(r, "S2_TAXABLE_INCOME").result == 0  # never negative


def test_gross_1800001_gives_taxable_1_and_tax_0_06():
    r = calculate(D(1800001), D(0), RULES)
    assert r.taxable_income == D(1)
    assert r.total_tax == D("0.06")


# --- APIT credit: payable or refund -------------------------------------------


def test_gross_2500000_apit_underpaid_gives_payable_12000():
    r = calculate(D(2500000), D(30000), RULES)
    assert (r.total_tax, r.tax_payable, r.refund) == (42000, 12000, 0)
    assert step(r, "S6_PAYABLE").result == D(12000)
    assert "S6_REFUND" not in step_ids(r)


def test_gross_2500000_apit_overpaid_gives_refund_8000():
    r = calculate(D(2500000), D(50000), RULES)
    assert (r.total_tax, r.tax_payable, r.refund) == (42000, 0, 8000)
    assert step(r, "S5_APIT_CREDIT").result == D(50000)
    assert step(r, "S6_REFUND").result == D(8000)
    assert "S6_PAYABLE" not in step_ids(r)


def test_gross_2500000_apit_exact_gives_payable_0():
    r = calculate(D(2500000), D(42000), RULES)
    assert (r.tax_payable, r.refund) == (0, 0)
    assert step(r, "S6_PAYABLE").result == 0


def test_gross_below_relief_with_apit_gives_full_refund():
    r = calculate(D(1500000), D(5000), RULES)
    assert (r.total_tax, r.tax_payable, r.refund) == (0, 0, 5000)


# --- cents and rounding -------------------------------------------------------


def test_taxable_0_50_gives_tax_0_03():
    assert from_taxable(D("0.50")).total_tax == D("0.03")


def test_taxable_0_25_rounds_half_up_to_0_02():
    # 0.25 * 6% = 0.015 -> 0.02
    assert from_taxable(D("0.25")).total_tax == D("0.02")


def test_total_is_sum_of_rounded_band_taxes():
    r = from_taxable(D("1000000.25"))  # band 2: 0.25 * 18% = 0.045 -> 0.05
    assert step(r, "S3_BAND_2").result == D("0.05")
    assert r.total_tax == D("60000.05")


# --- steps are cited, chained and recomputable --------------------------------


def test_steps_cite_the_rule_they_use():
    r = from_taxable(3000000, apit=D(1))
    assert step(r, "S1_RELIEF").citation_refs == (RELIEF_RULE,)
    assert step(r, "S2_TAXABLE_INCOME").citation_refs == (RELIEF_RULE,)
    for s in r.steps:
        if s.step_id.startswith("S3_BAND_"):
            assert s.citation_refs == (BANDS_RULE,)
    assert step(r, "S4_TOTAL_TAX").citation_refs == (BANDS_RULE,)


def test_apit_steps_have_no_citation_yet():
    # Known gap: no APIT-credit rule in the RuleSet. Fails once one is added, as a reminder.
    r = calculate(D(2500000), D(50000), RULES)
    assert step(r, "S5_APIT_CREDIT").citation_refs == ()
    assert step(r, "S6_REFUND").citation_refs == ()


def test_steps_chain_their_inputs():
    r = calculate(D(4300000), D(400000), RULES)
    relief = step(r, "S1_RELIEF").result
    taxable = step(r, "S2_TAXABLE_INCOME")
    assert taxable.inputs == {"employment_income": D(4300000), "personal_relief": relief}
    bands = [s for s in r.steps if s.step_id.startswith("S3_BAND_")]
    for b in bands:
        assert b.inputs["taxable_income"] == taxable.result
    total = step(r, "S4_TOTAL_TAX")
    assert list(total.inputs.values()) == [b.result for b in bands]
    assert total.result == sum(b.result for b in bands)
    assert step(r, "S6_PAYABLE").inputs == {"total_tax": total.result, "apit_paid": D(400000)}


def test_every_step_from_engine_with_full_confidence():
    for s in from_taxable(3000000).steps:
        assert s.source == "engine"
        assert s.confidence == 1


def test_all_figures_are_decimal():
    r = from_taxable(D("2500000.75"), apit=D("100.10"))
    for s in r.steps:
        assert type(s.result) is D
        assert all(v is None or type(v) is D for v in s.inputs.values())


# --- bad inputs ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("income", "apit", "message"),
    [
        (2500000.0, D(0), "Decimal or int"),
        (D(2500000), 0.0, "Decimal or int"),
        ("2500000", D(0), "Decimal or int"),
        (True, D(0), "Decimal or int"),
        (D(-1), D(0), "negative"),
        (D(0), D("-0.01"), "negative"),
        (D("1.005"), D(0), "2 decimal places"),
        (D("NaN"), D(0), "finite"),
        (D("Infinity"), D(0), "finite"),
    ],
)
def test_bad_money_rejected(income, apit, message):
    with pytest.raises(CalculationInputError, match=message):
        calculate(income, apit, RULES)


def test_int_amounts_accepted():
    assert calculate(2500000, 0, RULES).total_tax == D(42000)
