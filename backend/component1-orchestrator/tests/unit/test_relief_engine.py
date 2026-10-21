"""Other reliefs (beyond personal relief): building them from C3 / the seed, and the engine
applying them generically by value kind (cap, rate, amount). Figures checked by hand."""

import copy
import csv
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from j26_contracts.retrieval import RetrievedRule

from tax_calculation_agent.engine import (
    CalcResult,
    CalculationInputError,
    ReliefClaim,
    calculate,
    remaining_claimable,
)
from tax_calculation_agent.ruleset import (
    ReliefRule,
    RuleRef,
    RuleSetError,
    load_seed_reliefs,
    load_seed_rules,
    relief_conflicts,
    relief_from_retrieval,
)

RULES = load_seed_rules("2025/26")
(SOLAR,) = load_seed_reliefs("2025/26")
GROUND_TRUTH = Path(__file__).resolve().parents[3] / "component3-graphrag/ground_truth"

# C3's answer for "solar panel relief" as of 2025-06-01 (graph_backend over the real corpus)
C3_SOLAR = {
    "rank": 1,
    "rule_id": "RULE-SOLAR_RELIEF-V1",
    "rule_key": "solar_panel_relief",
    "rule_type": "relief",
    "title": "Solar panel relief",
    "text": "Solar panel relief: up to Rs. 600,000 for each year of assessment, and not more "
    "than the total cost of the solar panels, or the amounts paid to a bank on a loan taken "
    "to buy them. Who can claim: a resident individual who has fixed solar panels on their "
    "premises, connected to the national grid.",
    "values": {"cap": {"amount": 600000, "currency": "LKR", "period": "year_of_assessment"}},
    "applies_to": {
        "income_types": ["employment", "business", "investment", "other"],
        "taxpayer_categories": ["resident_individual"],
    },
    "source": {
        "document_id": "DOC-IRA-AMEND-10-2021",
        "document_title": "Inland Revenue (Amendment) Act, No. 10 of 2021",
        "document_type": "amendment_act",
        "citation": "Inland Revenue (Amendment) Act, No. 10 of 2021, s.55(2) (amending Fifth "
        "Schedule, paragraph 2(g) of Act No. 24 of 2017), pp.55-59",
    },
    "validity": {
        "valid_from": "2021-04-01",
        "valid_to": None,
        "status": "active",
        "in_force_on_as_of_date": True,
    },
    "confidence": {
        "score": 0.897,
        "level": "high",
        "components": {
            "semantic_similarity": 0.726,
            "graph_match": 1.0,
            "extraction_confidence": 0.95,
            "temporal_match": 1.0,
        },
    },
    "evidence": {
        "chunk_ids": ["CHUNK-IRA-AMEND-10-2021-S055-2-P2"],
        "matched_text": "(g) in the case of a resident individual who has acquired solar panels "
        "to fix on his premises and connected to the national grid, Rs. 600,000 for each year "
        "of assessment, upto the total expenditure on such solar panels or upto the amounts "
        "paid to a bank in respect of any loan obtained to acquire such solar panels.",
    },
}


def c3_rule(**changes) -> RetrievedRule:
    data = copy.deepcopy(C3_SOLAR)
    data.update(changes)
    return RetrievedRule.model_validate(data)


def relief(kind: str, value, key: str = "test_relief", income=("employment",)) -> ReliefRule:
    return ReliefRule(
        rule_key=key,
        title="Test relief",
        value_kind=kind,
        value=D(value),
        income_types=income,
        ref=RuleRef(
            rule_id=f"RULE-{key.upper()}-V1",
            citation="Test Act, s.1",
            origin="c3",
            valid_from=date(2021, 4, 1),
            valid_to=None,
        ),
    )


def solar_claim(paid="800000.00", source="user") -> ReliefClaim:
    return ReliefClaim(
        rule=SOLAR,
        base=D(paid),
        base_source=source,
        document_ids=("solar_invoice_2025",),
        document_type="invoice",
    )


def step(result: CalcResult, step_id: str):
    (match,) = [s for s in result.steps if s.step_id == step_id]
    return match


# --- the solar scenario from the brief ------------------------------------------------------


def test_solar_invoice_800000_gives_relief_600000_and_refund_14000():
    claim = solar_claim()
    r = calculate(D(3000000), D(50000), RULES, [claim])
    s = step(r, "S1_SOLAR_PANEL_RELIEF")
    assert (s.result, s.inputs) == (D(600000), {"cap": D(600000), "amount_paid": D(800000)})
    assert s.operation == "min(cap, amount_paid)"
    assert s.citation_refs == ("SEED-SOLAR_RELIEF-YA2025_26",)
    assert r.taxable_income == D(600000)  # 3,000,000 - 1,800,000 - 600,000
    assert r.total_tax == D(36000)  # 600,000 @ 6%
    assert (r.tax_payable, r.refund) == (D(0), D(14000))  # APIT 50,000 - 36,000
    assert remaining_claimable(claim) == D(200000)


def test_solar_steps_in_order_and_taxable_step_cites_both_reliefs():
    r = calculate(D(3000000), D(50000), RULES, [solar_claim()])
    assert [s.step_id for s in r.steps] == [
        "S1_RELIEF",
        "S1_SOLAR_PANEL_RELIEF",
        "S2_TAXABLE_INCOME",
        "S3_BAND_1",
        "S4_TOTAL_TAX",
        "S5_APIT_CREDIT",
        "S6_REFUND",
    ]
    taxable = step(r, "S2_TAXABLE_INCOME")
    assert taxable.inputs == {
        "employment_income": D(3000000),
        "personal_relief": D(1800000),
        "solar_panel_relief": D(600000),
    }
    assert taxable.operation == ("max(0, employment_income - personal_relief - solar_panel_relief)")
    assert taxable.citation_refs == (
        "SEED-PERSONAL_RELIEF-YA2025_26",
        "SEED-SOLAR_RELIEF-YA2025_26",
    )


def test_relief_step_says_where_the_amount_came_from_without_figures():
    user = step(calculate(D(3000000), D(0), RULES, [solar_claim()]), "S1_SOLAR_PANEL_RELIEF")
    assert user.description == (
        "Solar panel relief: the lower of the yearly limit and the amount paid "
        "(from the uploaded invoice, amount confirmed by the user)"
    )
    ocr = step(
        calculate(D(3000000), D(0), RULES, [solar_claim(source="ocr")]), "S1_SOLAR_PANEL_RELIEF"
    )
    assert ocr.description.endswith("(from the uploaded invoice, amount read by OCR)")


def test_no_relief_claimed_is_unchanged():
    plain = calculate(D(3000000), D(50000), RULES)
    assert calculate(D(3000000), D(50000), RULES, ()).model_dump_json() == plain.model_dump_json()
    assert plain.taxable_income == D(1200000)
    assert plain.total_tax == D(96000)  # 60,000 + 200,000 @ 18%
    s2 = step(plain, "S2_TAXABLE_INCOME")
    assert s2.description == (
        "Taxable income: employment income less personal relief (not below zero)"
    )
    assert s2.operation == "max(0, employment_income - personal_relief)"
    assert [s.step_id for s in plain.steps][:2] == ["S1_RELIEF", "S2_TAXABLE_INCOME"]


# --- cap ------------------------------------------------------------------------------------


def test_cap_paid_below_the_cap_gives_the_amount_paid():
    claim = solar_claim("450000.50")
    r = calculate(D(3000000), D(0), RULES, [claim])
    assert step(r, "S1_SOLAR_PANEL_RELIEF").result == D("450000.50")
    assert r.taxable_income == D("749999.50")
    assert remaining_claimable(claim) == D(0)


def test_taxable_income_never_below_zero():
    r = calculate(D(2000000), D(10000), RULES, [solar_claim()])
    assert r.taxable_income == D(0)  # 2,000,000 - 1,800,000 - 600,000 < 0
    assert r.total_tax == D(0)
    assert r.refund == D(10000)


# --- rate and amount: the same engine, no relief-specific code -----------------------------


def test_rate_relief_is_rate_percent_of_the_limiting_income():
    claim = ReliefClaim(rule=relief("rate", 25), base=D("400000.10"), base_source="user")
    r = calculate(D(3000000), D(0), RULES, [claim])
    s = step(r, "S1_TEST_RELIEF")
    assert s.inputs == {"limiting_income": D("400000.10"), "rate_percent": D(25)}
    assert s.operation == "round_half_up(limiting_income * rate_percent / 100, 0.01)"
    assert s.result == D("100000.03")  # 100,000.025 rounded half up
    assert r.taxable_income == D("1099999.97")
    assert remaining_claimable(claim) == D(0)


def test_amount_relief_is_the_fixed_amount():
    r = calculate(D(3000000), D(0), RULES, [ReliefClaim(rule=relief("amount", 100000))])
    s = step(r, "S1_TEST_RELIEF")
    assert (s.result, s.inputs) == (D(100000), {"test_relief": D(100000)})
    assert r.taxable_income == D(1100000)


def test_several_reliefs_are_all_subtracted():
    claims = [solar_claim(), ReliefClaim(rule=relief("amount", 100000))]
    r = calculate(D(3000000), D(0), RULES, claims)
    assert r.taxable_income == D(500000)  # 3,000,000 - 1,800,000 - 600,000 - 100,000
    assert step(r, "S2_TAXABLE_INCOME").operation == (
        "max(0, employment_income - personal_relief - solar_panel_relief - test_relief)"
    )


@pytest.mark.parametrize(
    "claims, message",
    [
        ([solar_claim(), solar_claim()], "claimed once"),
        ([ReliefClaim(rule=SOLAR)], "needs the amount"),
        ([ReliefClaim(rule=relief("amount", 1), base=D(5))], "give no base"),
        ([ReliefClaim(rule=SOLAR, base=D(-1))], "cannot be negative"),
        ([ReliefClaim(rule=SOLAR, base=D("1.001"))], "2 decimal places"),
    ],
)
def test_bad_claims_are_refused(claims, message):
    with pytest.raises(CalculationInputError, match=message):
        calculate(D(3000000), D(0), RULES, claims)


def test_float_base_is_refused():
    with pytest.raises(CalculationInputError, match="Decimal or int"):
        calculate(D(3000000), D(0), RULES, [ReliefClaim.model_construct(rule=SOLAR, base=1.5)])


@pytest.mark.parametrize(
    "kind, value, key, message",
    [
        ("rate", 101, "x", "within 0..100"),
        ("cap", -1, "x", "cannot be negative"),
        ("amount", 1, "personal_relief", "personal relief"),
    ],
)
def test_invalid_relief_rules_are_refused(kind, value, key, message):
    with pytest.raises(ValueError, match=message):
        relief(kind, value, key)


@given(
    paid_a=st.integers(min_value=0, max_value=2_000_000),
    paid_b=st.integers(min_value=0, max_value=2_000_000),
    income=st.integers(min_value=0, max_value=10_000_000),
)
def test_paying_more_never_raises_the_tax(paid_a, paid_b, income):
    low, high = sorted((paid_a, paid_b))
    tax_low = calculate(D(income), D(0), RULES, [solar_claim(low)]).total_tax
    tax_high = calculate(D(income), D(0), RULES, [solar_claim(high)]).total_tax
    assert tax_high <= tax_low


def test_reproducible():
    first = calculate(D(3000000), D(50000), RULES, [solar_claim()]).model_dump_json()
    for _ in range(200):
        assert calculate(D(3000000), D(50000), RULES, [solar_claim()]).model_dump_json() == first


# --- C3 -> ReliefRule -----------------------------------------------------------------------


def test_c3_solar_cap_is_read_from_the_structured_value():
    r = relief_from_retrieval(c3_rule())
    assert (r.rule_key, r.value_kind, r.value) == ("solar_panel_relief", "cap", D(600000))
    assert r.income_types == ("employment", "business", "investment", "other")
    assert r.ref.origin == "c3"
    assert r.ref.rule_id == "RULE-SOLAR_RELIEF-V1"
    assert "paragraph 2(g) of Act No. 24 of 2017" in r.ref.citation
    assert "national grid" in r.ref.evidence
    assert r.ref.evidence_chunk_ids == ("CHUNK-IRA-AMEND-10-2021-S055-2-P2",)
    assert "Who can claim" in r.text  # shown to the user as is, never parsed


def test_c3_rate_relief():
    rental = c3_rule(
        rule_id="RULE-RENTAL_RELIEF-V1",
        rule_key="rental_relief",
        values={"rate_percent": 25.0},
        applies_to={"income_types": ["investment"], "taxpayer_categories": ["resident_individual"]},
    )
    r = relief_from_retrieval(rental)
    assert (r.value_kind, r.value, r.income_types) == ("rate", D(25), ("investment",))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rule_key": "personal_relief"}, "personal relief"),
        ({"rule_type": "tax_slab"}, "not a relief"),
        ({"values": None}, "no single structured value"),
        (
            {"values": {"bands": [{"order": 1, "lower": 0, "upper": None, "rate_percent": 6}]}},
            "no single structured value",
        ),
        (
            {"values": {"rate_percent": 5, "cap": {"amount": 1, "currency": "LKR"}}},
            "exactly one",
        ),
    ],
)
def test_c3_reliefs_the_engine_cannot_apply_are_errors(changes, message):
    with pytest.raises(RuleSetError, match=message):
        relief_from_retrieval(c3_rule(**changes))


def test_seed_and_c3_solar_agree():
    assert relief_conflicts(load_seed_reliefs("2025/26"), relief_from_retrieval(c3_rule())) == []


def test_seed_and_c3_disagreement_is_reported():
    other = c3_rule(values={"cap": {"amount": 500000, "currency": "LKR"}})
    assert relief_conflicts((SOLAR,), relief_from_retrieval(other)) == [
        "solar_panel_relief: seed cap 600000, c3 cap 500000"
    ]


def test_relief_without_a_seed_has_nothing_to_cross_check():
    assert relief_conflicts((), relief_from_retrieval(c3_rule())) == []


# --- the seed relief ------------------------------------------------------------------------


@pytest.mark.parametrize("ya", ["2024/25", "2025/26"])
def test_seed_solar_matches_c3_ground_truth(ya):
    (solar,) = load_seed_reliefs(ya)
    with (GROUND_TRUTH / "relief_versions.csv").open(encoding="utf-8", newline="") as f:
        (truth,) = [r for r in csv.DictReader(f) if r["rule_key"] == "solar_panel_relief"]
    assert truth["verified"] == "yes"
    assert (solar.value_kind, solar.value) == (truth["value_kind"], D(truth["value"]))
    assert solar.ref.valid_from == date.fromisoformat(truth["valid_from"])
    assert solar.ref.valid_to is None and truth["valid_to"] == ""
    assert solar.ref.origin == "seed"
    assert solar.ref.citation == SOLAR.ref.citation  # same rule, same citation in both years
    assert "paragraph 2(g) of Act No. 24 of 2017" in solar.ref.citation


@pytest.mark.parametrize(
    "ya, without, with_solar",
    [
        # taxable 1,800,000 -> 30,000 + 60,000 + 90,000 + 300,000 @ 24% = 252,000;
        # with solar 1,200,000 -> 30,000 + 60,000 + 200,000 @ 18% = 126,000
        ("2024/25", D(252000), D(126000)),
        # taxable 1,200,000 -> 96,000; with solar 600,000 @ 6% = 36,000
        ("2025/26", D(96000), D(36000)),
    ],
)
def test_solar_applies_in_both_years(ya, without, with_solar):
    rules = load_seed_rules(ya)
    (solar,) = load_seed_reliefs(ya)
    claim = ReliefClaim(rule=solar, base=D(800000), base_source="user")
    assert calculate(D(3000000), D(0), rules).total_tax == without
    r = calculate(D(3000000), D(0), rules, [claim])
    assert step(r, "S1_SOLAR_PANEL_RELIEF").result == D(600000)
    assert r.total_tax == with_solar
    assert remaining_claimable(claim) == D(200000)
