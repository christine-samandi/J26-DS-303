"""Properties that must hold for any income (Hypothesis), plus reproducibility and purity."""

import ast
from decimal import Decimal as D
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

import tax_calculation_agent
from tax_calculation_agent.engine import calculate
from tax_calculation_agent.ruleset import load_seed_rules

RULES = load_seed_rules("2025/26")
RELIEF = RULES.personal_relief.amount
TOP_RATE = RULES.tax_bands.bands[-1].rate_percent

# Gross employment income in LKR, cents included, up to 100 million.
gross_income = st.decimals(min_value=0, max_value=100_000_000, places=2)
apit = st.decimals(min_value=0, max_value=50_000_000, places=2)


def tax(gross):
    return calculate(gross, D(0), RULES).total_tax


@given(gross_income, gross_income)
def test_tax_never_decreases_as_gross_income_rises(a, b):
    low, high = sorted((a, b))
    assert tax(low) <= tax(high)


@given(st.decimals(min_value=0, max_value=RELIEF, places=2))
def test_gross_at_or_below_relief_gives_zero_tax(gross):
    assert tax(gross) == 0


@given(gross_income, st.decimals(min_value=D("0.01"), max_value=1_000_000, places=2))
def test_marginal_rate_never_exceeds_top_band(gross, extra):
    # Each band tax is rounded to the cent, so allow one cent per band of rounding drift.
    drift = D("0.01") * len(RULES.tax_bands.bands)
    assert tax(gross + extra) - tax(gross) <= extra * TOP_RATE / 100 + drift


@given(gross_income)
def test_tax_never_exceeds_taxable_income_times_top_rate(gross):
    r = calculate(gross, D(0), RULES)
    assert r.total_tax <= r.taxable_income * TOP_RATE / 100


@given(gross_income, apit)
def test_payable_and_refund_balance_against_apit(gross, paid):
    r = calculate(gross, paid, RULES)
    assert r.tax_payable >= 0 and r.refund >= 0
    assert r.tax_payable == 0 or r.refund == 0
    assert r.total_tax - paid == r.tax_payable - r.refund


@given(gross_income)
def test_total_tax_is_sum_of_band_steps(gross):
    r = calculate(gross, D(0), RULES)
    bands = [s.result for s in r.steps if s.step_id.startswith("S3_BAND_")]
    assert r.total_tax == sum(bands, D(0))


def test_1000_runs_give_identical_results():
    args = (D("4300000.37"), D("123456.78"), RULES)
    first = calculate(*args)
    for _ in range(1000):
        again = calculate(*args)
        assert again == first
        assert again.model_dump_json() == first.model_dump_json()


LLM_LIBRARIES = {
    "anthropic",
    "openai",
    "google",
    "langchain",
    "langchain_core",
    "langgraph",
    "crewai",
    "transformers",
    "litellm",
    "ollama",
    "mistralai",
    "cohere",
}


def test_engine_package_imports_no_llm_library():
    package_dir = Path(tax_calculation_agent.__file__).parent
    found = []
    for path in package_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            found += [f"{path.name}: {n}" for n in names if n.split(".")[0] in LLM_LIBRARIES]
    assert not found, f"LLM imports in the engine: {found}"
