"""The engine must match every golden scenario in eval/scenarios/ exactly.

Scenarios cite rule keys, not rule ids, so they hold for seed and C3 rules alike.
"""

import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from tax_calculation_agent.engine import calculate
from tax_calculation_agent.ruleset import RuleSet, load_seed_rules

SCENARIO_DIR = Path(__file__).resolve().parents[4] / "eval" / "scenarios"
SCENARIO_FILES = sorted(SCENARIO_DIR.glob("GS-*.json"))


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f, parse_float=D)


def rule_ids_by_key(rules: RuleSet) -> dict[str, str]:
    return {
        "personal_relief": rules.personal_relief.ref.rule_id,
        "individual_income_tax_slabs": rules.tax_bands.ref.rule_id,
    }


def test_scenarios_exist():
    assert SCENARIO_FILES, f"no golden scenarios found in {SCENARIO_DIR}"


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_id_matches_file_name(path):
    assert path.name.startswith(load(path)["scenario_id"] + "_")


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_engine_matches_scenario(path):
    scenario = load(path)
    rules = load_seed_rules(scenario["year_of_assessment"])
    ids = rule_ids_by_key(rules)
    given, expected = scenario["input"], scenario["expected"]

    result = calculate(D(given["employment_income"]), D(given["apit_paid"]), rules)

    for field in ("taxable_income", "total_tax", "tax_payable", "refund"):
        assert getattr(result, field) == D(expected[field]), field

    actual_steps = [(s.step_id, s.result, s.citation_refs) for s in result.steps]
    expected_steps = [
        (s["step_id"], D(s["result"]), tuple(ids[k] for k in s["cites"]))
        for s in expected["steps"]
    ]
    assert actual_steps == expected_steps
