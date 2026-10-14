"""eval/run_baseline_llm.py, scored against fake LLM clients (no network, no API key)."""

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[4] / "eval"
sys.path.insert(0, str(EVAL_DIR))
import run_baseline_llm as baseline  # noqa: E402

GS001 = EVAL_DIR / "scenarios" / "GS-001_band2_apit_underpaid.json"


class FakeClient:
    model = "fake-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        return self.replies.pop(0), "fake-model-001"


def reply(**totals) -> str:
    answer = {"taxable_income": 1200000, "total_tax": 96000, "tax_payable": 46000, "refund": 0}
    return json.dumps({**answer, **totals})


@pytest.fixture
def one_scenario(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    shutil.copy(GS001, scenarios / GS001.name)
    return scenarios


def run(scenarios, out, client):
    assert baseline.main(["--scenarios", str(scenarios), "--out", str(out)], client=client) == 0
    (row,) = csv.DictReader(next(out.glob("baseline_llm_*.csv")).open())
    return row, next(out.glob("baseline_llm_*.md")).read_text()


def test_prompt_gives_rules_and_input_but_not_the_answer():
    prompt = baseline.build_prompt(
        baseline.load_scenario(GS001), baseline.load_seed_rules("2025/26")
    )
    assert "Rs. 1800000" in prompt and "Rs. 3000000" in prompt and "Rs. 50000" in prompt
    assert "from 2500000 and above: 36%" in prompt
    assert "96000" not in prompt and "46000" not in prompt


def test_correct_answer_passes(one_scenario, tmp_path):
    row, md = run(one_scenario, tmp_path / "out", FakeClient(reply()))
    assert row["passed"] == "True"
    assert "1/1 = 100.0%" in md
    assert "answered by: `fake-model-001`" in md


def test_wrong_arithmetic_fails_with_the_difference(one_scenario, tmp_path):
    row, md = run(one_scenario, tmp_path / "out", FakeClient(reply(total_tax=96000.5)))
    assert row["passed"] == "False"
    assert "total_tax: expected 96000.00, got 96000.5" in row["mismatches"]
    assert "0/1 = 0.0%" in md


def test_malformed_reply_is_retried_once(one_scenario, tmp_path):
    client = FakeClient("The tax is 96,000.", f"```json\n{reply()}\n```")
    row, _ = run(one_scenario, tmp_path / "out", client)
    assert len(client.prompts) == 2
    assert row["passed"] == "True"


def test_two_malformed_replies_count_as_wrong(one_scenario, tmp_path):
    row, md = run(one_scenario, tmp_path / "out", FakeClient("no", '{"total_tax": 96000}'))
    assert row["passed"] == "False" and row["parsed"] == "False"
    assert "Unparseable replies: 1" in md

    (raw,) = (tmp_path / "out").glob("*.raw.jsonl")
    assert json.loads(raw.read_text())["raw"] == ["no", '{"total_tax": 96000}']


def test_api_error_counts_as_wrong_not_a_crash(one_scenario, tmp_path):
    class Broken(FakeClient):
        def generate(self, prompt):
            raise TimeoutError("deadline exceeded")

    row, _ = run(one_scenario, tmp_path / "out", Broken())
    assert row["passed"] == "False"
    assert "llm error: TimeoutError" in row["mismatches"]
