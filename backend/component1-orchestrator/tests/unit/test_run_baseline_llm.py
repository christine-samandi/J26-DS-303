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


def run(scenarios, out, client, exit_code=0):
    argv = ["--scenarios", str(scenarios), "--out", str(out), "--rpm", "0"]
    assert baseline.main(argv, client=client) == exit_code
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
    assert "Unparseable replies (counted wrong): 1" in md

    (raw,) = (tmp_path / "out").glob("*.raw.jsonl")
    assert json.loads(raw.read_text())["raw"] == ["no", '{"total_tax": 96000}']


class APIError(Exception):
    """Stands in for google.genai.errors.APIError, which carries the HTTP status as .code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class Flaky(FakeClient):
    """Raises the given errors first, then returns the replies."""

    def __init__(self, errors, *replies):
        super().__init__(*replies)
        self.errors = list(errors)

    def generate(self, prompt):
        if self.errors:
            self.prompts.append(prompt)
            raise self.errors.pop(0)
        return super().generate(prompt)


@pytest.fixture
def sleeps(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(baseline.time, "sleep", slept.append)
    return slept


def test_rate_limit_waits_as_asked_then_retries(one_scenario, tmp_path, sleeps):
    errors = [APIError(429, "Quota exceeded. Please retry in 2.5s."), APIError(503, "overloaded")]
    row, _ = run(one_scenario, tmp_path / "out", Flaky(errors, reply()))
    assert row["passed"] == "True" and row["api_error"] == ""
    assert sleeps == [3.5, 20.0]  # the API's own delay + 1 s, then backoff for attempt 2


def test_api_error_is_excluded_from_accuracy_and_fails_the_run(one_scenario, tmp_path, sleeps):
    client = Flaky([APIError(503, "overloaded")] * baseline.MAX_API_ATTEMPTS)
    row, md = run(one_scenario, tmp_path / "out", client, exit_code=1)
    assert len(client.prompts) == baseline.MAX_API_ATTEMPTS
    assert row["passed"] == "False" and row["mismatches"] == ""
    assert row["api_error"].startswith("APIError: 503")
    assert "0/0 = n/a" in md and "API errors, no answer (excluded): 1" in md


def test_non_transient_error_is_not_retried(one_scenario, tmp_path, sleeps):
    client = Flaky([TimeoutError("deadline exceeded")])
    row, _ = run(one_scenario, tmp_path / "out", client, exit_code=1)
    assert len(client.prompts) == 1 and sleeps == []
    assert row["api_error"] == "TimeoutError: deadline exceeded"


def test_pacer_spaces_calls_to_the_rpm_limit(monkeypatch, sleeps):
    now = iter([0.0, 3.0, 12.0])  # 1st call; 2nd call asks at t=3, resumes at t=12
    monkeypatch.setattr(baseline.time, "monotonic", lambda: next(now))
    pacer = baseline.Pacer(rpm=5)
    pacer.wait()
    pacer.wait()
    assert sleeps == [9.0]
