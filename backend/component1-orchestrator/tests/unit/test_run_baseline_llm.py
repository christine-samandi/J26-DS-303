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
    # the API's own delay + 1 s; then the long overload wait (the 429 retry doesn't use up
    # the 503's single retry)
    assert sleeps == [3.5, baseline.OVERLOAD_WAIT_S]


@pytest.mark.parametrize("code", [503, 504])
def test_overload_is_retried_once_after_a_long_wait(one_scenario, tmp_path, sleeps, code):
    row, _ = run(one_scenario, tmp_path / "out", Flaky([APIError(code, "busy")], reply()))
    assert row["passed"] == "True"
    assert sleeps == [baseline.OVERLOAD_WAIT_S]


@pytest.mark.parametrize("code", [503, 504])
def test_api_error_is_excluded_from_accuracy_and_fails_the_run(
    one_scenario, tmp_path, sleeps, code
):
    client = Flaky([APIError(code, "busy")] * 2, reply())
    row, md = run(one_scenario, tmp_path / "out", client, exit_code=1)
    assert len(client.prompts) == 2  # one call + one retry, so little daily quota is spent
    assert row["passed"] == "False" and row["mismatches"] == ""
    assert row["api_error"].startswith(f"APIError: {code}")
    assert "0/0 = n/a" in md and "API errors, no answer (excluded): 1" in md


def test_non_transient_error_is_not_retried(one_scenario, tmp_path, sleeps):
    client = Flaky([TimeoutError("deadline exceeded")])
    row, _ = run(one_scenario, tmp_path / "out", client, exit_code=1)
    assert len(client.prompts) == 1 and sleeps == []
    assert row["api_error"] == "TimeoutError: deadline exceeded"


def test_daily_quota_is_not_retried(one_scenario, tmp_path, sleeps):
    daily = APIError(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    client = Flaky([daily], reply())
    row, _ = run(one_scenario, tmp_path / "out", client, exit_code=1)
    assert len(client.prompts) == 1 and sleeps == []
    assert "PerDay" in row["api_error"]


def test_pacer_spaces_calls_to_the_rpm_limit(monkeypatch, sleeps):
    now = iter([0.0, 3.0, 12.0])  # 1st call; 2nd call asks at t=3, resumes at t=12
    monkeypatch.setattr(baseline.time, "monotonic", lambda: next(now))
    pacer = baseline.Pacer(rpm=5)
    pacer.wait()
    pacer.wait()
    assert sleeps == [9.0]


# --- --only-missing and --merge --------------------------------------------------------------

GS007 = EVAL_DIR / "scenarios" / "GS-007_top_of_band3_apit_overpaid.json"


def answer_for(path: Path, **changes) -> str:
    """The scenario's verified totals as an LLM reply."""
    expected = json.loads(path.read_text())["expected"]
    return json.dumps({**{k: expected[k] for k in baseline.TOTALS}, **changes})


@pytest.fixture
def two_scenarios(tmp_path, monkeypatch):
    """GS-001 and GS-007, a clean git SHA, the fake model, and a clock that ticks a minute
    per run."""
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    for path in (GS001, GS007):
        shutil.copy(path, scenarios / path.name)
    monkeypatch.setattr(baseline, "git_sha", lambda: "abc1234")
    monkeypatch.setattr(baseline, "configured_model", lambda: FakeClient.model)  # not .env
    minutes = iter(range(60))
    real = baseline.datetime

    class Clock:
        @staticmethod
        def now(tz):
            return real(2026, 10, 16, 8, next(minutes), tzinfo=tz)

    monkeypatch.setattr(baseline, "datetime", Clock)
    return scenarios


def cli(scenarios, out, client, *flags):
    argv = ["--scenarios", str(scenarios), "--out", str(out), "--rpm", "0", *flags]
    return baseline.main(argv, client=client)


def rows(path: Path) -> dict[str, dict]:
    return {r["scenario_id"]: r for r in csv.DictReader(path.open())}


def merged(out: Path) -> tuple[dict[str, dict], str]:
    (csv_path,) = out.glob("baseline_merged_*.csv")
    return rows(csv_path), csv_path.with_suffix(".md").read_text()


def test_only_missing_asks_again_only_what_had_no_answer_then_merges(
    two_scenarios, tmp_path, sleeps
):
    out = tmp_path / "out"
    first = Flaky([APIError(504, "deadline")] * 2, answer_for(GS007))  # GS-001 fails twice
    assert cli(two_scenarios, out, first) == 1

    second = FakeClient(answer_for(GS001))
    assert cli(two_scenarios, out, second, "--only-missing", "--merge") == 0
    assert len(second.prompts) == 1 and "Rs. 3000000" in second.prompts[0]  # GS-001 only

    table, md = merged(out)
    assert table["GS-001"]["passed"] == table["GS-007"]["passed"] == "True"
    assert table["GS-007"]["source_run"] == "baseline_llm_20261016T080000Z_abc1234"
    assert table["GS-001"]["source_run"] == "baseline_llm_20261016T080100Z_abc1234"
    assert "2/2 = 100.0%" in md and "Runs combined (earliest answer per scenario wins): 2" in md
    assert "| Answer from |" in md and "`baseline_llm_20261016T080100Z_abc1234`" in md


def test_nothing_missing_makes_no_llm_calls(two_scenarios, tmp_path):
    out = tmp_path / "out"
    assert cli(two_scenarios, out, FakeClient(answer_for(GS001), answer_for(GS007))) == 0
    idle = FakeClient()  # would raise IndexError if asked anything
    assert cli(two_scenarios, out, idle, "--only-missing") == 0
    assert idle.prompts == []


def test_an_unparseable_reply_is_a_result_and_is_not_asked_again(two_scenarios, tmp_path):
    out = tmp_path / "out"
    cli(two_scenarios, out, FakeClient("no", "still no", answer_for(GS007)))
    idle = FakeClient()
    assert cli(two_scenarios, out, idle, "--only-missing", "--merge") == 0
    assert idle.prompts == []
    table, md = merged(out)
    assert table["GS-001"]["parsed"] == "False" and table["GS-001"]["passed"] == "False"
    assert "1/2 = 50.0%" in md and "Unparseable replies (counted wrong): 1" in md


def test_merge_keeps_the_earliest_answer_not_the_best(two_scenarios, tmp_path):
    out = tmp_path / "out"
    cli(two_scenarios, out, FakeClient(answer_for(GS001, total_tax=1), answer_for(GS007)))
    cli(two_scenarios, out, FakeClient(answer_for(GS001), answer_for(GS007)))
    assert cli(two_scenarios, out, None, "--merge") == 0
    table, _ = merged(out)
    assert table["GS-001"]["passed"] == "False"
    assert table["GS-001"]["source_run"].startswith("baseline_llm_20261016T080000Z")


def test_merge_alone_makes_no_llm_calls_and_flags_what_is_still_missing(
    two_scenarios, tmp_path, monkeypatch
):
    monkeypatch.setattr(baseline, "gemini_from_env", lambda: pytest.fail("no LLM for --merge"))
    out = tmp_path / "out"
    cli(two_scenarios, out, Flaky([TimeoutError("down")], answer_for(GS007)))
    assert cli(two_scenarios, out, None, "--merge") == 1
    table, md = merged(out)
    assert table["GS-001"]["api_error"] == baseline.NO_ANSWER
    assert table["GS-001"]["source_run"] == ""
    assert "API errors, no answer (excluded): 1" in md


def _first_run(two_scenarios, out):
    cli(two_scenarios, out, FakeClient(answer_for(GS001), answer_for(GS007)))
    (csv_path,) = out.glob("baseline_llm_*.csv")
    return csv_path


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"git_sha": "abc1234-dirty"}, "not from a clean commit (abc1234-dirty)"),
        ({"model_requested": "other-model"}, "different model (other-model)"),
        ({"prompt_version": "baseline-v0"}, "different prompt version (baseline-v0)"),
        ({"prompt_sha256": "0" * 16}, "different prompt text under the same version"),
    ],
)
def test_incomparable_runs_are_skipped(two_scenarios, tmp_path, change, reason, capsys):
    out = tmp_path / "out"
    meta = _first_run(two_scenarios, out).with_suffix(".meta.json")
    meta.write_text(json.dumps({**json.loads(meta.read_text()), **change}))
    retry = FakeClient(answer_for(GS001), answer_for(GS007))
    assert cli(two_scenarios, out, retry, "--only-missing") == 0
    assert len(retry.prompts) == 2  # nothing reusable, so both are asked
    assert reason in capsys.readouterr().out


def test_older_runs_are_read_from_their_markdown_header(two_scenarios, tmp_path):
    out = tmp_path / "out"
    csv_path = _first_run(two_scenarios, out)
    csv_path.with_suffix(".meta.json").unlink()  # runs before this change have no .meta.json
    idle = FakeClient()
    assert cli(two_scenarios, out, idle, "--only-missing") == 0
    assert idle.prompts == []


def test_runs_that_counted_api_errors_as_wrong_are_skipped(two_scenarios, tmp_path, capsys):
    out = tmp_path / "out"
    csv_path = _first_run(two_scenarios, out)
    table = list(csv.DictReader(csv_path.open()))
    with csv_path.open("w", newline="") as f:  # the bb4e596 layout: no api_error column
        fields = [k for k in table[0] if k != "api_error"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    retry = FakeClient(answer_for(GS001), answer_for(GS007))
    cli(two_scenarios, out, retry, "--only-missing")
    assert len(retry.prompts) == 2
    assert "scored before API errors were kept apart" in capsys.readouterr().out


def test_answers_to_a_since_corrected_scenario_are_not_reused(two_scenarios, tmp_path):
    out = tmp_path / "out"
    _first_run(two_scenarios, out)
    path = two_scenarios / GS001.name
    scenario = json.loads(path.read_text())
    scenario["expected"]["tax_payable"] = 46001  # the hand calculation was corrected
    path.write_text(json.dumps(scenario))
    retry = FakeClient(answer_for(GS001))
    cli(two_scenarios, out, retry, "--only-missing")
    assert len(retry.prompts) == 1 and "Rs. 3000000" in retry.prompts[0]


# --- IPv4 option -------------------------------------------------------------------------------


@pytest.fixture
def resolver(monkeypatch):
    """A fake socket.getaddrinfo that records the address family it was asked for."""
    families: list[int] = []

    def getaddrinfo(host, port, family=0, *args, **kwargs):
        families.append(family)
        return []

    monkeypatch.setattr(baseline.socket, "getaddrinfo", getaddrinfo)
    return families


def test_force_ipv4_resolves_ipv4_only_and_is_idempotent(resolver):
    baseline.force_ipv4()
    patched = baseline.socket.getaddrinfo
    baseline.force_ipv4()
    assert baseline.socket.getaddrinfo is patched
    baseline.socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    assert resolver == [baseline.socket.AF_INET]


@pytest.fixture
def gemini_env(monkeypatch, resolver):
    """gemini_from_env without .env, the SDK or the network."""
    monkeypatch.setattr(baseline, "configured_model", lambda: "fake-model")
    monkeypatch.setattr(baseline, "GeminiClient", lambda key, model: (key, model))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_FORCE_IPV4", raising=False)
    return resolver


def ipv4_forced() -> bool:
    return getattr(baseline.socket.getaddrinfo, "ipv4_only", False)


def test_ipv4_is_off_by_default(gemini_env):
    assert baseline.gemini_from_env() == ("test-key", "fake-model")
    assert not ipv4_forced()


def test_ipv4_flag_forces_ipv4(gemini_env):
    baseline.gemini_from_env(ipv4=True)
    assert ipv4_forced()


def test_ipv4_env_var_forces_ipv4(gemini_env, monkeypatch):
    monkeypatch.setenv("GEMINI_FORCE_IPV4", "1")
    baseline.gemini_from_env()
    assert ipv4_forced()


def test_ipv4_cli_flag_reaches_the_client_factory(two_scenarios, tmp_path, monkeypatch):
    asked: list[bool] = []

    def factory(ipv4=False):
        asked.append(ipv4)
        return FakeClient(answer_for(GS001), answer_for(GS007))

    monkeypatch.setattr(baseline, "gemini_from_env", factory)
    assert cli(two_scenarios, tmp_path / "out", None, "--ipv4") == 0
    assert asked == [True]
