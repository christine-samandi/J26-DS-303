"""Pure-LLM baseline: the LLM does the tax arithmetic itself, no engine.

    python eval/run_baseline_llm.py [--scenarios eval/scenarios] [--out eval/results]
    python eval/run_baseline_llm.py --only-missing [--merge]   # resume after API errors
    python eval/run_baseline_llm.py --merge                    # combine runs, no LLM calls

For each golden scenario the LLM gets the same YA rules the engine gets (relief + bands,
from the seed rules) and the scenario input, and must answer the four totals as JSON.
Scored exactly like run_engine.py: a scenario passes only if taxable income, total tax,
payable and refund all match the verified answer to the cent. Comparing this table with
the engine's is the core evidence for "the LLM never does arithmetic".

Config from the environment or the repo-root .env (git-ignored):
    GEMINI_API_KEY     required
    GEMINI_MODEL       default gemini-3.8-flash
    GEMINI_FORCE_IPV4  1 = connect over IPv4 only, same as --ipv4

Every raw response is saved next to the results for audit. Scenarios are synthetic, so
nothing here carries PII.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from run_engine import EVAL_DIR, TOTALS, git_sha, load_scenario, money

from tax_calculation_agent.ruleset import RuleSet, load_seed_rules

PROMPT_VERSION = "baseline-v1"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_RPM = 5.0  # Gemini free tier: 5 requests per minute per model
HTTP_TIMEOUT_MS = 60_000
# Extra attempts per HTTP status. Every failed call still spends the small daily quota, so
# overload (503/504) gets a single retry after a long wait; the per-minute rate limit (429)
# and 500 keep the original backoff. Anything else is not retried.
RETRIES = {429: 3, 500: 3, 503: 1, 504: 1}
OVERLOAD_CODES = {503, 504}
OVERLOAD_WAIT_S = 60.0
CSV_FIELDS = (
    "scenario_id",
    "verified",
    "passed",
    "parsed",
    *(f"{side}_{k}" for k in TOTALS for side in ("expected", "llm")),
    "mismatches",
    "api_error",
    "latency_s",
    "source_run",
)

PROMPT_TEMPLATE = """\
You are a Sri Lankan income tax calculator. Compute the income tax of a resident individual
for the year of assessment {ya}, using ONLY the rules below.

Rules:
- Personal relief: Rs. {relief} per year.
- Taxable income = employment income - personal relief, not below zero.
- Tax on taxable income, progressive bands:
{bands}
- Round each band's tax to the nearest cent (half up). Total tax = sum of the band taxes.
- APIT already paid is credited against the total tax.
  If total tax >= APIT: tax_payable = total tax - APIT, refund = 0.
  Otherwise: tax_payable = 0, refund = APIT - total tax.

Taxpayer:
- Employment income: Rs. {income}
- APIT paid: Rs. {apit}

Answer with JSON only, no other text, in exactly this form (plain numbers in LKR, no commas):
{{"taxable_income": <number>, "total_tax": <number>, "tax_payable": <number>, "refund": <number>}}
"""


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str) -> tuple[str, str]:
        """Return (response text, model version that actually answered)."""
        ...


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai  # imported here so the scorer is testable without the SDK
        from google.genai import types

        self.model = model
        # Without a timeout a dead connection hangs forever (seen with broken IPv6 routes).
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS)
        )
        self._config = types.GenerateContentConfig(
            temperature=0, seed=0, response_mime_type="application/json"
        )

    def generate(self, prompt: str) -> tuple[str, str]:
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config
        )
        return response.text or "", response.model_version or self.model


class Pacer:
    """Spaces LLM calls so the run stays under the API's requests-per-minute quota."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60 / rpm if rpm > 0 else 0.0
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._last = time.monotonic()


def retry_delay(exc: Exception, retry: int) -> float:
    """The delay the API asks for ("Please retry in 56.4s"), else exponential backoff.

    Overload errors wait at least OVERLOAD_WAIT_S: retrying a busy model quickly just
    spends quota on another failure.
    """
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    delay = float(match.group(1)) + 1 if match else 10.0 * 2 ** (retry - 1)
    if getattr(exc, "code", None) in OVERLOAD_CODES:
        delay = max(delay, OVERLOAD_WAIT_S)
    return delay


def call_llm(client: LLMClient, prompt: str, pacer: Pacer) -> tuple[str, str]:
    """client.generate, retried per RETRIES (counted per status) on rate-limit/overload
    errors. Anything else is raised."""
    retried: Counter[int | None] = Counter()
    while True:
        pacer.wait()
        try:
            return client.generate(prompt)
        except Exception as exc:
            code = getattr(exc, "code", None)
            # The daily quota only resets at midnight Pacific: waiting inside a run can't help.
            if "PerDay" in str(exc) or retried[code] >= RETRIES.get(code, 0):
                raise
            retried[code] += 1
            time.sleep(retry_delay(exc, retried[code]))


@dataclass
class Outcome:
    scenario_id: str
    verified: bool
    expected: dict
    answer: dict = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    model_version: str = ""
    api_error: str = ""
    latency_s: float = 0.0
    source_run: str = ""  # the run (results file stem) this answer came from

    @property
    def parsed(self) -> bool:
        return bool(self.answer)

    @property
    def passed(self) -> bool:
        return self.parsed and not self.mismatches


def build_prompt(scenario: dict, rules: RuleSet) -> str:
    bands = []
    for b in rules.tax_bands.bands:
        span = "and above" if b.upper is None else f"to {b.upper}"
        bands.append(f"  - taxable income from {b.lower} {span}: {b.rate_percent}%")
    return PROMPT_TEMPLATE.format(
        ya=scenario["year_of_assessment"],
        relief=rules.personal_relief.amount,
        bands="\n".join(bands),
        income=scenario["input"]["employment_income"],
        apit=scenario["input"]["apit_paid"],
    )


def parse_answer(text: str) -> dict[str, Decimal]:
    """The four totals as Decimal, or ValueError if the reply isn't the JSON we asked for."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(cleaned, parse_float=Decimal)
        return {k: Decimal(data[k]) for k in TOTALS}
    except (json.JSONDecodeError, KeyError, TypeError, InvalidOperation) as exc:
        raise ValueError(f"unparseable reply: {type(exc).__name__}") from exc


def new_outcome(scenario: dict) -> Outcome:
    return Outcome(
        scenario_id=scenario["scenario_id"],
        verified=scenario["verification"]["status"] == "verified",
        expected={k: Decimal(scenario["expected"][k]) for k in TOTALS},
    )


def score(outcome: Outcome) -> None:
    """Add a mismatch for every parsed total that differs from the expected one, to the cent."""
    for k in TOTALS if outcome.parsed else ():
        if outcome.answer[k] != outcome.expected[k]:
            outcome.mismatches.append(
                f"{k}: expected {money(outcome.expected[k])}, got {outcome.answer[k]}"
            )


def evaluate(scenario: dict, client: LLMClient, pacer: Pacer) -> Outcome:
    outcome = new_outcome(scenario)
    prompt = build_prompt(scenario, load_seed_rules(scenario["year_of_assessment"]))

    start = time.perf_counter()
    for _ in range(2):  # one retry on a malformed reply, as for the interpretation agent
        try:
            text, outcome.model_version = call_llm(client, prompt, pacer)
        except Exception as exc:  # no answer at all: excluded from accuracy, not counted wrong
            outcome.api_error = f"{type(exc).__name__}: {exc}"
            outcome.mismatches = []
            break
        outcome.raw.append(text)
        try:
            outcome.answer = parse_answer(text)
            outcome.mismatches = []
            break
        except ValueError as exc:
            outcome.mismatches = [str(exc)]
    outcome.latency_s = round(time.perf_counter() - start, 3)
    score(outcome)
    return outcome


def write_csv(path: Path, outcomes: list[Outcome]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for o in outcomes:
            row = {
                "scenario_id": o.scenario_id,
                "verified": o.verified,
                "passed": o.passed,
                "parsed": o.parsed,
                "mismatches": "; ".join(o.mismatches),
                "api_error": o.api_error,
                "latency_s": o.latency_s,
                "source_run": o.source_run,
            }
            for k in TOTALS:
                row[f"expected_{k}"] = money(o.expected[k])
                row[f"llm_{k}"] = o.answer.get(k, "")
            writer.writerow(row)


def write_raw(path: Path, outcomes: list[Outcome]) -> None:
    with path.open("w") as f:
        for o in outcomes:
            record = {
                "scenario_id": o.scenario_id,
                "source_run": o.source_run,
                "model_version": o.model_version,
                "raw": o.raw,
            }
            f.write(json.dumps(record) + "\n")


def write_markdown(
    path: Path,
    outcomes: list[Outcome],
    model: str,
    sha: str,
    run_at: str,
    merged_from: list[str] | None = None,
) -> None:
    """The results table. `merged_from` lists the combined runs and adds an "Answer from" column."""
    verified = [o for o in outcomes if o.verified]
    answered = [o for o in verified if not o.api_error]
    passed = sum(o.passed for o in answered)
    accuracy = f"{100 * passed / len(answered):.1f}%" if answered else "n/a"
    versions = sorted({o.model_version for o in outcomes if o.model_version}) or ["—"]
    title = "Pure-LLM baseline calculation accuracy" + (" (merged runs)" if merged_from else "")
    lines = [
        f"# {title}",
        "",
        f"- {'Merged' if merged_from else 'Run'}: {run_at}",
        f"- Git SHA: `{sha}`",
    ]
    if merged_from:
        lines.append(f"- Runs combined (earliest answer per scenario wins): {len(merged_from)}")
        lines += [f"  - `{run}`" for run in merged_from]
    lines += [
        f"- Model requested: `{model}`; answered by: {', '.join(f'`{v}`' for v in versions)}",
        f"- Prompt: `{PROMPT_VERSION}` (sha256 `{prompt_sha256()}`), temperature 0, seed 0,"
        " seed rules given in the prompt",
        f"- **Accuracy (verified scenarios answered): {passed}/{len(answered)} = {accuracy}**",
        f"- Unparseable replies (counted wrong): {sum(not o.parsed for o in answered)}",
        f"- API errors, no answer (excluded): {len(verified) - len(answered)}",
        f"- Unverified scenarios (excluded): {len(outcomes) - len(verified)}",
        "",
    ]
    source = ["Answer from"] if merged_from else []
    columns = [
        "Scenario",
        "Verified",
        "Pass",
        "Expected tax",
        "LLM tax",
        "Expected payable",
        "LLM payable",
        "Expected refund",
        "LLM refund",
        *source,
        "Mismatches",
    ]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for o in outcomes:
        notes = f"API error: {o.api_error}" if o.api_error else "; ".join(o.mismatches)
        cells = [
            o.scenario_id,
            "yes" if o.verified else "no",
            "⚠️" if o.api_error else "✅" if o.passed else "❌",
            money(o.expected["total_tax"]),
            o.answer.get("total_tax", "—"),
            money(o.expected["tax_payable"]),
            o.answer.get("tax_payable", "—"),
            money(o.expected["refund"]),
            o.answer.get("refund", "—"),
            *([f"`{o.source_run}`" if o.source_run else "—"] if merged_from else []),
            notes.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def prompt_sha256() -> str:
    """Identifies the prompt text, so runs are only combined if it really is the same."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16]


def write_outputs(
    stem: Path, outcomes: list[Outcome], model: str, sha: str, run_at: str, **md
) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(stem.with_suffix(".csv"), outcomes)
    write_raw(stem.with_suffix(".raw.jsonl"), outcomes)
    write_markdown(stem.with_suffix(".md"), outcomes, model, sha, run_at, **md)
    return stem.with_suffix(".md")


def load_scenarios(scenario_dir: Path) -> dict[str, dict]:
    paths = sorted(scenario_dir.glob("GS-*.json"))
    if not paths:
        raise SystemExit(f"no golden scenarios found in {scenario_dir}")
    return {s["scenario_id"]: s for s in map(load_scenario, paths)}


def run(
    scenario_dir: Path,
    out_dir: Path,
    client: LLMClient,
    rpm: float = DEFAULT_RPM,
    skip: Collection[str] = (),
) -> tuple[list[Outcome], Path]:
    """Ask the LLM every scenario except `skip`; write the run's results and metadata."""
    scenarios = [s for sid, s in load_scenarios(scenario_dir).items() if sid not in skip]
    pacer = Pacer(rpm)
    outcomes = [evaluate(s, client, pacer) for s in scenarios]

    now = datetime.now(UTC)
    sha = git_sha()
    run_at = now.isoformat(timespec="seconds")
    stem = out_dir / f"{RUN_PREFIX}{now:%Y%m%dT%H%M%SZ}_{sha}"
    for o in outcomes:
        o.source_run = stem.name
    md_path = write_outputs(stem, outcomes, client.model, sha, run_at)
    meta = {
        "run_at": run_at,
        "git_sha": sha,
        "model_requested": client.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "scenarios": [o.scenario_id for o in outcomes],
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return outcomes, md_path


# --- combining runs ----------------------------------------------------------------------

RUN_PREFIX = "baseline_llm_"  # single runs; merged tables use MERGED_PREFIX, so never re-read
MERGED_PREFIX = "baseline_merged_"
NO_ANSWER = "no answer in any comparable run"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    git_sha: str
    model: str
    prompt_version: str
    prompt_sha256: str | None  # None for runs made before it was recorded


def read_run_info(csv_path: Path) -> RunInfo | None:
    """From the run's .meta.json, or for older runs from its markdown header."""
    stem = csv_path.with_suffix("")
    meta = stem.with_suffix(".meta.json")
    if meta.exists():
        m = json.loads(meta.read_text())
        return RunInfo(
            stem.name, m["git_sha"], m["model_requested"], m["prompt_version"], m["prompt_sha256"]
        )
    md = stem.with_suffix(".md")
    if not md.exists():
        return None
    text = md.read_text()
    found = [
        re.search(pattern, text, re.M)
        for pattern in (
            r"^- Git SHA: `([^`]+)`",
            r"^- Model requested: `([^`]+)`",
            r"^- Prompt: `([^`]+)`",
        )
    ]
    if not all(found):
        return None
    sha, model, version = (f.group(1) for f in found)
    return RunInfo(stem.name, sha, model, version, None)


def why_not_comparable(info: RunInfo | None, header: list[str], model: str) -> str | None:
    if info is None:
        return "no run metadata"
    if "api_error" not in header:
        return "scored before API errors were kept apart from wrong answers"
    if info.git_sha.endswith("-dirty") or info.git_sha == "unknown":
        return f"not from a clean commit ({info.git_sha})"
    if info.model != model:
        return f"different model ({info.model})"
    if info.prompt_version != PROMPT_VERSION:
        return f"different prompt version ({info.prompt_version})"
    if info.prompt_sha256 not in (None, prompt_sha256()):
        return "different prompt text under the same version"
    return None


def earlier_answers(
    out_dir: Path, model: str, scenarios: dict[str, dict]
) -> tuple[dict[str, Outcome], list[str]]:
    """The earliest reply per scenario from comparable earlier runs, rescored against today's
    scenarios, plus why each other run was skipped.

    A reply counts if the API answered at all: an unparseable reply is a real (wrong) result,
    so it is never asked again; only API errors are. The earliest reply wins, never the best
    one, so resuming cannot pick favourable answers. A row whose expected totals differ from
    the scenario's (the hand calculation was corrected since) is ignored.
    """
    answers: dict[str, Outcome] = {}
    skipped: list[str] = []
    for path in sorted(out_dir.glob(f"{RUN_PREFIX}*.csv")):  # names sort by run time
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = list(reader.fieldnames or [])
        info = read_run_info(path)
        if reason := why_not_comparable(info, header, model):
            skipped.append(f"{path.stem}: {reason}")
            continue
        raw = _read_raw(path.with_suffix(".raw.jsonl"))
        for row in rows:
            sid = row["scenario_id"]
            if sid in answers or sid not in scenarios or row["api_error"]:
                continue
            outcome = new_outcome(scenarios[sid])
            if any(row[f"expected_{k}"] != money(outcome.expected[k]) for k in TOTALS):
                continue
            outcome.source_run = info.run_id
            outcome.model_version = raw.get(sid, {}).get("model_version", "")
            outcome.raw = raw.get(sid, {}).get("raw", [])
            outcome.latency_s = float(row["latency_s"] or 0)
            if row["parsed"] == "True":
                outcome.answer = {k: Decimal(row[f"llm_{k}"]) for k in TOTALS}
                score(outcome)
            else:
                outcome.mismatches = [m for m in row["mismatches"].split("; ") if m]
            answers[sid] = outcome
    return answers, skipped


def merge(scenario_dir: Path, out_dir: Path, model: str) -> tuple[list[Outcome], Path, list[str]]:
    """One results table from all comparable runs, each answer labelled with its run."""
    scenarios = load_scenarios(scenario_dir)
    answers, skipped = earlier_answers(out_dir, model, scenarios)
    outcomes = []
    for sid, scenario in scenarios.items():
        if sid in answers:
            outcomes.append(answers[sid])
        else:
            missing = new_outcome(scenario)
            missing.api_error = NO_ANSWER
            outcomes.append(missing)
    now = datetime.now(UTC)
    sha = git_sha()
    runs = sorted({o.source_run for o in outcomes if o.source_run})
    stem = out_dir / f"{MERGED_PREFIX}{now:%Y%m%dT%H%M%SZ}_{sha}"
    md_path = write_outputs(
        stem, outcomes, model, sha, now.isoformat(timespec="seconds"), merged_from=runs
    )
    return outcomes, md_path, skipped


def _read_raw(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["scenario_id"]: r for r in records}


# --- command line ------------------------------------------------------------------------


def configured_model() -> str:
    from dotenv import load_dotenv

    load_dotenv(EVAL_DIR.parent / ".env")
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def force_ipv4() -> None:
    """Resolve every host to IPv4 addresses only, for this process.

    On some networks IPv6 to Google is routed but dead: Python tries it first and each call
    hangs until the timeout. Idempotent.
    """
    original = socket.getaddrinfo
    if getattr(original, "ipv4_only", False):
        return

    def ipv4_only(host, port, family=0, *args, **kwargs):
        return original(host, port, socket.AF_INET, *args, **kwargs)

    ipv4_only.ipv4_only = True  # type: ignore[attr-defined]
    socket.getaddrinfo = ipv4_only


def gemini_from_env(ipv4: bool = False) -> GeminiClient:
    model = configured_model()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set (add it to the repo-root .env)")
    if ipv4 or os.environ.get("GEMINI_FORCE_IPV4", "").strip() == "1":
        force_ipv4()
        print("Connecting over IPv4 only")
    return GeminiClient(api_key, model)


def report(label: str, outcomes: list[Outcome], md_path: Path) -> int:
    """Print the summary. Exit code 1 if any verified scenario still has no answer."""
    verified = [o for o in outcomes if o.verified]
    answered = [o for o in verified if not o.api_error]
    failed = [o for o in answered if not o.passed]
    print(f"{label}: {len(answered) - len(failed)}/{len(answered)} correct")
    for o in failed:
        print(f"  WRONG {o.scenario_id}: {'; '.join(o.mismatches)}")
    for o in verified:
        if o.api_error:
            print(f"  NO ANSWER {o.scenario_id}: {o.api_error[:200]}")
    print(f"Wrote {md_path} (+ .csv, .raw.jsonl)")
    # Wrong LLM answers are the result being measured; missing answers mean the run is incomplete.
    return 1 if len(answered) < len(verified) else 0


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=Path, default=EVAL_DIR / "scenarios")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument(
        "--rpm", type=float, default=DEFAULT_RPM, help="max LLM requests per minute (0 = no limit)"
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="ask only scenarios with no answer in earlier runs of the same model and prompt",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="combine all comparable runs into one table (no LLM calls unless --only-missing)",
    )
    parser.add_argument(
        "--ipv4",
        action="store_true",
        help="connect to the API over IPv4 only (if IPv6 hangs); or set GEMINI_FORCE_IPV4=1",
    )
    args = parser.parse_args(argv)

    code = 0
    if args.only_missing or not args.merge:
        client = client or gemini_from_env(args.ipv4)
        skip: set[str] = set()
        if args.only_missing:
            scenarios = load_scenarios(args.scenarios)
            answers, skipped = earlier_answers(args.out, client.model, scenarios)
            for s in skipped:
                print(f"  skipped earlier run {s}")
            skip = set(answers)
            print(f"Already answered: {len(skip)}/{len(scenarios)}; asking the rest")
        if args.only_missing and skip >= set(load_scenarios(args.scenarios)):
            print("Nothing missing: no LLM calls made")
        else:
            outcomes, md_path = run(args.scenarios, args.out, client, args.rpm, skip)
            code = report(f"LLM baseline ({client.model})", outcomes, md_path)

    if args.merge:
        model = client.model if client else configured_model()
        outcomes, md_path, skipped = merge(args.scenarios, args.out, model)
        for s in skipped:
            print(f"  skipped run {s}")
        code = report(f"Merged LLM baseline ({model})", outcomes, md_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
