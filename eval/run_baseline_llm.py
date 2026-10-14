"""Pure-LLM baseline: the LLM does the tax arithmetic itself, no engine.

    python eval/run_baseline_llm.py [--scenarios eval/scenarios] [--out eval/results]

For each golden scenario the LLM gets the same YA rules the engine gets (relief + bands,
from the seed rules) and the scenario input, and must answer the four totals as JSON.
Scored exactly like run_engine.py: a scenario passes only if taxable income, total tax,
payable and refund all match the verified answer to the cent. Comparing this table with
the engine's is the core evidence for "the LLM never does arithmetic".

Config from the environment or the repo-root .env (git-ignored):
    GEMINI_API_KEY   required
    GEMINI_MODEL     default gemini-3.8-flash

Every raw response is saved next to the results for audit. Scenarios are synthetic, so
nothing here carries PII.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
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
TRANSIENT_CODES = {429, 500, 503}  # rate limit / overloaded: wait and call again
MAX_API_ATTEMPTS = 4
CSV_FIELDS = (
    "scenario_id",
    "verified",
    "passed",
    "parsed",
    *(f"{side}_{k}" for k in TOTALS for side in ("expected", "llm")),
    "mismatches",
    "api_error",
    "latency_s",
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


def retry_delay(exc: Exception, attempt: int) -> float:
    """The delay the API asks for ("Please retry in 56.4s"), else exponential backoff."""
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 1 if match else 10.0 * 2 ** (attempt - 1)


def call_llm(client: LLMClient, prompt: str, pacer: Pacer) -> tuple[str, str]:
    """client.generate, retried on rate-limit/overload errors. Anything else is raised."""
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        pacer.wait()
        try:
            return client.generate(prompt)
        except Exception as exc:
            transient = getattr(exc, "code", None) in TRANSIENT_CODES
            # The daily quota only resets at midnight Pacific: waiting inside a run can't help.
            if not transient or "PerDay" in str(exc) or attempt == MAX_API_ATTEMPTS:
                raise
            time.sleep(retry_delay(exc, attempt))
    raise AssertionError("unreachable")


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


def evaluate(scenario: dict, client: LLMClient, pacer: Pacer) -> Outcome:
    outcome = Outcome(
        scenario_id=scenario["scenario_id"],
        verified=scenario["verification"]["status"] == "verified",
        expected={k: Decimal(scenario["expected"][k]) for k in TOTALS},
    )
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

    for k in TOTALS if outcome.parsed else ():
        if outcome.answer[k] != outcome.expected[k]:
            outcome.mismatches.append(
                f"{k}: expected {money(outcome.expected[k])}, got {outcome.answer[k]}"
            )
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
            }
            for k in TOTALS:
                row[f"expected_{k}"] = money(o.expected[k])
                row[f"llm_{k}"] = o.answer.get(k, "")
            writer.writerow(row)


def write_raw(path: Path, outcomes: list[Outcome]) -> None:
    with path.open("w") as f:
        for o in outcomes:
            record = {"scenario_id": o.scenario_id, "model_version": o.model_version, "raw": o.raw}
            f.write(json.dumps(record) + "\n")


def write_markdown(path: Path, outcomes: list[Outcome], model: str, sha: str, run_at: str) -> None:
    verified = [o for o in outcomes if o.verified]
    answered = [o for o in verified if not o.api_error]
    passed = sum(o.passed for o in answered)
    accuracy = f"{100 * passed / len(answered):.1f}%" if answered else "n/a"
    versions = sorted({o.model_version for o in outcomes if o.model_version}) or ["—"]
    lines = [
        "# Pure-LLM baseline calculation accuracy",
        "",
        f"- Run: {run_at}",
        f"- Git SHA: `{sha}`",
        f"- Model requested: `{model}`; answered by: {', '.join(f'`{v}`' for v in versions)}",
        f"- Prompt: `{PROMPT_VERSION}`, temperature 0, seed 0, seed rules given in the prompt",
        f"- **Accuracy (verified scenarios answered): {passed}/{len(answered)} = {accuracy}**",
        f"- Unparseable replies (counted wrong): {sum(not o.parsed for o in answered)}",
        f"- API errors, no answer (excluded): {len(verified) - len(answered)}",
        f"- Unverified scenarios (excluded): {len(outcomes) - len(verified)}",
        "",
        "| Scenario | Verified | Pass | Expected tax | LLM tax | Expected payable"
        " | LLM payable | Expected refund | LLM refund | Mismatches |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
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
            notes.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def run(
    scenario_dir: Path, out_dir: Path, client: LLMClient, rpm: float = DEFAULT_RPM
) -> tuple[list[Outcome], Path]:
    paths = sorted(scenario_dir.glob("GS-*.json"))
    if not paths:
        raise SystemExit(f"no golden scenarios found in {scenario_dir}")
    pacer = Pacer(rpm)
    outcomes = [evaluate(load_scenario(p), client, pacer) for p in paths]

    now = datetime.now(UTC)
    sha = git_sha()
    stem = out_dir / f"baseline_llm_{now:%Y%m%dT%H%M%SZ}_{sha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(stem.with_suffix(".csv"), outcomes)
    write_raw(stem.with_suffix(".raw.jsonl"), outcomes)
    write_markdown(
        stem.with_suffix(".md"), outcomes, client.model, sha, now.isoformat(timespec="seconds")
    )
    return outcomes, stem.with_suffix(".md")


def gemini_from_env() -> GeminiClient:
    from dotenv import load_dotenv

    load_dotenv(EVAL_DIR.parent / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set (add it to the repo-root .env)")
    return GeminiClient(api_key, os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=Path, default=EVAL_DIR / "scenarios")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument(
        "--rpm", type=float, default=DEFAULT_RPM, help="max LLM requests per minute (0 = no limit)"
    )
    args = parser.parse_args(argv)

    client = client or gemini_from_env()
    outcomes, md_path = run(args.scenarios, args.out, client, args.rpm)
    verified = [o for o in outcomes if o.verified]
    answered = [o for o in verified if not o.api_error]
    failed = [o for o in answered if not o.passed]
    print(f"LLM baseline ({client.model}): {len(answered) - len(failed)}/{len(answered)} correct")
    for o in failed:
        print(f"  WRONG {o.scenario_id}: {'; '.join(o.mismatches)}")
    for o in verified:
        if o.api_error:
            print(f"  NO ANSWER {o.scenario_id}: {o.api_error[:200]}")
    print(f"Wrote {md_path} (+ .csv, .raw.jsonl)")
    # Wrong LLM answers are the result being measured; missing answers mean the run is incomplete.
    return 1 if len(answered) < len(verified) else 0


if __name__ == "__main__":
    sys.exit(main())
