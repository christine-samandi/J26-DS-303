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
CSV_FIELDS = (
    "scenario_id",
    "verified",
    "passed",
    "parsed",
    *(f"{side}_{k}" for k in TOTALS for side in ("expected", "llm")),
    "mismatches",
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
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            temperature=0, seed=0, response_mime_type="application/json"
        )

    def generate(self, prompt: str) -> tuple[str, str]:
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config
        )
        return response.text or "", response.model_version or self.model


@dataclass
class Outcome:
    scenario_id: str
    verified: bool
    expected: dict
    answer: dict = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    model_version: str = ""
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


def evaluate(scenario: dict, client: LLMClient) -> Outcome:
    outcome = Outcome(
        scenario_id=scenario["scenario_id"],
        verified=scenario["verification"]["status"] == "verified",
        expected={k: Decimal(scenario["expected"][k]) for k in TOTALS},
    )
    prompt = build_prompt(scenario, load_seed_rules(scenario["year_of_assessment"]))

    start = time.perf_counter()
    for _ in range(2):  # one retry on a malformed reply, as for the interpretation agent
        try:
            text, outcome.model_version = client.generate(prompt)
        except Exception as exc:  # an API failure is a failed scenario, not a failed run
            outcome.mismatches = [f"llm error: {type(exc).__name__}: {exc}"]
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
    passed = sum(o.passed for o in verified)
    accuracy = f"{100 * passed / len(verified):.1f}%" if verified else "n/a"
    versions = sorted({o.model_version for o in outcomes if o.model_version}) or ["—"]
    lines = [
        "# Pure-LLM baseline calculation accuracy",
        "",
        f"- Run: {run_at}",
        f"- Git SHA: `{sha}`",
        f"- Model requested: `{model}`; answered by: {', '.join(f'`{v}`' for v in versions)}",
        f"- Prompt: `{PROMPT_VERSION}`, temperature 0, seed 0, seed rules given in the prompt",
        f"- **Accuracy (verified scenarios): {passed}/{len(verified)} = {accuracy}**",
        f"- Unparseable replies: {sum(not o.parsed for o in verified)}",
        f"- Unverified scenarios (excluded): {len(outcomes) - len(verified)}",
        "",
        "| Scenario | Verified | Pass | Expected tax | LLM tax | Expected payable"
        " | LLM payable | Expected refund | LLM refund | Mismatches |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        cells = [
            o.scenario_id,
            "yes" if o.verified else "no",
            "✅" if o.passed else "❌",
            money(o.expected["total_tax"]),
            o.answer.get("total_tax", "—"),
            money(o.expected["tax_payable"]),
            o.answer.get("tax_payable", "—"),
            money(o.expected["refund"]),
            o.answer.get("refund", "—"),
            "; ".join(o.mismatches).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def run(scenario_dir: Path, out_dir: Path, client: LLMClient) -> tuple[list[Outcome], Path]:
    paths = sorted(scenario_dir.glob("GS-*.json"))
    if not paths:
        raise SystemExit(f"no golden scenarios found in {scenario_dir}")
    outcomes = [evaluate(load_scenario(p), client) for p in paths]

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
    args = parser.parse_args(argv)

    client = client or gemini_from_env()
    outcomes, md_path = run(args.scenarios, args.out, client)
    verified = [o for o in outcomes if o.verified]
    failed = [o for o in verified if not o.passed]
    print(f"LLM baseline ({client.model}): {len(verified) - len(failed)}/{len(verified)} correct")
    for o in failed:
        print(f"  WRONG {o.scenario_id}: {'; '.join(o.mismatches)}")
    print(f"Wrote {md_path} (+ .csv, .raw.jsonl)")
    return 0  # wrong LLM answers are a result to report, not a failed run


if __name__ == "__main__":
    sys.exit(main())
