"""Calculation accuracy of the deterministic engine on the golden scenarios.

    python eval/run_engine.py [--scenarios eval/scenarios] [--out eval/results]

Runs the engine on every GS-*.json, compares totals, steps and citations to the expected
values, and writes one results table as CSV + markdown, stamped with the git SHA.
Only `verified` scenarios count towards accuracy; unverified ones are reported but excluded.
Exits 1 if any verified scenario fails (target: 100%).

Needs the C1 package installed (pip install -e backend/component1-orchestrator).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from tax_calculation_agent.engine import calculate
from tax_calculation_agent.ruleset import RuleSet, load_seed_rules

EVAL_DIR = Path(__file__).resolve().parent
TOTALS = ("taxable_income", "total_tax", "tax_payable", "refund")
CENT = Decimal("0.01")
CSV_FIELDS = (
    "scenario_id",
    "title",
    "verified",
    "passed",
    "expected_total_tax",
    "actual_total_tax",
    "expected_tax_payable",
    "actual_tax_payable",
    "expected_refund",
    "actual_refund",
    "mismatches",
)


@dataclass
class Outcome:
    scenario_id: str
    title: str
    verified: bool
    expected: dict
    actual: dict = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def load_scenario(path: Path) -> dict:
    with path.open() as f:
        return json.load(f, parse_float=Decimal)


def rule_ids_by_key(rules: RuleSet) -> dict[str, str]:
    return {
        "personal_relief": rules.personal_relief.ref.rule_id,
        "individual_income_tax_slabs": rules.tax_bands.ref.rule_id,
    }


def evaluate(scenario: dict) -> Outcome:
    expected = scenario["expected"]
    outcome = Outcome(
        scenario_id=scenario["scenario_id"],
        title=scenario["title"],
        verified=scenario["verification"]["status"] == "verified",
        expected={k: Decimal(expected[k]) for k in TOTALS},
    )
    try:
        rules = load_seed_rules(scenario["year_of_assessment"])
        given = scenario["input"]
        result = calculate(
            Decimal(given["employment_income"]), Decimal(given["apit_paid"]), rules
        )
    except Exception as exc:  # a crash is a failed scenario, not a failed run
        outcome.mismatches.append(f"engine error: {type(exc).__name__}: {exc}")
        return outcome

    outcome.actual = {k: getattr(result, k) for k in TOTALS}
    for k in TOTALS:
        if outcome.actual[k] != outcome.expected[k]:
            outcome.mismatches.append(
                f"{k}: expected {outcome.expected[k]}, got {outcome.actual[k]}"
            )

    ids = rule_ids_by_key(rules)
    want = [
        (s["step_id"], Decimal(s["result"]), tuple(ids[c] for c in s["cites"]))
        for s in expected["steps"]
    ]
    got = [(s.step_id, s.result, s.citation_refs) for s in result.steps]
    if [w[0] for w in want] != [g[0] for g in got]:
        outcome.mismatches.append(
            f"steps: expected {[w[0] for w in want]}, got {[g[0] for g in got]}"
        )
    else:
        for (sid, w_result, w_cites), (_, g_result, g_cites) in zip(
            want, got, strict=True
        ):
            if w_result != g_result:
                outcome.mismatches.append(
                    f"{sid} result: expected {w_result}, got {g_result}"
                )
            if w_cites != g_cites:
                outcome.mismatches.append(
                    f"{sid} cites: expected {w_cites}, got {g_cites}"
                )
    return outcome


def git_sha() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=EVAL_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        return sha + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def money(value: Decimal | None) -> str:
    """Show expected and actual figures the same way, e.g. 96000 and 96000.00 -> 96000.00."""
    return "" if value is None else str(value.quantize(CENT))


def write_csv(path: Path, outcomes: list[Outcome]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for o in outcomes:
            writer.writerow(
                {
                    "scenario_id": o.scenario_id,
                    "title": o.title,
                    "verified": o.verified,
                    "passed": o.passed,
                    "expected_total_tax": money(o.expected["total_tax"]),
                    "actual_total_tax": money(o.actual.get("total_tax")),
                    "expected_tax_payable": money(o.expected["tax_payable"]),
                    "actual_tax_payable": money(o.actual.get("tax_payable")),
                    "expected_refund": money(o.expected["refund"]),
                    "actual_refund": money(o.actual.get("refund")),
                    "mismatches": "; ".join(o.mismatches),
                }
            )


def write_markdown(path: Path, outcomes: list[Outcome], sha: str, run_at: str) -> None:
    verified = [o for o in outcomes if o.verified]
    passed = sum(o.passed for o in verified)
    accuracy = f"{100 * passed / len(verified):.1f}%" if verified else "n/a"
    lines = [
        "# Engine calculation accuracy",
        "",
        f"- Run: {run_at}",
        f"- Git SHA: `{sha}`",
        "- System: deterministic engine (no LLM), seed rules",
        f"- **Accuracy (verified scenarios): {passed}/{len(verified)} = {accuracy}**",
        f"- Unverified scenarios (excluded): {len(outcomes) - len(verified)}",
        "",
        "| Scenario | Verified | Pass | Expected tax | Actual tax | Expected payable"
        " | Actual payable | Expected refund | Actual refund | Mismatches |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        cells = [
            o.scenario_id,
            "yes" if o.verified else "no",
            "✅" if o.passed else "❌",
            money(o.expected["total_tax"]),
            money(o.actual.get("total_tax")) or "—",
            money(o.expected["tax_payable"]),
            money(o.actual.get("tax_payable")) or "—",
            money(o.expected["refund"]),
            money(o.actual.get("refund")) or "—",
            "; ".join(o.mismatches).replace("|", "\\|") or "",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def run(scenario_dir: Path, out_dir: Path) -> tuple[list[Outcome], Path, Path]:
    paths = sorted(scenario_dir.glob("GS-*.json"))
    if not paths:
        raise SystemExit(f"no golden scenarios found in {scenario_dir}")
    outcomes = [evaluate(load_scenario(p)) for p in paths]

    now = datetime.now(UTC)
    sha = git_sha()
    stem = f"engine_{now:%Y%m%dT%H%M%SZ}_{sha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.md"
    write_csv(csv_path, outcomes)
    write_markdown(md_path, outcomes, sha, now.isoformat(timespec="seconds"))
    return outcomes, csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=Path, default=EVAL_DIR / "scenarios")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    args = parser.parse_args(argv)

    outcomes, csv_path, md_path = run(args.scenarios, args.out)
    verified = [o for o in outcomes if o.verified]
    failed = [o for o in verified if not o.passed]
    print(
        f"Engine accuracy: {len(verified) - len(failed)}/{len(verified)} verified scenarios"
    )
    for o in failed:
        print(f"  FAIL {o.scenario_id}: {'; '.join(o.mismatches)}")
    print(f"Wrote {csv_path}\n      {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
