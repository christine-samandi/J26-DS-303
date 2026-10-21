"""How well the explanation post-check catches a lying LLM. No LLM calls, fully deterministic.

    python eval/run_explanation_checks.py [--scenarios eval/scenarios] [--out eval/results]

For every golden scenario, builds the correct template explanation from the engine's steps,
then tampered versions of it (seeded), and runs each through the real `post_check.check`.
Reports the catch rate per tamper type and overall, and the false-alarm rate on harmless
rewrites (paraphrases, and separately an honest mention of the correct year of assessment).
Writes one CSV row per variant and a markdown summary, stamped with the git SHA.
Exits 1 only if an untampered explanation fails the check (a broken harness); misses and
false alarms are the result being measured.

Needs the C1 package installed (pip install -e backend/component1-orchestrator).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from run_engine import EVAL_DIR, git_sha, load_scenario

from explanation_agent import post_check
from explanation_agent.explainer import template_lines
from explanation_agent.post_check import TAG, CheckResult, check
from tax_calculation_agent.engine import CalcStep, calculate
from tax_calculation_agent.ruleset import load_seed_rules

Lines = tuple[tuple[str, str], ...]  # (step_id, text) per step, as the LLM would return them


@dataclass(frozen=True)
class Explained:
    """One scenario's engine output and its correct, untampered explanation."""

    scenario_id: str
    year_of_assessment: str
    steps: tuple[CalcStep, ...]
    template: tuple[str, ...]
    rules_used: dict[str, str]  # rule_id -> human citation

    @property
    def lines(self) -> Lines:
        return tuple((s.step_id, t) for s, t in zip(self.steps, self.template, strict=True))


def explain_scenario(scenario: dict) -> Explained:
    rules = load_seed_rules(scenario["year_of_assessment"])
    given = scenario["input"]
    result = calculate(Decimal(given["employment_income"]), Decimal(given["apit_paid"]), rules)
    refs = (rules.personal_relief.ref, rules.tax_bands.ref)
    rules_used = {ref.rule_id: ref.citation for ref in refs}
    return Explained(
        scenario_id=scenario["scenario_id"],
        year_of_assessment=scenario["year_of_assessment"],
        steps=tuple(result.steps),
        template=tuple(template_lines(result.steps, rules_used)),
        rules_used=rules_used,
    )


def reply(lines: Sequence[tuple[str, str]]) -> str:
    """The lines in the JSON shape the explainer asks the LLM to answer in."""
    items = [{"step_id": sid, "text": text} for sid, text in lines]
    return json.dumps({"lines": items}, ensure_ascii=False)


def run_check(explained: Explained, lines: Sequence[tuple[str, str]]) -> CheckResult:
    return check(
        reply(lines),
        explained.steps,
        explained.template,
        explained.rules_used,
        explained.year_of_assessment,
    )


# --- tampering ---------------------------------------------------------------------------
# A generator lists every distinct way it can alter one scenario's explanation (each target
# step, sign, digit pair, ...). `variants` samples from that list with a seeded RNG, without
# replacement, so a run never counts the same tamper twice. An empty list means the tamper
# type does not apply to that scenario (e.g. a digit swap on 0.00), and it is not scored.

FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")  # a figure in digits, as the post-check reads them
OTHER_YEAR = {"2025/26": "2024/25", "2024/25": "2025/26"}


@dataclass(frozen=True)
class Candidate:
    step_id: str  # the step altered ("" when the whole explanation is)
    lines: Lines
    detail: str  # what changed, e.g. "96,000.00 -> 96,001.00"


@dataclass(frozen=True)
class Variant:
    scenario_id: str
    harmless: bool  # True: the text is still correct and must pass
    type: str
    subtype: str
    index: int
    candidate: Candidate | None  # None: not applicable to this scenario


Generator = Callable[[Explained], list[Candidate]]


def money_text(value: Decimal) -> str:
    return f"{value:,.2f}"


def _result_span(text: str, result: Decimal) -> tuple[int, int]:
    """Where the step's result is written: its last occurrence outside citation tags."""
    masked = TAG.sub(lambda m: " " * len(m.group()), text)
    spans = [
        m.span() for m in FIGURE.finditer(masked) if Decimal(m.group().replace(",", "")) == result
    ]
    return spans[-1]


def _with_result(text: str, result: Decimal, new: str) -> str:
    start, end = _result_span(text, result)
    return text[:start] + new + text[end:]


def _replace_line(lines: Lines, i: int, text: str) -> Lines:
    return (*lines[:i], (lines[i][0], text), *lines[i + 1 :])


def _split_tags(text: str) -> tuple[str, list[str]]:
    return TAG.sub("", text).rstrip(), TAG.findall(text)


def _join_tags(base: str, tags: Sequence[str]) -> str:
    return base + "".join(f" [{t}]" for t in tags)


def _insert_before_result(text: str, phrase: str) -> str:
    """Put `phrase` just before the ": Rs. <result>" every template line ends with."""
    idx = text.rfind(": Rs.")
    return text[:idx] + phrase + text[idx:]


def _per_step(
    e: Explained, change: Callable[[CalcStep, str], list[tuple[str, str]]]
) -> list[Candidate]:
    """One candidate per (step, new text) that `change` proposes and that differs."""
    out = []
    for i, (step, (_, text)) in enumerate(zip(e.steps, e.lines, strict=True)):
        for new_text, detail in change(step, text):
            if new_text != text:
                out.append(Candidate(step.step_id, _replace_line(e.lines, i, new_text), detail))
    return out


def _new_result(step: CalcStep, text: str, values: Sequence[Decimal], fmt=money_text):
    old = money_text(step.result)
    return [
        (_with_result(text, step.result, fmt(v)), f"{old} -> {fmt(v)}")
        for v in values
        if v >= 0 and v != step.result
    ]


def amount_small(e: Explained) -> list[Candidate]:
    return _per_step(e, lambda s, t: _new_result(s, t, [s.result + 1, s.result - 1]))


def amount_large(e: Explained) -> list[Candidate]:
    return _per_step(
        e, lambda s, t: _new_result(s, t, [s.result + 10_000, s.result - 10_000, s.result * 10])
    )


def amount_digit_swap(e: Explained) -> list[Candidate]:
    def swaps(step: CalcStep, text: str):
        whole, cents = money_text(step.result).replace(",", "").split(".")
        values = []
        for i in range(len(whole) - 1):
            if whole[i] != whole[i + 1] and not (i == 0 and whole[1] == "0"):
                swapped = whole[:i] + whole[i + 1] + whole[i] + whole[i + 2 :]
                values.append(Decimal(f"{swapped}.{cents}"))
        return _new_result(step, text, values)

    return _per_step(e, swaps)


def _label(input_name: str) -> str:
    """An input's name as words, without digits or number words ("band_1_tax" -> "band tax",
    "rate_percent" -> "rate"), so a role swap is never caught just for how it is labelled."""
    words = [w for w in input_name.split("_") if not w.isdigit() and w != "percent"]
    return " ".join(words)


def amount_role_swap(e: Explained) -> list[Candidate]:
    """Keep the result in the line but state another of the step's own figures as the result,
    labelling the real result as that input: every figure is allowed, only the roles change."""

    def swap(step: CalcStep, text: str):
        out = []
        for name, value in step.inputs.items():
            if value is None or value == step.result:
                continue
            label = f" ({_label(name)}: Rs. {money_text(step.result)})"
            new = _insert_before_result(_with_result(text, step.result, money_text(value)), label)
            out.append((new, f"result {money_text(step.result)} <-> {name} {money_text(value)}"))
        return out

    return _per_step(e, swap)


def rounding_cents(e: Explained) -> list[Candidate]:
    return _per_step(e, lambda s, t: _new_result(s, t, [s.result + Decimal("0.50")]))


def rounding_whole(e: Explained) -> list[Candidate]:
    """Cents rounded away, or a whole figure written one rupee short (96,000 -> 95,999)."""

    def whole(step: CalcStep, text: str):
        r = step.result
        new = r.quantize(Decimal(1), ROUND_HALF_UP) if r != r.to_integral_value() else r - 1
        return _new_result(step, text, [new], fmt=lambda v: f"{v:,.0f}")

    return _per_step(e, whole)


def rounding_one_figure(e: Explained) -> list[Candidate]:
    """Rounded to one significant figure, as in "about Rs. 100,000" for 96,000."""

    def approx(step: CalcStep, text: str):
        r = step.result
        if r < 10:
            return []
        unit = Decimal(10) ** (len(str(int(r))) - 1)
        new = (r / unit).quantize(Decimal(1), ROUND_HALF_UP) * unit
        return _new_result(step, text, [new], fmt=lambda v: f"{v:,.0f}")

    return _per_step(e, approx)


def _per_tag(e: Explained, replacements: Callable[[str], list[str]]) -> list[Candidate]:
    """One candidate per (cited step, tag, replacement tag); "" as replacement drops the tag."""
    out = []
    for i, (sid, text) in enumerate(e.lines):
        base, tags = _split_tags(text)
        for j, tag in enumerate(tags):
            for new in replacements(tag):
                if new in tags:  # already cited on this line: no visible change
                    continue
                new_tags = [*tags[:j], *([new] if new else []), *tags[j + 1 :]]
                detail = f"[{tag}] -> [{new}]" if new else f"dropped [{tag}]"
                out.append(
                    Candidate(sid, _replace_line(e.lines, i, _join_tags(base, new_tags)), detail)
                )
    return out


def citation_swap(e: Explained) -> list[Candidate]:
    """Two steps that cite different rules exchange their citations."""
    out = []
    split = [_split_tags(text) for _, text in e.lines]
    for i in range(len(split)):
        for j in range(i + 1, len(split)):
            (bi, ti), (bj, tj) = split[i], split[j]
            if ti and tj and set(ti) != set(tj):
                lines = _replace_line(e.lines, i, _join_tags(bi, tj))
                lines = _replace_line(lines, j, _join_tags(bj, ti))
                sids = f"{e.lines[i][0]} <-> {e.lines[j][0]}"
                out.append(Candidate(sids, lines, f"citations swapped between {sids}"))
    return out


def citation_wrong_rule(e: Explained) -> list[Candidate]:
    """A citation replaced by the other rule C3 returned: real, retrieved, but wrong here."""
    return _per_tag(e, lambda tag: [r for r in e.rules_used if r != tag])


def citation_invented(e: Explained) -> list[Candidate]:
    """A citation replaced by a plausible-looking rule id C3 never returned."""
    ya = e.year_of_assessment.replace("/", "_")
    return _per_tag(e, lambda tag: [f"SEED-QUALIFYING_PAYMENTS-YA{ya}"])


def drop_step(e: Explained) -> list[Candidate]:
    return [
        Candidate(sid, (*e.lines[:i], *e.lines[i + 1 :]), f"dropped line {sid}")
        for i, (sid, _) in enumerate(e.lines)
    ]


def drop_citation(e: Explained) -> list[Candidate]:
    return _per_tag(e, lambda tag: [""])


def _year_mention(e: Explained, ya: str) -> list[Candidate]:
    return _per_step(
        e, lambda s, t: [(_insert_before_result(t, f" for YA {ya}"), f"adds 'for YA {ya}'")]
    )


def wrong_year_text(e: Explained) -> list[Candidate]:
    return _year_mention(e, OTHER_YEAR[e.year_of_assessment])


def wrong_year_cited_rule(e: Explained) -> list[Candidate]:
    """A citation replaced by the same rule for the other year of assessment."""
    this, other = (
        load_seed_rules(e.year_of_assessment),
        load_seed_rules(OTHER_YEAR[e.year_of_assessment]),
    )
    same_rule = {
        this.personal_relief.ref.rule_id: other.personal_relief.ref.rule_id,
        this.tax_bands.ref.rule_id: other.tax_bands.ref.rule_id,
    }
    return _per_tag(e, lambda tag: [same_rule[tag]])


def correct_year_mention(e: Explained) -> list[Candidate]:
    """Harmless: an honest "for YA 2025/26" naming the scenario's own year."""
    return _year_mention(e, e.year_of_assessment)


PARAPHRASES = {  # every line reworded; the description, figure and tags are kept as they are
    "which_comes_to": "{desc}, which comes to {rs}",
    "in_short": "In short, {desc_mid}: {rs}",
    "works_out_to": "{desc}. That works out to {rs}",
}


def paraphrase_reword(e: Explained) -> list[Candidate]:
    out = []
    for name, pattern in PARAPHRASES.items():
        lines = []
        for sid, text in e.lines:
            base, tags = _split_tags(text)
            idx = base.rfind(": Rs.")
            desc = base[:idx]
            # mid-sentence: "personal relief", but "APIT" stays as it is
            desc_mid = desc if desc[:2].isupper() else desc[0].lower() + desc[1:]
            reworded = pattern.format(desc=desc, desc_mid=desc_mid, rs=base[idx + 2 :])
            lines.append((sid, _join_tags(reworded, tags)))
        out.append(Candidate("", tuple(lines), f"pattern {name}"))
    return out


def paraphrase_format(e: Explained) -> list[Candidate]:
    """Same figures written differently, or citations moved to the start of the line."""
    formats = {
        "no_decimals": lambda t: re.sub(r"(\d)\.00\b", r"\1", t),
        "no_commas": lambda t: re.sub(r"(?<=\d),(?=\d{3})", "", t),
        "tags_first": lambda t: " ".join(
            [*(f"[{x}]" for x in _split_tags(t)[1]), _split_tags(t)[0]]
        ),
    }
    out = []
    for name, fmt in formats.items():
        lines = tuple((sid, fmt(text)) for sid, text in e.lines)
        if lines != e.lines:
            out.append(Candidate("", lines, f"format {name}"))
    return out


# (harmless, type, subtype, generator), in report order
GENERATORS: tuple[tuple[bool, str, str, Generator], ...] = (
    (False, "amount", "small", amount_small),
    (False, "amount", "large", amount_large),
    (False, "amount", "digit_swap", amount_digit_swap),
    (False, "amount", "role_swap", amount_role_swap),
    (False, "rounding", "cents", rounding_cents),
    (False, "rounding", "whole", rounding_whole),
    (False, "rounding", "one_figure", rounding_one_figure),
    (False, "citation", "swap", citation_swap),
    (False, "citation", "wrong_rule", citation_wrong_rule),
    (False, "citation", "invented", citation_invented),
    (False, "drop", "step", drop_step),
    (False, "drop", "citation", drop_citation),
    (False, "wrong_year", "text", wrong_year_text),
    (False, "wrong_year", "cited_rule", wrong_year_cited_rule),
    (True, "paraphrase", "reword", paraphrase_reword),
    (True, "paraphrase", "format", paraphrase_format),
    (True, "correct_year_mention", "text", correct_year_mention),
)


def variants(e: Explained, seed: int, per_type: int) -> list[Variant]:
    """Up to `per_type` variants per generator, sampled with an RNG seeded per scenario and
    generator, so adding a generator or scenario leaves every other sample unchanged."""
    out = []
    for harmless, type_, subtype, generate in GENERATORS:
        candidates = [c for c in generate(e) if c.lines != e.lines]
        if not candidates:
            out.append(Variant(e.scenario_id, harmless, type_, subtype, 0, None))
            continue
        rng = random.Random(f"{seed}:{e.scenario_id}:{type_}/{subtype}")
        picked = rng.sample(candidates, min(per_type, len(candidates)))
        out += [
            Variant(e.scenario_id, harmless, type_, subtype, i, c) for i, c in enumerate(picked)
        ]
    return out


def load_explained(scenario_dir: Path) -> list[Explained]:
    """Every scenario's correct explanation. Raises if one fails the post-check, since then
    the harness, not a tamper, would be what the check rejects."""
    paths = sorted(scenario_dir.glob("GS-*.json"))
    if not paths:
        raise SystemExit(f"no golden scenarios found in {scenario_dir}")
    explained = [explain_scenario(load_scenario(p)) for p in paths]
    for e in explained:
        result = run_check(e, e.lines)
        if not result.passed:
            raise SystemExit(
                f"{e.scenario_id}: untampered template fails the post-check: "
                + "; ".join(result.failures)
            )
    return explained


# --- scoring -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    variant: Variant
    changed: str  # the lines that differ from the template, "- old | + new"
    flagged: bool = False  # the post-check rejected the text
    reason: str = ""  # "structural" or "hallucination" when flagged
    failures: tuple[str, ...] = ()

    @property
    def outcome(self) -> str:
        if self.variant.candidate is None:
            return "not_applicable"
        if self.variant.harmless:
            return "false_alarm" if self.flagged else "passed"
        return "caught" if self.flagged else "missed"


def changed_lines(original: Lines, tampered: Lines) -> str:
    removed = [f"- {sid}: {t}" for sid, t in original if (sid, t) not in tampered]
    added = [f"+ {sid}: {t}" for sid, t in tampered if (sid, t) not in original]
    return " | ".join([*removed, *added])


def score(e: Explained, v: Variant) -> Scored:
    if v.candidate is None:
        return Scored(v, "")
    result = run_check(e, v.candidate.lines)
    reason = "structural" if result.structural else "hallucination" if result.hallucinations else ""
    return Scored(
        v, changed_lines(e.lines, v.candidate.lines), not result.passed, reason, result.failures
    )


def score_all(explained: Sequence[Explained], seed: int, per_type: int) -> list[Scored]:
    return [score(e, v) for e in explained for v in variants(e, seed, per_type)]


@dataclass(frozen=True)
class Rate:
    label: str
    hits: int  # caught (tampers) or false alarms (harmless)
    total: int

    @property
    def text(self) -> str:
        return (
            f"{self.hits}/{self.total} = {100 * self.hits / self.total:.1f}%"
            if self.total
            else "n/a"
        )


def rate(label: str, scored: Sequence[Scored]) -> Rate:
    applicable = [s for s in scored if s.variant.candidate is not None]
    return Rate(label, sum(s.flagged for s in applicable), len(applicable))


def summary(scored: Sequence[Scored]) -> dict[str, Rate]:
    """The headline rates. For tampers a hit is a catch; for harmless text, a false alarm."""
    tampers = [s for s in scored if not s.variant.harmless]
    by_type = lambda t: [s for s in scored if s.variant.type == t]  # noqa: E731
    return {
        "catch_rate": rate("Overall catch rate (all tamper types)", tampers),
        "paraphrase_false_alarm": rate("False-alarm rate: paraphrase", by_type("paraphrase")),
        "year_false_alarm": rate(
            "False-alarm rate: correct_year_mention", by_type("correct_year_mention")
        ),
    }


def breakdown(scored: Sequence[Scored]) -> list[tuple[bool, str, str, Rate, int]]:
    """(harmless, type, subtype, rate, not-applicable count) per generator, in report order."""
    rows = []
    for harmless, type_, subtype, _ in GENERATORS:
        group = [s for s in scored if (s.variant.type, s.variant.subtype) == (type_, subtype)]
        na = sum(s.variant.candidate is None for s in group)
        rows.append((harmless, type_, subtype, rate(f"{type_}/{subtype}", group), na))
    return rows


# --- output ------------------------------------------------------------------------------

CSV_FIELDS = (
    "scenario_id",
    "harmless",
    "type",
    "subtype",
    "variant",
    "outcome",
    "step_id",
    "detail",
    "reason",
    "failures",
    "changed_lines",
)
POST_CHECK = Path(post_check.__file__)


def post_check_hash() -> str:
    return hashlib.sha256(POST_CHECK.read_bytes()).hexdigest()[:12]


def write_csv(path: Path, scored: Sequence[Scored]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in scored:
            v, c = s.variant, s.variant.candidate
            writer.writerow(
                {
                    "scenario_id": v.scenario_id,
                    "harmless": v.harmless,
                    "type": v.type,
                    "subtype": v.subtype,
                    "variant": v.index,
                    "outcome": s.outcome,
                    "step_id": c.step_id if c else "",
                    "detail": c.detail if c else "",
                    "reason": s.reason,
                    "failures": "; ".join(s.failures),
                    "changed_lines": s.changed,
                }
            )


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def write_markdown(
    path: Path, scored: Sequence[Scored], sha: str, run_at: str, seed: int, per_type: int
) -> None:
    head = summary(scored)
    scenarios = len({s.variant.scenario_id for s in scored})
    lines = [
        "# Explanation post-check: tamper detection",
        "",
        f"- Run: {run_at}",
        f"- Git SHA: `{sha}`",
        f"- Post-check: `explanation_agent/post_check.py` (sha256 `{post_check_hash()}`)",
        f"- Scenarios: {scenarios}; seed {seed}; up to {per_type} variants per type and scenario",
        "- System: engine template explanations, tampered deterministically; no LLM calls",
        f"- **{head['catch_rate'].label}: {head['catch_rate'].text}**",
        f"- **{head['paraphrase_false_alarm'].label}: {head['paraphrase_false_alarm'].text}**",
        f"- **{head['year_false_alarm'].label}: {head['year_false_alarm'].text}**"
        ' (an honest "for YA <the scenario\'s own year>"; reported apart from paraphrases)',
        "",
        "## Tampered explanations (must be rejected)",
        "",
        "| Type | Subtype | Caught | Not applicable |",
        "|---|---|---|---|",
    ]
    rows = breakdown(scored)
    lines += [f"| {t} | {st} | {r.text} | {na} |" for h, t, st, r, na in rows if not h]
    lines += [
        "",
        "## Harmless explanations (must pass)",
        "",
        "| Type | Subtype | False alarms | Not applicable |",
        "|---|---|---|---|",
    ]
    lines += [f"| {t} | {st} | {r.text} | {na} |" for h, t, st, r, na in rows if h]

    misses = [s for s in scored if s.outcome == "missed"]
    lines += ["", f"## Misses ({len(misses)})", ""]
    if misses:
        lines += ["| Scenario | Tamper | Step | Change | Lines |", "|---|---|---|---|---|"]
        lines += [
            f"| {s.variant.scenario_id} | {s.variant.type}/{s.variant.subtype} "
            f"| {s.variant.candidate.step_id} | {_cell(s.variant.candidate.detail)} "
            f"| {_cell(s.changed)} |"
            for s in misses
        ]
    else:
        lines.append("None.")

    alarms = [s for s in scored if s.outcome == "false_alarm"]
    lines += ["", f"## False alarms ({len(alarms)})", ""]
    if alarms:
        lines += ["| Scenario | Harmless change | Change | Post-check said |", "|---|---|---|---|"]
        lines += [
            f"| {s.variant.scenario_id} | {s.variant.type}/{s.variant.subtype} "
            f"| {_cell(s.variant.candidate.detail)} | {_cell('; '.join(s.failures))} |"
            for s in alarms
        ]
    else:
        lines.append("None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    scenario_dir: Path, out_dir: Path, seed: int, per_type: int
) -> tuple[list[Scored], Path, Path]:
    scored = score_all(load_explained(scenario_dir), seed, per_type)
    now = datetime.now(UTC)
    sha = git_sha()
    stem = f"explanation_checks_{now:%Y%m%dT%H%M%SZ}_{sha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.md"
    write_csv(csv_path, scored)
    write_markdown(md_path, scored, sha, now.isoformat(timespec="seconds"), seed, per_type)
    return scored, csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=Path, default=EVAL_DIR / "scenarios")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=3, help="per tamper type and scenario")
    args = parser.parse_args(argv)

    scored, csv_path, md_path = run(args.scenarios, args.out, args.seed, args.variants)
    for r in summary(scored).values():
        print(f"{r.label}: {r.text}")
    for s in scored:
        if s.outcome in ("missed", "false_alarm"):
            c = s.variant.candidate
            print(
                f"  {s.outcome.upper()} {s.variant.scenario_id} {s.variant.type}/"
                f"{s.variant.subtype} {c.step_id}: {c.detail}"
            )
    print(f"Wrote {csv_path}\n      {md_path}")
    return 0  # misses and false alarms are the measurement; a broken harness exits earlier


if __name__ == "__main__":
    sys.exit(main())
