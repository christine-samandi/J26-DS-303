"""Deterministic check of the LLM's rewrite against the engine's steps (CLAUDE.md section 7).

A rewrite passes only if it has one line per step, same step_ids, same order, and each line:
  - states the step's result,
  - contains no number except the step's own figures (result, inputs, numbers in the
    template line), compared by value, so "1,200,000" and "1200000.00" are the same,
  - writes no number in words: zero to twenty, the tens, hundred / thousand / lakh /
    million / billion / crore, and "percent" / "per cent" (the template writes "%"). A word
    is allowed only where the step's template line already uses it ("not below zero"),
  - carries exactly the step's citation tags, each naming a rule in `rules_used`.

Failures are split in two: a changed/added number or citation is a hallucination; a
malformed reply (bad JSON, missing or reordered steps) is only a structural failure.
The word check is deliberately strict: an innocent "one" ("no one") is also rejected, which
costs a fallback to the template, never a wrong figure. Not detected: ordinals ("first"),
fractions ("half") and multipliers ("twice").
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from decimal import Decimal

from tax_calculation_agent.engine import CalcStep

TAG = re.compile(r"\[([^\[\]]+)\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = "hundred thousand lakh million billion crore".split()
_NUMBER_WORD = re.compile(
    r"\b(?:" + "|".join([*_UNITS, *_TENS, *(f"{w}s?" for w in _SCALES)]) + r"|per\s*cent)\b",
    re.I,
)


@dataclass(frozen=True)
class CheckResult:
    lines: tuple[str, ...] = ()  # the rewritten lines, still with rule_id tags
    structural: tuple[str, ...] = ()
    hallucinations: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (self.structural or self.hallucinations)

    @property
    def failures(self) -> tuple[str, ...]:
        return (*self.structural, *self.hallucinations)


def numbers_in(text: str) -> list[Decimal]:
    """Every number written in digits, citation tags excluded (rule ids contain digits)."""
    return [Decimal(m.rstrip(",").replace(",", "")) for m in _NUMBER.findall(TAG.sub(" ", text))]


def number_words_in(text: str) -> list[str]:
    """Number words, lower-cased, with "per cent" normalised to "percent"."""
    return [re.sub(r"\s+", "", w.lower()) for w in _NUMBER_WORD.findall(text)]


def check(
    reply: str,
    steps: Sequence[CalcStep],
    template_lines: Sequence[str],
    rules_used: Collection[str],
) -> CheckResult:
    lines = _parse(reply)
    if isinstance(lines, str):
        return CheckResult(structural=(lines,))
    expected_ids = [s.step_id for s in steps]
    got_ids = [step_id for step_id, _ in lines]
    if got_ids != expected_ids:
        return CheckResult(structural=(f"steps {got_ids} do not match {expected_ids}",))

    problems: list[str] = []
    for step, template, (_, text) in zip(steps, template_lines, lines, strict=True):
        problems += _check_line(step, template, text, rules_used)
    return CheckResult(lines=tuple(text for _, text in lines), hallucinations=tuple(problems))


def _check_line(step: CalcStep, template: str, text: str, rules_used: Collection[str]) -> list[str]:
    sid = step.step_id
    problems = []
    allowed = {step.result, *(v for v in step.inputs.values() if v is not None)}
    allowed.update(numbers_in(template))
    found = numbers_in(text)
    if step.result not in found:
        problems.append(f"{sid}: the result {step.result} is missing")
    for n in dict.fromkeys(found):
        if n not in allowed:
            problems.append(f"{sid}: {n} is not one of the engine's figures for this step")
    in_template = set(number_words_in(template))
    for word in dict.fromkeys(number_words_in(text)):
        if word not in in_template:
            problems.append(f"{sid}: number written in words ({word!r})")

    tags = TAG.findall(text)
    for tag in dict.fromkeys(tags):
        if tag not in rules_used:
            problems.append(f"{sid}: cites {tag!r}, which is not in rules_used")
    if sorted(set(tags)) != sorted(set(step.citation_refs)):
        problems.append(f"{sid}: citations {sorted(set(tags))} != {sorted(step.citation_refs)}")
    return problems


def _parse(reply: str) -> list[tuple[str, str]] | str:
    """[(step_id, text)], or a description of why the reply is not the JSON we asked for."""
    cleaned = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        items = json.loads(cleaned)["lines"]
        lines = [(item["step_id"], item["text"]) for item in items]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return f"reply is not the requested JSON ({type(exc).__name__})"
    if not all(isinstance(s, str) and isinstance(t, str) and t.strip() for s, t in lines):
        return "every line needs a string step_id and a non-empty text"
    return lines
