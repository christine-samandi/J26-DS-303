"""Explanation (XAI) agent: the engine's steps -> text a non-expert can follow and verify.

1. `template_lines`: one line per CalcStep, built deterministically. Numbers come from the
   engine; citations are tags naming the rule_id, e.g. "[SEED-PERSONAL_RELIEF-YA2025_26]".
2. The LLM rewrites each line in plain English. It sees only those lines (figures and rule
   ids, no PII) and must not add, drop or change a number or a tag.
3. `post_check.check` rejects a rewrite that breaks this; the template lines are used instead.
4. Tags are rendered as the rule's human citation for display.

No LLM SDK is imported here: the caller passes any `TextLLM`, so tests use a fake.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from explanation_agent.models import Claim, Explanation, LlmCall, RewriteStatus
from explanation_agent.post_check import TAG, check
from tax_calculation_agent.engine import CalcStep

PROMPT_VERSION = "explain-v2"  # v2: "%" and no number words
PROMPT_TEMPLATE = """\
You explain a Sri Lankan income tax calculation to a taxpayer who is not a tax expert.
Below is one line per calculation step, as JSON. Rewrite each line in plain, friendly English,
in one or two short sentences.

Rules you must follow:
- Keep every number exactly as it is. Do not add, remove, round or recompute any number.
- Write numbers only in digits and percentages only with "%". Never use number words such as
  "one", "zero", "twenty", "million" or "percent", even in phrases like "no one".
- Keep every citation tag in square brackets, such as [RULE-ID], exactly as written and on
  the same line. Do not add any other square-bracket tags.
- Keep one line per step, with the same step_id, in the same order.

Steps:
{steps}

Answer with JSON only, in exactly this form:
{{"lines": [{{"step_id": "<step_id>", "text": "<plain-language text>"}}]}}
"""


class TextLLM(Protocol):
    model: str
    temperature: float

    def generate(self, prompt: str) -> tuple[str, str]:
        """Return (response text, model version that actually answered)."""
        ...


@dataclass(frozen=True)
class ExplainOutcome:
    explanation: Explanation
    llm_call: LlmCall | None  # None when no LLM was configured or the call itself failed
    warnings: tuple[str, ...] = ()


def template_lines(steps: Sequence[CalcStep], rules_used: Mapping[str, str]) -> list[str]:
    """One line per step, citations as rule_id tags. Raises if a step cites an unused rule."""
    lines = []
    for step in steps:
        unsupported = [ref for ref in step.citation_refs if ref not in rules_used]
        if unsupported:  # rule 3: only cite what C3 actually returned
            raise ValueError(f"{step.step_id} cites rules C3 did not return: {unsupported}")
        tags = "".join(f" [{ref}]" for ref in step.citation_refs)
        lines.append(f"{step.description}: Rs. {step.result:,.2f}{tags}")
    return lines


def build_prompt(steps: Sequence[CalcStep], lines: Sequence[str]) -> str:
    items = [{"step_id": s.step_id, "text": t} for s, t in zip(steps, lines, strict=True)]
    return PROMPT_TEMPLATE.format(steps=json.dumps(items, indent=2, ensure_ascii=False))


def explain(
    steps: Sequence[CalcStep], rules_used: Mapping[str, str], llm: TextLLM | None = None
) -> ExplainOutcome:
    """Explain `steps`. `rules_used` maps each rule_id the answer may cite to its citation.

    Without an LLM the template is the explanation. With one, its rewrite is used only if it
    passes the post-check. An LLM error falls back to the template without a confidence
    penalty (nothing wrong was produced); a rejected rewrite sets `used_fallback`.
    `rewrite_status` records which of these happened: not_requested, failed, rejected or
    accepted.
    """
    template = template_lines(steps, rules_used)
    if llm is None:
        return ExplainOutcome(_explanation(steps, template, rules_used), None)

    try:
        reply, model_version = llm.generate(build_prompt(steps, template))
    except Exception as exc:  # no answer at all; the type name carries no PII
        return ExplainOutcome(
            _explanation(steps, template, rules_used, "failed"),
            None,
            (f"explanation LLM call failed ({type(exc).__name__}); template text shown",),
        )

    call = LlmCall(
        stage="explain",
        model=model_version or llm.model,
        prompt_version=PROMPT_VERSION,
        temperature=llm.temperature,
        output=reply,
    )
    result = check(reply, steps, template, rules_used)
    if result.passed:
        return ExplainOutcome(_explanation(steps, result.lines, rules_used, "accepted"), call)
    explanation = _explanation(
        steps, template, rules_used, "rejected", hallucination_flag=bool(result.hallucinations)
    )
    warnings = tuple(f"explanation rewrite rejected: {f}" for f in result.failures)
    return ExplainOutcome(explanation, call, warnings)


def _explanation(
    steps: Sequence[CalcStep],
    lines: Sequence[str],
    rules_used: Mapping[str, str],
    rewrite_status: RewriteStatus = "not_requested",
    hallucination_flag: bool = False,
) -> Explanation:
    shown = [TAG.sub(lambda m: f"[{rules_used[m.group(1)]}]", line) for line in lines]
    claims = tuple(
        Claim(claim_id=f"CL{i}", step_id=s.step_id, text=text, citation_refs=s.citation_refs)
        for i, (s, text) in enumerate(zip(steps, shown, strict=True), start=1)
    )
    return Explanation(
        lines=tuple(shown),
        claims=claims,
        used_fallback=rewrite_status == "rejected",
        hallucination_flag=hallucination_flag,
        rewrite_status=rewrite_status,
    )
