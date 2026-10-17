"""Demo only: a stand-in LLM whose rewrite is deliberately wrong, so the post-check can be
shown rejecting it without a network connection or a Gemini key.

`TamperedRewriteLLM` answers the real explanation prompt in the real reply format, but with
two lies, so it goes through the same `explain()` -> `post_check.check` path as Gemini:
  - the last step's figure (tax payable or refund) is raised by Rs. 10,000,
  - the citation tag of the last cited step is moved onto the last line.
Nothing downstream knows it is a demo; the trace records the model as `MODEL`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

from explanation_agent.post_check import TAG

MODEL = "simulated-tamper (demo)"
BUMP = Decimal(10_000)
_FIGURE = re.compile(r"Rs\. (\d[\d,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class Tampering:
    lines: tuple[tuple[str, str], ...]  # (step_id, text), tampered
    figure_step: str
    old_figure: str
    new_figure: str
    citation: str
    citation_from: str
    citation_to: str


def tamper_lines(lines: list[tuple[str, str]]) -> Tampering:
    """Change one figure and move one citation. Raises ValueError if either is impossible,
    so demo mode can never hand the post-check an untouched (passing) rewrite."""
    if not lines:
        raise ValueError("nothing to tamper with: no lines")
    out = [list(line) for line in lines]
    last = out[-1]

    figures = list(_FIGURE.finditer(TAG.sub(lambda m: " " * len(m.group()), last[1])))
    if not figures:
        raise ValueError(f"nothing to tamper with: no Rs. figure in {last[0]}")
    fig = figures[-1]
    old = fig.group(1)
    new = f"{Decimal(old.replace(',', '')) + BUMP:,.2f}"
    last[1] = last[1][: fig.start(1)] + new + last[1][fig.end(1) :]

    cited = [line for line in out if TAG.search(line[1])]
    if not cited:
        raise ValueError("nothing to tamper with: no citation tag in any line")
    source = cited[-1]
    tag = TAG.search(source[1])
    source[1] = (source[1][: tag.start()] + source[1][tag.end() :]).replace("  ", " ").strip()
    if source is not last:
        last[1] = f"{last[1]} {tag.group()}"

    return Tampering(
        lines=tuple((sid, text) for sid, text in out),
        figure_step=last[0],
        old_figure=old,
        new_figure=new,
        citation=tag.group(1),
        citation_from=source[0],
        citation_to=last[0],
    )


class TamperedRewriteLLM:
    """A `TextLLM` that never calls a provider; its reply always fails the post-check."""

    model = MODEL
    temperature = 0.0

    def __init__(self) -> None:
        self.last: Tampering | None = None  # what the latest reply changed, for display

    def generate(self, prompt: str) -> tuple[str, str]:
        steps = json.loads(prompt.split("Steps:\n", 1)[1].split("\n\nAnswer with JSON", 1)[0])
        self.last = tamper_lines([(s["step_id"], s["text"]) for s in steps])
        reply = [{"step_id": sid, "text": text} for sid, text in self.last.lines]
        return json.dumps({"lines": reply}, ensure_ascii=False), MODEL
