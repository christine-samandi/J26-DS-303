"""What the explanation agent produces. Shaped after the `pipeline_trace` v0.1.0 proposal
(Claim, Explanation, LlmCall) so they can be swapped for `j26_contracts.trace` once agreed.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

LlmStage = Literal["interpret", "explain"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Claim(_Frozen):
    """One checkable statement in the explanation: which step it states and what cites it."""

    claim_id: str
    step_id: str
    text: str
    citation_refs: tuple[str, ...]


class Explanation(_Frozen):
    lines: tuple[str, ...]
    claims: tuple[Claim, ...]
    used_fallback: bool = False  # True when the LLM rewrite failed its post-check
    hallucination_flag: bool = False  # True when the rewrite added/changed a number or citation


class LlmCall(_Frozen):
    """One LLM call that shaped the answer. Prompts and outputs carry no PII."""

    stage: LlmStage
    model: str  # the model version that actually answered
    prompt_version: str
    temperature: Annotated[float, Field(ge=0, le=2)]
    output: str  # raw model output, kept even when it was rejected
