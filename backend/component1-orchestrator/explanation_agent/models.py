"""What the explanation agent produces. Shaped after the `pipeline_trace` v0.1.0 proposal
(Claim, Explanation, LlmCall) so they can be swapped for `j26_contracts.trace` once agreed.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LlmStage = Literal["interpret", "explain"]
RewriteStatus = Literal["not_requested", "failed", "rejected", "accepted"]


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
    # What happened to the LLM rewrite. The template is shown unless it is "accepted";
    # "failed" means the call itself raised (nothing was produced, nothing was rejected).
    rewrite_status: RewriteStatus = "not_requested"

    @model_validator(mode="before")
    @classmethod
    def _status_of_older_data(cls, data: Any) -> Any:
        """Data from before rewrite_status existed: a fallback can only have been a rejection."""
        if isinstance(data, dict) and "rewrite_status" not in data and data.get("used_fallback"):
            return {**data, "rewrite_status": "rejected"}
        return data

    @model_validator(mode="after")
    def _flags_match_status(self) -> Explanation:
        if self.used_fallback != (self.rewrite_status == "rejected"):
            raise ValueError("used_fallback must be true exactly when rewrite_status is 'rejected'")
        if self.hallucination_flag and self.rewrite_status != "rejected":
            raise ValueError("hallucination_flag needs rewrite_status 'rejected'")
        return self


class LlmCall(_Frozen):
    """One LLM call that shaped the answer. Prompts and outputs carry no PII."""

    stage: LlmStage
    model: str  # the model version that actually answered
    prompt_version: str
    temperature: Annotated[float, Field(ge=0, le=2)]
    output: str  # raw model output, kept even when it was rejected
