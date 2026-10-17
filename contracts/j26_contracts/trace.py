"""Pipeline trace contract: C1 (orchestrator)  ->  C4 (output assurance).

Owner: C1 (Fernando W I C S). STATUS: PROPOSAL v0.1.0, not yet agreed. Changes need approval
from all four members (see CONTRACT_CHANGELOG.md).

One trace per attempt of one request. It holds everything C4 needs to check C1's answer
without calling C1 back:

    inputs      every amount the calculation used, where it came from, and how sure C2 was
    rules_used  every rule cited, by C3 rule_id, so C4 can re-fetch it from C3
    steps       every calculation step: C4 can recompute `result` from `inputs` + `operation`
    claims      every statement in the explanation, linked to a step and its citations
    confidence  C1's confidence, with each term of the formula

A trace contains NO personal data: no names, NIC, TIN, addresses or account numbers. Only
amounts, rule ids, and text generated from the steps.

Money and step values are exact decimals in Python (`Decimal`) and plain JSON numbers on the
wire (96000, not "96000.00"). Money has at most 2 decimal places.

Validate what you receive:
    from j26_contracts.trace import PipelineTrace
    trace = PipelineTrace.model_validate(request_json)
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    Field,
    PlainSerializer,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)

from j26_contracts.retrieval import (
    ContractModel,
    NonEmptyStr,
    Probability,
    RuleId,
    RuleKey,
    TaxpayerCategory,
)

SCHEMA_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

InputName = Literal["employment_income", "apit_paid"]  # v1 scope; grows with income types
InputSource = Literal["user", "ocr"]
StepSource = Literal["engine", "ocr", "llm_interpretation"]
LlmStage = Literal["interpret", "explain"]
RewriteStatus = Literal["not_requested", "failed", "rejected", "accepted"]
RerunStage = Literal["retrieve", "calculate", "explain"]

# ---------------------------------------------------------------------------
# Reusable field types
# ---------------------------------------------------------------------------

TraceRequestId = Annotated[str, StringConstraints(pattern=r"^req-[A-Za-z0-9-]+$")]
YearOfAssessment = Annotated[str, StringConstraints(pattern=r"^\d{4}/\d{2}$")]
StepId = Annotated[str, StringConstraints(pattern=r"^S[0-9]+_[A-Z0-9_]+$")]
ClaimId = Annotated[str, StringConstraints(pattern=r"^CL[0-9]+$")]
Attempt = Annotated[int, Field(ge=1, le=2)]


def _json_number(value: Decimal) -> int | float:
    """Whole amounts stay whole (96000); 2-decimal amounts round-trip exactly as floats."""
    return int(value) if value == value.to_integral_value() else float(value)


Money = Annotated[
    Decimal,
    Field(ge=0, decimal_places=2),
    PlainSerializer(_json_number, when_used="json"),
    WithJsonSchema({"type": "number", "minimum": 0, "description": "LKR, at most 2 decimals"}),
]
StepValue = Annotated[
    Decimal | None,
    PlainSerializer(lambda v: None if v is None else _json_number(v), when_used="json"),
    WithJsonSchema({"type": ["number", "null"]}),
]

# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


class InputValue(ContractModel):
    """One amount the calculation used."""

    name: InputName
    amount: Money
    source: InputSource
    document_id: str | None = Field(
        default=None, description="C2 document the amount was read from (source = ocr)."
    )
    field_confidence: Probability = Field(description="C2's confidence; 1 for typed input.")

    @model_validator(mode="after")
    def _ocr_names_its_document(self) -> InputValue:
        if (self.source == "ocr") != (self.document_id is not None):
            raise ValueError("document_id is required for ocr input and only for ocr input")
        return self


class RuleUsed(ContractModel):
    """A rule from C3 that a step cites. C4 can re-fetch it with RetrievalRequest.rule_ids."""

    rule_id: RuleId
    rule_key: RuleKey | None = None
    citation: NonEmptyStr = Field(description="source.citation as C3 returned it.")
    retrieval_confidence: Probability = Field(description="C3's confidence.score.")
    seed_agrees: bool | None = Field(
        default=None,
        description="Whether C1's seed rules agree with C3's values. null = no seed to compare.",
    )


class CalcStep(ContractModel):
    """One step of the deterministic calculation. `result` = `operation` applied to `inputs`."""

    step_id: StepId
    description: NonEmptyStr
    inputs: dict[str, StepValue]
    operation: NonEmptyStr = Field(
        description="Expression over the input names, e.g. 'max(0, employment_income - "
        "personal_relief)'. Rounding: round_half_up(x, 0.01)."
    )
    result: Money
    citation_refs: list[RuleId] = Field(description="rule_ids in rules_used. May be empty.")
    source: StepSource
    confidence: Probability


class TaxResult(ContractModel):
    taxable_income: Money
    total_tax: Money
    tax_payable: Money = Field(description="0 when there is a refund.")
    refund: Money = Field(description="0 when tax is payable.")

    @model_validator(mode="after")
    def _payable_or_refund(self) -> TaxResult:
        if self.tax_payable > 0 and self.refund > 0:
            raise ValueError("tax_payable and refund cannot both be above 0")
        return self


class Claim(ContractModel):
    """One checkable statement in the explanation."""

    claim_id: ClaimId
    step_id: StepId
    text: NonEmptyStr
    citation_refs: list[RuleId]


class Explanation(ContractModel):
    text: NonEmptyStr = Field(description="What the user is shown.")
    used_fallback: bool = Field(
        description="True if the LLM rewrite failed C1's post-check and the deterministic "
        "skeleton text was used instead."
    )
    hallucination_flag: bool = Field(
        description="True if the LLM rewrite added or changed a number or citation."
    )
    rewrite_status: RewriteStatus = Field(
        description="What happened to the LLM rewrite: not_requested (no LLM configured), "
        "failed (the LLM call raised; skeleton text shown, nothing rejected), rejected "
        "(failed C1's post-check; used_fallback is true), accepted (passed; rewrite shown)."
    )

    @model_validator(mode="after")
    def _flags_match_status(self) -> Explanation:
        if self.used_fallback != (self.rewrite_status == "rejected"):
            raise ValueError("used_fallback must be true exactly when rewrite_status is 'rejected'")
        if self.hallucination_flag and self.rewrite_status != "rejected":
            raise ValueError("hallucination_flag needs rewrite_status 'rejected'")
        return self


class LlmCall(ContractModel):
    """Every LLM call that shaped this answer. Prompts and outputs are PII-redacted."""

    stage: LlmStage
    model: NonEmptyStr = Field(description="Model version that actually answered.")
    prompt_version: NonEmptyStr
    temperature: Annotated[float, Field(ge=0, le=2)]
    output: str = Field(description="Raw (redacted) model output.")


class ConfidenceTerms(ContractModel):
    """score = min(ocr, retrieval, interpretation, explanation) x conflict_penalty."""

    method: Literal["c1-v1"]
    score: Probability
    ocr: Probability
    retrieval: Probability
    interpretation: Probability
    explanation: Probability
    conflict_penalty: Probability


class Regeneration(ContractModel):
    """Present on attempt 2: what C4 flagged on attempt 1 and where C1 restarted."""

    previous_attempt: Literal[1]
    rerun_from: RerunStage
    failed_step_ids: list[StepId] = Field(description="Step ids from attempt 1's trace.")


# ---------------------------------------------------------------------------
# Trace  (C1  →  C4)
# ---------------------------------------------------------------------------


class PipelineTrace(ContractModel):
    """Sent by C1 to C4 once per attempt. C4 answers with an assurance_verdict."""

    schema_version: Literal["0.1.0"]
    request_id: TraceRequestId = Field(description="Same request_id as every JSON of the request.")
    attempt: Attempt = Field(description="1, or 2 after one REGENERATE. Never more.")
    generated_at: AwareDatetime
    year_of_assessment: YearOfAssessment
    taxpayer_category: TaxpayerCategory
    inputs: Annotated[list[InputValue], Field(min_length=1)]
    rules_used: Annotated[list[RuleUsed], Field(min_length=1)]
    steps: Annotated[list[CalcStep], Field(min_length=1)]
    result: TaxResult
    claims: Annotated[list[Claim], Field(min_length=1)]
    explanation: Explanation
    llm_calls: list[LlmCall] = Field(description="Empty when no LLM was used.")
    confidence: ConfidenceTerms
    conflicts: list[str] = Field(description="Where C1's seed rules and C3 disagree.")
    warnings: list[str]
    regeneration: Regeneration | None = Field(
        default=None, description="Required on attempt 2, absent on attempt 1."
    )

    @model_validator(mode="after")
    def _internally_consistent(self) -> PipelineTrace:
        # [extra check] attempt and regeneration go together
        if (self.attempt == 2) != (self.regeneration is not None):
            raise ValueError("regeneration is required on attempt 2 and only on attempt 2")
        # [extra check] ids are unique
        for kind, ids in (
            ("input name", [i.name for i in self.inputs]),
            ("rule_id", [r.rule_id for r in self.rules_used]),
            ("step_id", [s.step_id for s in self.steps]),
            ("claim_id", [c.claim_id for c in self.claims]),
        ):
            dupes = [x for x, n in Counter(ids).items() if n > 1]
            if dupes:
                raise ValueError(f"duplicate {kind}: {dupes}")
        # [extra check] every citation points into rules_used (no invented sections)
        rule_ids = {r.rule_id for r in self.rules_used}
        cited = {ref for x in (*self.steps, *self.claims) for ref in x.citation_refs}
        if cited - rule_ids:
            raise ValueError(f"citations not in rules_used: {sorted(cited - rule_ids)}")
        # [extra check] claims point at real steps, and every step is explained
        step_ids = [s.step_id for s in self.steps]
        claimed = {c.step_id for c in self.claims}
        if claimed - set(step_ids):
            raise ValueError(f"claims about unknown steps: {sorted(claimed - set(step_ids))}")
        if set(step_ids) - claimed:
            raise ValueError(f"steps with no claim: {sorted(set(step_ids) - claimed)}")
        return self
