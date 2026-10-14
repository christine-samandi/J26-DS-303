"""Data the C1 orchestrator passes between stages.

Contracts that exist in `contracts/` are imported from `j26_contracts` (today: retrieval).
Everything marked PROVISIONAL stands in for a contract that has not been agreed yet:

    OcrField, OcrResult              -> C2's `ocr_output`        (owner C2)
    AssuranceVerdict, FailedClaim    -> C4's `assurance_verdict` (owner C4)
    PipelineTrace                    -> C1's `pipeline_trace`    (owner C1, proposal pending)

They are internal to C1 and get replaced by the `j26_contracts` models once those contracts
are agreed. Nothing outside C1 should depend on their shape.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, TypedDict

from j26_contracts.retrieval import RetrievalOutput
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from tax_calculation_agent.engine import CalcResult, CalcStep
from tax_calculation_agent.ruleset import RuleSet

RequestId = Annotated[str, StringConstraints(pattern=r"^req-[A-Za-z0-9-]+$")]
Confidence = Annotated[Decimal, Field(ge=0, le=1)]
Decision = Literal["PASS", "REGENERATE", "HUMAN_REVIEW"]
RerunStage = Literal["retrieve", "calculate", "explain"]
Status = Literal["delivered", "needs_review", "failed"]

MAX_ATTEMPTS = 2  # a trace is regenerated at most once (contracts/README.md)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaxRequest(_Frozen):
    """One user request. Amounts typed by the user win over amounts read from documents."""

    request_id: RequestId
    year_of_assessment: str
    employment_income: Decimal | None = None
    apit_paid: Decimal | None = None
    documents: tuple[str, ...] = ()  # document refs handed to C2


# --- PROVISIONAL: C2 ocr_output ------------------------------------------------------------


class OcrField(_Frozen):
    name: str
    value: Decimal
    confidence: Confidence


class OcrResult(_Frozen):
    document_id: str
    document_type: str
    fields: tuple[OcrField, ...]


# --- PROVISIONAL: C4 assurance_verdict -----------------------------------------------------


class FailedClaim(_Frozen):
    step_id: str
    reason: str


class AssuranceVerdict(_Frozen):
    request_id: RequestId
    attempt: Annotated[int, Field(ge=1, le=MAX_ATTEMPTS)]
    decision: Decision
    failed_claims: tuple[FailedClaim, ...] = ()
    rerun_from: RerunStage = "calculate"
    reasons: tuple[str, ...] = ()


# --- C1 internal -----------------------------------------------------------------------------


class ProfileValue(_Frozen):
    value: Decimal
    source: Literal["user", "ocr"]
    confidence: Confidence
    document_id: str | None = None


class TaxpayerProfile(_Frozen):
    """v1: one resident individual, employment income only."""

    employment_income: ProfileValue
    apit_paid: ProfileValue


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


class ConfidenceBreakdown(_Frozen):
    """CLAUDE.md section 8, v1 formula. Every term is kept so the score can be audited."""

    ocr: Confidence
    retrieval: Confidence
    interpretation: Confidence
    explanation: Confidence
    conflict_penalty: Confidence
    score: Confidence


class PipelineTrace(_Frozen):
    """PROVISIONAL: what C1 sends C4. Superseded by the `pipeline_trace` contract proposal."""

    request_id: RequestId
    attempt: Annotated[int, Field(ge=1, le=MAX_ATTEMPTS)]
    year_of_assessment: str
    profile: TaxpayerProfile
    rules: RuleSet
    retrieval_rule_ids: tuple[str, ...]
    calc_steps: tuple[CalcStep, ...]
    claims: tuple[Claim, ...]
    confidence: ConfidenceBreakdown
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]


class OrchestratorResult(_Frozen):
    request_id: RequestId
    status: Status
    attempt: int
    calc: CalcResult | None = None
    explanation: Explanation | None = None
    confidence: ConfidenceBreakdown | None = None
    verdicts: tuple[AssuranceVerdict, ...] = ()
    review_reasons: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    stages_run: tuple[str, ...] = ()


class PipelineState(TypedDict, total=False):
    """Shared state of one run. Each stage reads what it needs and returns only what it sets."""

    request: TaxRequest
    attempt: int
    ocr_results: tuple[OcrResult, ...]
    profile: TaxpayerProfile
    retrieval: RetrievalOutput
    rules: RuleSet
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]
    calc: CalcResult
    explanation: Explanation
    confidence: ConfidenceBreakdown
    trace: PipelineTrace
    verdicts: tuple[AssuranceVerdict, ...]
    rerun_from: RerunStage
    error: str
    stages_run: tuple[str, ...]
    result: OrchestratorResult
