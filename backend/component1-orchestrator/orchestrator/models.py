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

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, TypedDict

from j26_contracts.retrieval import RetrievalOutput
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from explanation_agent.models import Claim, Explanation, LlmCall  # noqa: F401  (re-exported)
from tax_calculation_agent.engine import CalcResult, CalcStep, ReliefClaim
from tax_calculation_agent.ruleset import RuleSet

RequestId = Annotated[str, StringConstraints(pattern=r"^req-[A-Za-z0-9-]+$")]
Confidence = Annotated[Decimal, Field(ge=0, le=1)]
Decision = Literal["PASS", "REGENERATE", "HUMAN_REVIEW"]
RerunStage = Literal["retrieve", "calculate", "explain"]
Status = Literal["delivered", "needs_review", "failed"]
Assurance = Literal["mock", "live", "unavailable"]  # who checked the result, if anyone
Answer = Literal["yes", "no"]  # to a relief condition question; unanswered = left out
# applied: in the calculation. not_applied: a condition fails (date, a "No" answer).
# needs_review: something is missing, so it is not applied and a person must look.
# not_supported: needs an income type C1 does not calculate yet. no_relief: C3 found none.
ReliefStatus = Literal["applied", "not_applied", "needs_review", "not_supported", "no_relief"]
# payslip: used for income and APIT. relief_candidate: invoice/receipt, checked for reliefs.
# not_analysed: bank statements and unknown documents, listed only.
DocumentRole = Literal["payslip", "relief_candidate", "not_analysed"]

MAX_ATTEMPTS = 2  # a trace is regenerated at most once (contracts/README.md)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentInput(_Frozen):
    """What the user confirmed or answered about one uploaded invoice/receipt.

    C2's values only pre-fill the form: a relief is applied only on the amount and date the
    user confirmed here, and only when every condition question is answered.
    """

    document: str  # the document ref, as in TaxRequest.documents
    payment_for: str | None = None  # "What was this payment for?"
    amount_paid: Decimal | None = None  # confirmed by the user
    document_date: date | None = None  # confirmed by the user
    answers: dict[str, Answer] = {}  # question_id (CONDITION_QUESTIONS) -> answer


class TaxRequest(_Frozen):
    """One user request. Amounts typed by the user win over amounts read from documents."""

    request_id: RequestId
    year_of_assessment: str
    employment_income: Decimal | None = None
    apit_paid: Decimal | None = None
    documents: tuple[str, ...] = ()  # document refs handed to C2
    document_inputs: tuple[DocumentInput, ...] = ()


# --- PROVISIONAL: C2 ocr_output ------------------------------------------------------------


class OcrField(_Frozen):
    name: str
    value: Decimal
    confidence: Confidence


class OcrResult(_Frozen):
    document_id: str
    document_type: str
    fields: tuple[OcrField, ...]
    # Invoices and receipts: what C2 read, used only to pre-fill the user's confirmation
    issuer: str | None = None
    document_date: date | None = None
    total_amount: Decimal | None = None


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


class DocumentSummary(_Frozen):
    """One uploaded document and what C1 did with it."""

    document_id: str
    document_type: str
    role: DocumentRole
    note: str = ""


class ReliefQuestion(_Frozen):
    question_id: str
    text: str
    answer: Answer | None = None  # None = not answered yet


class ReliefFinding(_Frozen):
    """A relief found (or not) for one invoice/receipt, and why it was or was not applied."""

    document_id: str
    status: ReliefStatus
    reasons: tuple[str, ...] = ()
    query: str | None = None  # what C3 was asked (issuer / payment purpose, PII stripped)
    rule_id: str | None = None
    rule_key: str | None = None
    title: str | None = None
    citation: str | None = None
    rule_text: str | None = None  # C3's text, conditions included, shown as is
    questions: tuple[ReliefQuestion, ...] = ()
    amount_paid: Decimal | None = None
    remaining_claimable: Decimal | None = None  # paid above this YA's cap; display only
    retrieval_confidence: Confidence | None = None


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
    explanation: Explanation  # what the user is shown, with used_fallback/hallucination_flag
    llm_calls: tuple[LlmCall, ...]  # this attempt's LLM calls; empty when no LLM was used
    confidence: ConfidenceBreakdown
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]


class OrchestratorResult(_Frozen):
    request_id: RequestId
    status: Status
    attempt: int
    calc: CalcResult | None = None
    rules: RuleSet | None = None  # what calc's citation_refs point to (citation + evidence)
    explanation: Explanation | None = None
    confidence: ConfidenceBreakdown | None = None
    verdicts: tuple[AssuranceVerdict, ...] = ()
    review_reasons: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    stages_run: tuple[str, ...] = ()
    documents: tuple[DocumentSummary, ...] = ()
    relief_findings: tuple[ReliefFinding, ...] = ()
    pii_redacted: int = 0  # PII values redacted from the explanation prompt; not sent to C4
    assurance: Assurance = "mock"
    assurance_note: str = ""  # says what checked the result, or why nothing did


class PipelineState(TypedDict, total=False):
    """Shared state of one run. Each stage reads what it needs and returns only what it sets."""

    request: TaxRequest
    attempt: int
    ocr_results: tuple[OcrResult, ...]
    profile: TaxpayerProfile
    documents: tuple[DocumentSummary, ...]
    relief_candidates: tuple[OcrResult, ...]  # invoices and receipts
    relief_claims: tuple[ReliefClaim, ...]  # reliefs to apply, one per rule
    relief_findings: tuple[ReliefFinding, ...]
    relief_retrievals: tuple[RetrievalOutput, ...]  # C3's answers for the relief queries
    retrieval: RetrievalOutput
    rules: RuleSet
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]
    calc: CalcResult
    explanation: Explanation
    explain_warnings: tuple[str, ...]  # replaced on every explain run, unlike `warnings`
    llm_calls: tuple[LlmCall, ...]
    pii_redacted: int
    confidence: ConfidenceBreakdown
    trace: PipelineTrace
    verdicts: tuple[AssuranceVerdict, ...]
    assurance_error: str  # C4 could not check the latest attempt; not fatal, unlike `error`
    rerun_from: RerunStage
    error: str
    stages_run: tuple[str, ...]
    result: OrchestratorResult
