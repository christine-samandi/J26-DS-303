from datetime import datetime
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from decision_engine.engine import AssuranceAction
from human_review.models import (
    ReviewCase,
    ReviewCaseStatus,
    ReviewerDecision,
)
from human_review.repository import (
    ReviewCaseNotFoundError,
    ReviewCaseRepository,
)
from human_review.service import create_review_case

from assurance_pipeline.pipeline import (
    assure_citation,
    assure_combined,
    assure_line_item_total,
    assure_tax_total,
)
from citation_verifier.verifier import CitationClaim, EvidenceSource


app = FastAPI(
    title="Component 4 Assurance API",
    version="0.1.0",
    description="Independent assurance service for the Agentic AI Tax Assistant.",
)


class TotalCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reported_total: Decimal
    calculated_total: Decimal
    regeneration_attempts: int = Field(default=0, ge=0)

    @field_validator(
        "reported_total",
        "calculated_total",
        mode="before",
    )
    @classmethod
    def require_decimal_string(cls, value):
        if not isinstance(value, str):
            raise ValueError(
                "Monetary totals must be sent as decimal strings."
            )
        return value

    @field_validator("reported_total", "calculated_total")
    @classmethod
    def validate_finite_total(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Monetary totals must be finite.")
        return value


class LineItemCheckRequest(BaseModel):
    """Input model for independent line-item verification."""

    model_config = ConfigDict(extra="forbid")

    reported_total: Decimal
    line_items: list[Decimal] = Field(min_length=1)
    regeneration_attempts: int = Field(default=0, ge=0)

    @field_validator("reported_total", mode="before")
    @classmethod
    def require_reported_total_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("Reported total must be a decimal string.")
        return value

    @field_validator("line_items", mode="before")
    @classmethod
    def require_line_item_strings(cls, value):
        if not isinstance(value, list):
            raise ValueError("Line items must be a list.")

        if any(not isinstance(item, str) for item in value):
            raise ValueError("Each line item must be a decimal string.")

        return value

    @field_validator("reported_total")
    @classmethod
    def validate_reported_total(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Reported total must be finite.")
        return value

    @field_validator("line_items")
    @classmethod
    def validate_line_items(cls, values: list[Decimal]) -> list[Decimal]:
        if any(not item.is_finite() for item in values):
            raise ValueError("All line items must be finite.")
        return values


class CitationClaimRequest(BaseModel):
    """A generated claim with its citation."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    source_id: str | None = None
    excerpt: str | None = None

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim ID cannot be empty.")
        return value


class EvidenceSourceRequest(BaseModel):
    """An evidence source available for verification."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    text: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source ID cannot be empty.")
        return value


class CitationCheckRequest(BaseModel):
    """Complete request for citation assurance."""

    model_config = ConfigDict(extra="forbid")

    claim: CitationClaimRequest
    sources: list[EvidenceSourceRequest]
    regeneration_attempts: int = Field(default=0, ge=0)

    @field_validator("sources")
    @classmethod
    def validate_unique_sources(
        cls,
        sources: list[EvidenceSourceRequest],
    ) -> list[EvidenceSourceRequest]:
        source_ids = [source.source_id for source in sources]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Duplicate source ID detected.")

        return sources


class CombinedCheckRequest(LineItemCheckRequest):
    """Request for combined numerical and citation assurance."""

    claims: list[CitationClaimRequest] = Field(min_length=1)
    sources: list[EvidenceSourceRequest]

    @field_validator("sources")
    @classmethod
    def validate_unique_sources(
        cls,
        sources: list[EvidenceSourceRequest],
    ) -> list[EvidenceSourceRequest]:
        source_ids = [source.source_id for source in sources]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Duplicate source ID detected.")

        return sources

    @field_validator("claims")
    @classmethod
    def validate_unique_claims(
        cls,
        claims: list[CitationClaimRequest],
    ) -> list[CitationClaimRequest]:
        claim_ids = [claim.claim_id for claim in claims]

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Duplicate claim ID detected.")

        return claims

class FindingResponse(BaseModel):
    check_name: str
    passed: bool
    message: str
    severity: str
    error_code: str
    expected_value: str | None
    actual_value: str | None


class DecisionResponse(BaseModel):
    action: str
    reason: str
    failed_checks: int


class AssuranceResponse(BaseModel):
    findings: list[FindingResponse]
    decision: DecisionResponse


@app.get("/health")
def health_check():
    return {"status": "ok", "component": "component4-assurance"}


@app.post(
    "/assurance/total-check",
    response_model=AssuranceResponse,
)
def verify_tax_total(request: TotalCheckRequest):
    result = assure_tax_total(
        reported_total=request.reported_total,
        calculated_total=request.calculated_total,
        regeneration_attempts=request.regeneration_attempts,
    )

    findings = [
        FindingResponse(
            check_name=finding.check_name,
            passed=finding.passed,
            message=finding.message,
            severity=finding.severity.value,
            error_code=finding.error_code.value,
            expected_value=(
                str(finding.expected_value)
                if finding.expected_value is not None
                else None
            ),
            actual_value=(
                str(finding.actual_value)
                if finding.actual_value is not None
                else None
            ),
        )
        for finding in result.findings
    ]

    decision = DecisionResponse(
        action=result.decision.action.value,
        reason=result.decision.reason,
        failed_checks=result.decision.failed_checks,
    )

    return AssuranceResponse(
        findings=findings,
        decision=decision,
    )


@app.post(
    "/assurance/line-item-check",
    response_model=AssuranceResponse,
)
def verify_line_item_total(request: LineItemCheckRequest):
    """Verify a reported total using independently summed line items."""

    result = assure_line_item_total(
        reported_total=request.reported_total,
        line_items=request.line_items,
        regeneration_attempts=request.regeneration_attempts,
    )

    findings = [
        FindingResponse(
            check_name=finding.check_name,
            passed=finding.passed,
            message=finding.message,
            severity=finding.severity.value,
            error_code=finding.error_code.value,
            expected_value=(
                str(finding.expected_value)
                if finding.expected_value is not None
                else None
            ),
            actual_value=(
                str(finding.actual_value)
                if finding.actual_value is not None
                else None
            ),
        )
        for finding in result.findings
    ]

    decision = DecisionResponse(
        action=result.decision.action.value,
        reason=result.decision.reason,
        failed_checks=result.decision.failed_checks,
    )

    return AssuranceResponse(
        findings=findings,
        decision=decision,
    )


@app.post(
    "/assurance/citation-check",
    response_model=AssuranceResponse,
)
def verify_claim_citation(request: CitationCheckRequest):
    """Verify citation traceability and return an assurance decision."""

    claim = CitationClaim(
        claim_id=request.claim.claim_id,
        source_id=request.claim.source_id,
        excerpt=request.claim.excerpt,
    )

    sources = [
        EvidenceSource(
            source_id=source.source_id,
            text=source.text,
        )
        for source in request.sources
    ]

    result = assure_citation(
        claim=claim,
        sources=sources,
        regeneration_attempts=request.regeneration_attempts,
    )

    findings = [
        FindingResponse(
            check_name=finding.check_name,
            passed=finding.passed,
            message=finding.message,
            severity=finding.severity.value,
            error_code=finding.error_code.value,
            expected_value=(
                str(finding.expected_value)
                if finding.expected_value is not None
                else None
            ),
            actual_value=(
                str(finding.actual_value)
                if finding.actual_value is not None
                else None
            ),
        )
        for finding in result.findings
    ]

    decision = DecisionResponse(
        action=result.decision.action.value,
        reason=result.decision.reason,
        failed_checks=result.decision.failed_checks,
    )

    return AssuranceResponse(
        findings=findings,
        decision=decision,
    )


@app.post(
    "/assurance/combined-check",
    response_model=AssuranceResponse,
)
def verify_combined_assurance(request: CombinedCheckRequest):
    """Verify numerical consistency and multiple citations together."""

    claims = [
        CitationClaim(
            claim_id=claim.claim_id,
            source_id=claim.source_id,
            excerpt=claim.excerpt,
        )
        for claim in request.claims
    ]

    sources = [
        EvidenceSource(
            source_id=source.source_id,
            text=source.text,
        )
        for source in request.sources
    ]

    result = assure_combined(
        reported_total=request.reported_total,
        line_items=request.line_items,
        claims=claims,
        sources=sources,
        regeneration_attempts=request.regeneration_attempts,
    )

    findings = [
        FindingResponse(
            check_name=finding.check_name,
            passed=finding.passed,
            message=finding.message,
            severity=finding.severity.value,
            error_code=finding.error_code.value,
            expected_value=(
                str(finding.expected_value)
                if finding.expected_value is not None
                else None
            ),
            actual_value=(
                str(finding.actual_value)
                if finding.actual_value is not None
                else None
            ),
        )
        for finding in result.findings
    ]

    decision = DecisionResponse(
        action=result.decision.action.value,
        reason=result.decision.reason,
        failed_checks=result.decision.failed_checks,
    )

    return AssuranceResponse(
        findings=findings,
        decision=decision,
    )


class CreateReviewCaseRequest(CombinedCheckRequest):
    request_id: str | None = None

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Request ID cannot be blank.")
        return value


class CompleteReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_id: str
    decision: ReviewerDecision
    comment: str

    @field_validator("reviewer_id", "comment")
    @classmethod
    def validate_nonblank(cls, value):
        if not value.strip():
            raise ValueError("Field cannot be blank.")
        return value


class ReviewFeedbackResponse(BaseModel):
    reviewer_id: str
    decision: ReviewerDecision
    comment: str
    reviewed_at: datetime


class ReviewCaseResponse(BaseModel):
    case_id: str
    request_id: str | None
    status: ReviewCaseStatus
    created_at: datetime
    findings: list[FindingResponse]
    decision: DecisionResponse
    feedback: ReviewFeedbackResponse | None


# Prototype repository. Cases are stored in memory.
review_repository = ReviewCaseRepository()


def review_case_response(case: ReviewCase) -> ReviewCaseResponse:
    """Convert a review case into a structured API response."""

    findings = [
        FindingResponse(
            check_name=finding.check_name,
            passed=finding.passed,
            message=finding.message,
            severity=finding.severity.value,
            error_code=finding.error_code.value,
            expected_value=(
                str(finding.expected_value)
                if finding.expected_value is not None
                else None
            ),
            actual_value=(
                str(finding.actual_value)
                if finding.actual_value is not None
                else None
            ),
        )
        for finding in case.findings
    ]

    feedback = None

    if case.feedback is not None:
        feedback = ReviewFeedbackResponse(
            reviewer_id=case.feedback.reviewer_id,
            decision=case.feedback.decision,
            comment=case.feedback.comment,
            reviewed_at=case.feedback.reviewed_at,
        )

    return ReviewCaseResponse(
        case_id=case.case_id,
        request_id=case.request_id,
        status=case.status,
        created_at=case.created_at,
        findings=findings,
        decision=DecisionResponse(
            action=case.decision.action.value,
            reason=case.decision.reason,
            failed_checks=case.decision.failed_checks,
        ),
        feedback=feedback,
    )


@app.post(
    "/human-review/cases",
    response_model=ReviewCaseResponse,
    status_code=201,
)
def create_human_review_case(request: CreateReviewCaseRequest):
    claims = [
        CitationClaim(
            claim_id=claim.claim_id,
            source_id=claim.source_id,
            excerpt=claim.excerpt,
        )
        for claim in request.claims
    ]

    sources = [
        EvidenceSource(
            source_id=source.source_id,
            text=source.text,
        )
        for source in request.sources
    ]

    result = assure_combined(
        reported_total=request.reported_total,
        line_items=request.line_items,
        claims=claims,
        sources=sources,
        regeneration_attempts=request.regeneration_attempts,
    )

    if result.decision.action != AssuranceAction.HUMAN_REVIEW:
        raise HTTPException(
            status_code=409,
            detail="This result does not require human review.",
        )

    case = create_review_case(
        result,
        request_id=request.request_id,
    )

    review_repository.add_case(case)

    return review_case_response(case)


@app.get(
    "/human-review/cases",
    response_model=list[ReviewCaseResponse],
)
def list_human_review_cases(
    status: ReviewCaseStatus | None = None,
):
    cases = review_repository.list_cases(status=status)
    return [review_case_response(case) for case in cases]


@app.get(
    "/human-review/cases/{case_id}",
    response_model=ReviewCaseResponse,
)
def get_human_review_case(case_id: str):
    try:
        case = review_repository.get_case(case_id)
    except ReviewCaseNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Review case not found.",
        )

    return review_case_response(case)


@app.post(
    "/human-review/cases/{case_id}/start",
    response_model=ReviewCaseResponse,
)
def start_human_review(case_id: str):
    try:
        case = review_repository.start_review(case_id)
    except ReviewCaseNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Review case not found.",
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    return review_case_response(case)


@app.post(
    "/human-review/cases/{case_id}/complete",
    response_model=ReviewCaseResponse,
)
def complete_human_review(
    case_id: str,
    request: CompleteReviewRequest,
):
    try:
        case = review_repository.complete_review(
            case_id=case_id,
            reviewer_id=request.reviewer_id,
            decision=request.decision,
            comment=request.comment,
        )
    except ReviewCaseNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Review case not found.",
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    return review_case_response(case)
