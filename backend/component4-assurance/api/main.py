
from decimal import Decimal

from fastapi import FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from assurance_pipeline.pipeline import assure_tax_total


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
