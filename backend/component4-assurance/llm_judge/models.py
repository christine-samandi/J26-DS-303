
from datetime import date
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class JudgeVerdictType(str, Enum):
    """Possible semantic evidence-verification outcomes."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class JudgeEvidence(BaseModel):
    """Evidence supplied for checking a claim."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    text: str
    legal_reference: str | None = None
    effective_from: date | None = None
    effective_until: date | None = None

    @field_validator("source_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence fields cannot be blank.")
        return value.strip()

    @model_validator(mode="after")
    def validate_effective_dates(self):
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_from > self.effective_until
        ):
            raise ValueError(
                "Evidence effective_from must not exceed effective_until."
            )
        return self


class JudgeRequest(BaseModel):
    """A tax-related claim submitted for semantic evaluation."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    assessment_date: date
    evidence: list[JudgeEvidence] = Field(default_factory=list)

    @field_validator("claim_id", "claim_text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim fields cannot be blank.")
        return value.strip()

    @model_validator(mode="after")
    def validate_unique_sources(self):
        source_ids = [
            source.source_id for source in self.evidence
        ]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "Duplicate evidence source IDs are not allowed."
            )

        return self


class JudgeVerdict(BaseModel):
    """Validated structured response produced by a judge."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: JudgeVerdictType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence_source_ids: list[str] = Field(default_factory=list)

    @field_validator("claim_id", "reason")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Judge response fields cannot be blank.")
        return value.strip()

    @model_validator(mode="after")
    def validate_evidence_references(self):
        if len(self.evidence_source_ids) != len(
            set(self.evidence_source_ids)
        ):
            raise ValueError("Duplicate evidence references.")

        if (
            self.verdict in (
                JudgeVerdictType.SUPPORTED,
                JudgeVerdictType.CONTRADICTED,
            )
            and not self.evidence_source_ids
        ):
            raise ValueError(
                "Supported or contradicted verdicts require evidence."
            )

        return self


def parse_judge_response(
    raw_response: str,
    request: JudgeRequest,
) -> JudgeVerdict:
    """Validate a judge's JSON response against the original request."""

    verdict = JudgeVerdict.model_validate_json(raw_response)

    if verdict.claim_id != request.claim_id:
        raise ValueError("Judge response claim ID does not match.")

    allowed_source_ids = {
        source.source_id for source in request.evidence
    }

    for source_id in verdict.evidence_source_ids:
        if source_id not in allowed_source_ids:
            raise ValueError(
                f"Judge referenced unknown source: {source_id}"
            )

    return verdict
