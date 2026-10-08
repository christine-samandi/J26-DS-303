"""Retrieval contract: C1 / C4  ->  C3 (GraphRAG regulatory retrieval)  ->  C1 / C4.

Owner: C3 (Elvitigala C S). Changes need approval from all four members (see CONTRACT_CHANGELOG.md).

These Pydantic models ARE the contract. The JSON Schema files
`contracts/schemas/retrieval_request.schema.json` and `retrieval_output.schema.json` are
generated from them:   python -m j26_contracts.export

Validate data you receive:
    from j26_contracts.retrieval import RetrievalOutput
    output = RetrievalOutput.model_validate(response_json)   # raises ValidationError if invalid

Produce JSON in contract form (optional fields that were never set are left out):
    to_contract_json(model)

Pydantic also checks a few things JSON Schema cannot express, marked "[extra check]" below.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

IncomeType = Literal["employment", "business", "investment", "other"]
TaxpayerCategory = Literal[
    "resident_individual", "non_resident_individual", "senior_citizen", "any"
]
RuleType = Literal[
    "tax_slab", "tax_rate", "relief", "deduction", "exemption", "penalty", "definition", "other"
]
Caller = Literal["c1-orchestrator", "c4-assurance", "evaluation", "manual"]
Warning = Literal[
    "NO_MATCH",
    "NO_RULE_IN_FORCE_ON_DATE",
    "LOW_CONFIDENCE",
    "AMBIGUOUS_QUERY",
    "AS_OF_DATE_BEFORE_CORPUS",
    "PARTIAL_CORPUS_COVERAGE",
]
RetrievalMode = Literal["graphrag", "flat_rag", "mock"]
DocumentType = Literal["act", "amendment_act", "gazette", "circular", "notice", "guide"]
RuleStatus = Literal["active", "replaced", "outdated"]
Relation = Literal[
    "DEFINED_IN", "APPLIES_TO", "AMENDS", "SUPERSEDES", "PART_OF", "HAS_BAND", "REFERS_TO"
]
EntityLabel = Literal[
    "AMOUNT",
    "RATE",
    "DATE",
    "SECTION_REF",
    "RELIEF",
    "DEDUCTION",
    "EXEMPTION",
    "INCOME_TYPE",
    "TAXPAYER_CATEGORY",
    "TAX_SLAB",
]
ConfidenceLevel = Literal["high", "medium", "low"]
Period = Literal["year_of_assessment", "month", "one_off"]

# ---------------------------------------------------------------------------
# Reusable field types
# ---------------------------------------------------------------------------

RuleId = Annotated[str, StringConstraints(pattern=r"^RULE-[A-Z0-9_-]+$")]
DocumentId = Annotated[str, StringConstraints(pattern=r"^DOC-[A-Z0-9_-]+$")]
ChunkId = Annotated[str, StringConstraints(pattern=r"^CHUNK-[A-Z0-9_-]+$")]
RuleKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
RequestId = Annotated[str, StringConstraints(min_length=1, max_length=128)]


def Number(minimum: float, maximum: float | None = None):  # noqa: N802 — used like a type
    """A number (whole numbers stay whole, e.g. 1800000 not 1800000.0) within a range."""
    json_schema: dict = {"type": "number", "minimum": minimum}
    if maximum is not None:
        json_schema["maximum"] = maximum
    return Annotated[int | float, Field(ge=minimum, le=maximum), WithJsonSchema(json_schema)]


NonNegative = Number(0)
Percent = Number(0, 100)
Probability = Number(0, 1)
TopK = Annotated[int, Field(ge=1, le=20)]


def _unique(items: list) -> list:
    if len(items) != len(set(items)):
        raise ValueError("list items must be unique")
    return items


def UniqueList(item_type):  # noqa: N802 — used like a type
    return Annotated[
        list[item_type], AfterValidator(_unique), Field(json_schema_extra={"uniqueItems": True})
    ]


class ContractModel(BaseModel):
    """Base class: unknown fields are rejected, like `additionalProperties: false`."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Request  (C1 / C4  →  C3)
# ---------------------------------------------------------------------------


class RetrievalRequest(ContractModel):
    """Query sent to C3. Needs at least one of query_text, rule_types or rule_ids."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "request_id": "demo-1",
                    "query_text": "What personal relief applied?",
                    "as_of_date": "2021-06-01",
                    "taxpayer_category": "resident_individual",
                    "include_history": True,
                }
            ],
            # same rule as _needs_a_query, written into the generated JSON Schema
            "anyOf": [
                {"required": ["query_text"], "properties": {"query_text": {"type": "string"}}},
                {"required": ["rule_types"], "properties": {"rule_types": {"type": "array"}}},
                {"required": ["rule_ids"], "properties": {"rule_ids": {"type": "array"}}},
            ],
        },
    )

    schema_version: Literal["1.0.0"]
    request_id: RequestId = Field(description="Caller-generated ID, echoed in the response.")
    caller: Caller | None = None
    query_text: Annotated[str, StringConstraints(min_length=1, max_length=2000)] | None = Field(
        default=None, description="Natural-language question. PII must already be redacted."
    )
    as_of_date: date = Field(description="Point-in-time date: only rules in force on it apply.")
    income_types: UniqueList(IncomeType) | None = None
    taxpayer_category: TaxpayerCategory | None = None
    rule_types: Annotated[UniqueList(RuleType), Field(min_length=1)] | None = None
    rule_ids: Annotated[UniqueList(RuleId), Field(min_length=1)] | None = None
    top_k: TopK = 5
    include_history: bool = Field(
        default=False, description="Also return replaced/outdated versions of matched rules."
    )

    @model_validator(mode="after")
    def _needs_a_query(self) -> RetrievalRequest:
        if self.query_text is None and self.rule_types is None and self.rule_ids is None:
            raise ValueError("give at least one of query_text, rule_types or rule_ids")
        return self


# ---------------------------------------------------------------------------
# Response  (C3  →  C1 / C4)
# ---------------------------------------------------------------------------


class QueryEcho(ContractModel):
    query_text: str | None = None
    as_of_date: date
    income_types: list[IncomeType] | None = None
    taxpayer_category: TaxpayerCategory | None = None
    rule_types: list[RuleType] | None = None
    rule_ids: list[RuleId] | None = None
    top_k: TopK
    include_history: bool


class Money(ContractModel):
    amount: NonNegative
    currency: Literal["LKR"]
    period: Period | None = None


class SlabBand(ContractModel):
    order: Annotated[int, Field(ge=1)]
    lower: NonNegative
    upper: NonNegative | None = Field(default=None, description="null = no upper limit.")
    rate_percent: Percent


class RuleValues(ContractModel):
    """Structured numbers for C1's deterministic engine. C1 must not parse numbers from text."""

    amount: Money | None = None
    cap: Money | None = None
    rate_percent: Percent | None = None
    bands: Annotated[list[SlabBand], Field(min_length=1)] | None = None


class AppliesTo(ContractModel):
    income_types: list[IncomeType]
    taxpayer_categories: Annotated[list[TaxpayerCategory], Field(min_length=1)]


class Source(ContractModel):
    document_id: DocumentId
    document_title: NonEmptyStr
    document_type: DocumentType
    act_number: str | None = None
    section: str | None = None
    subsection: str | None = None
    paragraph: str | None = None
    schedule: str | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    citation: NonEmptyStr = Field(description="Ready-to-display citation for the user.")
    published_date: date | None = None
    url: str | None = None


class Validity(ContractModel):
    """C3 versioning: when the rule applied, and what replaced it."""

    model_config = ConfigDict(
        extra="forbid",
        # same rules as _status_rules, written into the generated JSON Schema
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "replaced"}}},
                    "then": {
                        "required": ["superseded_by"],
                        "properties": {
                            "superseded_by": {"type": "string", "pattern": "^RULE-[A-Z0-9_-]+$"}
                        },
                    },
                },
                {
                    "if": {"properties": {"status": {"const": "active"}}},
                    "then": {"properties": {"valid_to": {"type": "null"}}},
                },
            ]
        },
    )

    valid_from: date
    valid_to: date | None = Field(
        description="Last day in force (inclusive). null = still in force."
    )
    status: RuleStatus = Field(description="Lifecycle state TODAY.")
    in_force_on_as_of_date: bool = Field(description="Did the rule apply on the queried date?")
    superseded_by: RuleId | None = None
    supersedes: RuleId | None = None
    changed_by_document_id: DocumentId | None = None

    @model_validator(mode="after")
    def _status_rules(self) -> Validity:
        if self.status == "replaced" and self.superseded_by is None:
            raise ValueError("a replaced rule must name superseded_by")
        if self.status == "active" and self.valid_to is not None:
            raise ValueError("an active rule cannot have valid_to")
        # [extra check] not expressible in JSON Schema
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be before valid_from")
        return self

    def in_force_on(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)


class ConfidenceComponents(ContractModel):
    semantic_similarity: Probability
    graph_match: Probability
    extraction_confidence: Probability
    temporal_match: Probability


class Confidence(ContractModel):
    score: Probability
    level: ConfidenceLevel
    components: ConfidenceComponents


class GraphEdge(ContractModel):
    from_: str = Field(alias="from")
    relation: Relation
    to: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)


class MatchedEntity(ContractModel):
    text: str
    label: EntityLabel


class Evidence(ContractModel):
    chunk_ids: Annotated[list[ChunkId], Field(min_length=1)]
    matched_text: NonEmptyStr = Field(description="Verbatim source text. C4 verifies against it.")
    graph_path: list[GraphEdge] | None = None
    matched_entities: list[MatchedEntity] | None = None


class RetrievedRule(ContractModel):
    rank: Annotated[int, Field(ge=1)]
    rule_id: RuleId = Field(description="ID of this specific version of the rule.")
    rule_key: RuleKey | None = Field(default=None, description="Shared by all versions.")
    rule_type: RuleType
    title: NonEmptyStr
    text: NonEmptyStr
    values: RuleValues | None = None
    applies_to: AppliesTo
    source: Source
    validity: Validity
    confidence: Confidence
    evidence: Evidence


class Metadata(ContractModel):
    retrieval_mode: RetrievalMode
    latency_ms: NonNegative
    corpus_version: NonEmptyStr
    embedding_model: str | None = None
    candidates_considered: Annotated[int, Field(ge=0)] | None = None
    confidence_method: str | None = None


class RetrievalOutput(ContractModel):
    """Response from C3."""

    schema_version: Literal["1.0.0"]
    request_id: RequestId
    generated_at: AwareDatetime
    query: QueryEcho
    results: list[RetrievedRule]
    warnings: UniqueList(Warning)
    metadata: Metadata

    @model_validator(mode="after")
    def _consistent_with_query(self) -> RetrievalOutput:
        # [extra check] ranks run 1, 2, 3, ... in order
        ranks = [r.rank for r in self.results]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"results must be ranked 1..n in order, got {ranks}")
        # [extra check] in_force_on_as_of_date must agree with the dates and the query
        for r in self.results:
            expected = r.validity.in_force_on(self.query.as_of_date)
            if r.validity.in_force_on_as_of_date != expected:
                raise ValueError(
                    f"{r.rule_id}: in_force_on_as_of_date={r.validity.in_force_on_as_of_date} "
                    f"but its dates say {expected} for {self.query.as_of_date}"
                )
        # [extra check] without include_history, only rules in force may be returned
        if not self.query.include_history:
            stale = [r.rule_id for r in self.results if not r.validity.in_force_on_as_of_date]
            if stale:
                raise ValueError(
                    f"include_history is false but returned rules not in force: {stale}"
                )
        return self


def to_contract_json(model: BaseModel) -> dict:
    """Plain-JSON dict in contract form: dates as strings, unset optional fields left out."""
    return model.model_dump(mode="json", exclude_unset=True, by_alias=True)
