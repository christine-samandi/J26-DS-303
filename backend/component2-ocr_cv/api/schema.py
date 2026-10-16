"""Pydantic models mirroring contracts/schemas/ocr_output_schema.json (v0.1).

Why: FastAPI validates every response against these models, so the pipeline
can never return JSON that breaks Components 1, 3 or 4.
"""
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

REVIEW_THRESHOLD = 0.95  # FR7: fields below this are flagged for the user

DocType = Literal["payslip", "bank_statement", "invoice", "receipt", "unknown"]
FieldName = Literal[
    "issuer", "pay_period", "document_date", "gross_income", "epf_employee", "epf_employer",
    "etf", "apit", "vat", "sscl", "total_deductions", "net_amount", "total_amount",
]


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: FieldName
    value: Union[float, str, None]
    raw_text: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    bbox: List[int] = Field(min_length=4, max_length=4, description="[x, y, w, h] in original-image pixels")
    tax_entity: Optional[str] = None
    needs_review: bool


class OCROutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["0.1"] = "0.1"
    doc_id: str
    document_type: DocType
    languages: List[Literal["si", "ta", "en"]]
    extracted_fields: List[ExtractedField]
    overall_confidence: float = Field(ge=0, le=1)
    needs_review: bool
    engine: Optional[str] = None
    processing_ms: Optional[int] = Field(default=None, ge=0)
