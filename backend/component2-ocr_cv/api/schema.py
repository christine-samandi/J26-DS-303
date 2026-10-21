"""Pydantic models mirroring contracts/schemas/ocr_output_schema.json (v0.2).

Why: FastAPI validates every response against these models, so the pipeline
can never return JSON that breaks Components 1, 3 or 4.

v0.2 (2026-10-21) is ADDITIVE: invoices, receipts and quotations add line items, tax components and
printed attributes. Payslips and bank statements still return schema_version "0.1" with exactly the
same fields, so Component 1's existing check (schema_version == "0.1") keeps working for them.
"""
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

REVIEW_THRESHOLD = 0.95  # FR7: fields below this are flagged for the user

DocType = Literal["payslip", "bank_statement", "invoice", "receipt", "quotation", "unknown"]
FieldName = Literal[
    "issuer", "pay_period", "document_date", "gross_income", "epf_employee", "epf_employer",
    "etf", "apit", "vat", "sscl", "total_deductions", "net_amount", "total_amount",
    # v0.2: invoices, receipts, quotations
    "document_number", "due_date", "seller_reg_no", "subtotal", "discount", "amount_paid",
    "balance_due", "previous_balance",
]
BBox = List[int]


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: FieldName
    value: Union[float, str, None]
    raw_text: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    bbox: BBox = Field(min_length=4, max_length=4, description="[x, y, w, h] in original-image pixels")
    tax_entity: Optional[str] = None
    needs_review: bool


class LineItem(BaseModel):
    """One row of an items table. The description is kept exactly as printed, because other
    components decide what the item is (e.g. 'Solar PV panel 550W')."""
    model_config = ConfigDict(extra="forbid")
    description: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    discount: Optional[float] = None
    tax_rate: Optional[float] = Field(default=None, ge=0, le=1, description="0.18 = 18%")
    amount: Optional[float] = None
    amount_includes_tax: Optional[bool] = None
    confidence: float = Field(ge=0, le=1)
    bbox: BBox = Field(min_length=4, max_length=4)
    needs_review: bool


class TaxComponent(BaseModel):
    """A tax printed on the document: VAT, SSCL, SGST, CGST ..."""
    model_config = ConfigDict(extra="forbid")
    name: str
    rate: Optional[float] = Field(default=None, ge=0, le=1)
    amount: float
    confidence: float = Field(ge=0, le=1)
    bbox: BBox = Field(min_length=4, max_length=4)
    needs_review: bool


class Attribute(BaseModel):
    """A printed key-value fact that is not a standard field, e.g. 'System Size: 5 kW'."""
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str
    confidence: float = Field(ge=0, le=1)
    bbox: BBox = Field(min_length=4, max_length=4)


class OCROutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["0.1", "0.2"] = "0.1"
    doc_id: str
    document_type: DocType
    languages: List[Literal["si", "ta", "en"]]
    extracted_fields: List[ExtractedField]
    overall_confidence: float = Field(ge=0, le=1)
    needs_review: bool
    engine: Optional[str] = None
    processing_ms: Optional[int] = Field(default=None, ge=0)
    line_items: Optional[List[LineItem]] = None
    tax_components: Optional[List[TaxComponent]] = None
    attributes: Optional[List[Attribute]] = None
