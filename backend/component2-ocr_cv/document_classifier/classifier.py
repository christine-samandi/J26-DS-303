"""Document type classification: payslip | bank_statement | invoice | receipt | unknown (FR8).

Why this module exists:
- Each document type needs a different extractor (a payslip has EPF/ETF, a utility bill has
  VAT/SSCL), so the type must be known BEFORE field extraction.
- Just as important: a document that is NOT a financial document (a course outline, a letter)
  must be recognised as 'unknown' instead of being forced into payslip fields. Saying "I don't
  know" is safer for the tax assistant than inventing EPF values.

Method: count the distinct type-specific key phrases found in the OCR text (English, Sinhala,
Tamil). The type with the most evidence wins only if it has enough evidence AND clearly beats the
runner-up; otherwise the answer is 'unknown'. Simple, explainable, and needs no training data.
Sinhala/Tamil phrases must be verified against real documents (TODO).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

KEYWORDS: Dict[str, List[str]] = {
    "payslip": [
        "pay slip", "payslip", "salary slip", "pay statement", "basic salary", "basic pay", "gross salary",
        "gross pay", "net salary", "net pay", "epf", "etf", "apit", "paye", "allowance", "allowances",
        "total deductions", "earnings", "employee no", "employee id", "designation",
        "වැටුප් පත්‍රිකාව", "මූලික වැටුප", "සම්බළ", "சம்பளச் சீட்டு", "அடிப்படைச் சம்பளம்",   # TODO verify
    ],
    "bank_statement": [
        "statement of account", "account statement", "bank statement", "opening balance", "closing balance",
        "balance brought forward", "brought forward", "withdrawal", "withdrawals", "deposit", "deposits",
        "account no", "account number", "transaction", "value date", "cheque", "debit", "credit",
        "ගිණුම් ප්‍රකාශය", "வங்கி அறிக்கை",                                                   # TODO verify
    ],
    "invoice": [
        "invoice", "tax invoice", "bill date", "billing period", "due date", "amount due", "total due",
        "payment due", "previous balance", "account summary", "vat", "sscl", "subscriber", "customer id",
        "broadband", "usage", "monthly rental", "telephone", "electricity", "water bill",
        "බිල්පත", "பட்டியல்",                                                                  # TODO verify
    ],
    "receipt": [
        "receipt", "cash", "change", "subtotal", "sub total", "qty", "item", "cashier", "tendered",
        "thank you", "pos", "card no", "bill no", "price",
        "රිසිට්පත", "ரசீது",                                                                    # TODO verify
    ],
}

MIN_EVIDENCE = 2      # at least 2 different key phrases of the winning type
MIN_MARGIN = 1        # and at least 1 more than the runner-up


@dataclass
class Classification:
    document_type: str
    confidence: float
    scores: Dict[str, int] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w඀-෿஀-௿ ]+", " ", text)
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def classify_text(text: str) -> Classification:
    norm = _normalise(text)
    evidence = {t: [k for k in kws if f" {_normalise(k).strip()} " in norm] for t, kws in KEYWORDS.items()}
    scores = {t: len(v) for t, v in evidence.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < MIN_EVIDENCE:
        return Classification("unknown", 0.0, scores, evidence,
                              f"not enough financial-document evidence (best: {best} with {s1} key phrase(s))")
    if s1 - s2 < MIN_MARGIN:
        return Classification("unknown", 0.0, scores, evidence,
                              f"ambiguous: {ranked[0][0]} and {ranked[1][0]} both have {s1} key phrases")
    conf = min(0.99, 0.5 + 0.08 * s1 + 0.05 * (s1 - s2))
    return Classification(best, round(conf, 3), scores, {best: evidence[best]},
                          f"{s1} {best} key phrases vs {s2} for the next type")


def classify_tokens(tokens: Sequence, min_confidence: float = 0.4) -> Classification:
    """Classify from OCR tokens, ignoring very low-confidence fragments (noise, logos)."""
    return classify_text(" ".join(t.text for t in tokens if t.confidence >= min_confidence))
