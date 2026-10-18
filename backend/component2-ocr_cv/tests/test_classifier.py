"""Why: the type decides which extractor runs; a wrong 'payslip' on a non-financial document would
invent tax data. These tests cover the four types, the 'unknown' answer and language evidence."""
from dataclasses import dataclass

import pytest

from document_classifier import classify_text, classify_tokens
from entity_extraction.payslip import detect_languages, extract_document

PAYSLIP = ("LANKA TEXTILES (PVT) LTD PAY SLIP SEPTEMBER 2026 Basic Salary 120,000.00 Allowances 30,000.00 "
           "Gross Salary 150,000.00 EPF Employee 8% 12,000.00 APIT 3,500.00 Total Deductions 15,500.00 Net Salary 134,500.00")


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list


def toks(text, conf=0.9):
    return [Tok(w, conf, [20 + 60 * i, 100, 50, 20]) for i, w in enumerate(text.split())]


@pytest.mark.parametrize("text,expected", [
    (PAYSLIP, "payslip"),
    ("Commercial Bank Statement of Account Account No 1234 Opening balance 10,000.00 Deposits 5,000 "
     "Withdrawals 2,000 Closing balance 13,000.00", "bank_statement"),
    ("SLT Broadband bill Bill date 2026-09-30 Due date 2026-10-20 Monthly rental 2,500.00 VAT 450.00 "
     "SSCL 62.50 Amount due 3,012.50", "invoice"),
    ("KEELLS SUPER Receipt Item Qty Price Subtotal 1,250.00 Cash 2,000.00 Change 750.00 Thank you", "receipt"),
])
def test_types(text, expected):
    assert classify_text(text).document_type == expected


@pytest.mark.parametrize("text", [
    "Sri Lanka Institute of Information Technology Faculty of Computing BSc (Hons) Data Science course outline",
    "Dear Sir, I am writing to request leave for two days. Yours faithfully",
    "",
])
def test_non_financial_documents_are_unknown(text):
    c = classify_text(text)
    assert c.document_type == "unknown" and c.confidence == 0.0 and c.reason


def test_ambiguous_evidence_is_unknown():
    # one payslip phrase and one invoice phrase: not enough to decide
    assert classify_text("net salary due date").document_type == "unknown"


def test_low_confidence_noise_is_ignored():
    noisy = toks(PAYSLIP, conf=0.1)               # unreadable fragments must not count as evidence
    assert classify_tokens(noisy).document_type == "unknown"


def test_language_needs_real_words():
    english = toks("Sri Lanka Institute of Information Technology Faculty")
    stray = [Tok("ශ", 0.9, [0, 0, 5, 5]), Tok("ம", 0.3, [0, 0, 5, 5])]   # single chars / low confidence
    assert detect_languages(english + stray) == ["en"]
    sinhala = toks("දළ වැටුප ශුද්ධ වැටුප මූලික වැටුප")
    assert "si" in detect_languages(english + sinhala)


def test_unknown_document_gets_no_tax_fields():
    out, report = extract_document(toks("Sri Lanka Institute of Information Technology Faculty of Computing "
                                        "BSc Data Science course outline semester modules"), doc_id="DOC")
    assert out.document_type == "unknown" and out.extracted_fields == [] and out.needs_review
    assert "not recognised as a financial document" in report["notes"]["document"][0]


def test_payslip_still_extracted():
    out, report = extract_document(toks(PAYSLIP), doc_id="PS")
    assert out.document_type == "payslip" and report["classification"]["document_type"] == "payslip"


# --- Bank statements written in codes (B/F, ATM, POS), like real scanned statements -------------
# Made-up numbers. Each row: date, code, amount, running balance.
STATEMENT_ROWS = [
    "251001 B/F 10,000.00",
    "251001 ATM 2,000.00 8,000.00",
    "251003 POS 1,250.50 6,749.50",
    "251005 SALARY 50,000.00 56,749.50",
    "251007 POS 749.50 56,000.00",
    "251010 SMS 50.00 55,950.00",
    "251012 ATM 5,000.00 50,950.00",
]


def test_code_style_statement_is_bank_statement():
    # Before: only 'deposit'-style words counted, so this came back 'unknown' (or receipt via POS).
    c = classify_text("\n".join(STATEMENT_ROWS))
    assert c.document_type == "bank_statement"
    assert any(e.startswith("[layout]") for e in c.evidence["bank_statement"])
    assert c.scores["receipt"] == 0                 # POS is a statement code here, not receipt evidence


def test_running_balance_alone_is_evidence():
    # Garbled words, but the arithmetic still holds -> bank statement
    rows = [r.split()[0] + " x " + " ".join(r.split()[2:]) for r in STATEMENT_ROWS]
    assert classify_text("\n".join(rows)).document_type == "bank_statement"


def test_table_whose_balances_do_not_add_up_is_not_evidence():
    rows = ["1 100.00 900.00", "2 250.00 300.00", "3 75.00 5,000.00", "4 10.00 42.00", "5 60.00 1.00", "6 3.00 70.00"]
    c = classify_text("\n".join(rows))
    assert c.document_type == "unknown" and c.scores["bank_statement"] == 0


def test_receipt_item_lines_stay_receipt():
    receipt = ["KEELLS SUPER Receipt", "Milk 2 250.00 500.00", "Bread 1 180.00 180.00", "Rice 5 220.00 1,100.00",
               "Eggs 10 45.00 450.00", "Tea 1 650.00 650.00", "Subtotal 2,880.00", "Cash 3,000.00",
               "Change 120.00", "Thank you"]
    assert classify_text("\n".join(receipt)).document_type == "receipt"


@dataclass
class LineTok:
    text: str
    confidence: float
    bbox: list
    line_id: str


def test_tokens_keep_their_lines():
    # classify_tokens must rebuild rows from line_id, otherwise the balance check sees one long line
    tokens = [LineTok(w, 0.9, [0, 0, 10, 10], f"1_1_{i}") for i, row in enumerate(STATEMENT_ROWS) for w in row.split()]
    assert classify_tokens(tokens).document_type == "bank_statement"
