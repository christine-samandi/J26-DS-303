"""Why: on a real payslip (PS_001) the amount column was printed half a row lower than the labels,
so OCR paired every deduction with the wrong label. These tests use a FAKE slip with the same
layout and check that the arithmetic re-pairs the rows, plus the other fixes found on that slip."""
from dataclasses import dataclass

from entity_extraction.amounts import parse_amount
from entity_extraction.extractor import extract_fields
from entity_extraction.keywords import match_field
from entity_extraction.payslip import extract_document


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows, conf=0.9):
    """rows: (label or '', amount or ''); one row per printed line."""
    toks = []
    for i, (label, amount) in enumerate(rows):
        y = 100 + 40 * i
        for j, w in enumerate(label.split()):
            toks.append(Tok(w, conf, [20 + 90 * j, y, 80, 20]))
        for k, w in enumerate(amount.split()):
            toks.append(Tok(w, conf, [600 + 70 * k, y, 65, 20]))
    return toks


# Fake slip. Earnings rows are aligned; in the deduction block every amount appears on the line of
# the NEXT label (true values: Loan 5,000, Welfare 1,000, Paye 2,500, Total 8,500, Net 111,500).
SHIFTED = [("SAMPLE TRADING (PVT) LTD", ""), ("PAY SLIP SEPTEMBER 2026", ""),
           ("Consolidated Salary", "100,000.00"), ("Fuel Allowance", "20,000.00"),
           ("TOTAL EARNINGS", ""), ("", "120,000.00"),
           ("Sport Club", ""), ("Loan Recovery", "10.00"), ("Welfare Scheme", "5,000.00"),
           ("Paye Tax", "1,000.00"), ("Total Deductions", "2,500.00"), ("Net Salary", "8,500.00"),
           ("Stamp Duty", "111,500.00"), ("Salary to Bank", "25.00")]


def test_shifted_rows_are_repaired_by_arithmetic():
    out, report = extract_document(page(SHIFTED), doc_id="FAKE")
    by = {f.field_name: f for f in out.extracted_fields}
    assert by["gross_income"].value == 120000.0
    assert by["total_deductions"].value == 8500.0 and by["net_amount"].value == 111500.0
    assert by["apit"].value == 2500.0                                  # same shift as its block
    assert report["alignment"]["applied"]
    assert report["alignment"]["moved"] == {"total_deductions": "row below", "net_amount": "row below"}
    assert "apit" in report["alignment"]["propagated"]
    assert all(by[f].needs_review for f in ("total_deductions", "net_amount", "apit"))   # user confirms
    hard = [c for c in report["checks"] if c["hard"]]
    assert hard and all(c["passed"] for c in hard)


def test_aligned_slip_is_left_alone():
    aligned = [("SAMPLE TRADING (PVT) LTD", ""), ("Gross Salary", "120,000.00"), ("Paye Tax", "2,500.00"),
               ("Total Deductions", "8,500.00"), ("Net Salary", "111,500.00")]
    out, report = extract_document(page(aligned), doc_id="OK")
    by = {f.field_name: f.value for f in out.extracted_fields}
    assert by["net_amount"] == 111500.0 and not report["alignment"]["applied"]


def test_decimal_point_lost_as_space():
    assert parse_amount("28 775 29").value == 28775.29
    assert parse_amount("47,958 82").value == 47958.82
    assert parse_amount("9.000 00").value == 9000.0
    assert parse_amount("28 775 29").corrected
    assert parse_amount("150, 000.00").value == 150000.0      # earlier behaviour kept
    assert parse_amount("150 000").value == 150000.0           # 3-digit last group: not cents


def test_specific_label_beats_generic():
    rows = [("Net bonus payment", "9,000.00"), ("Net Salary", "111,500.00")]
    assert extract_fields(page(rows))["net_amount"].value == 111500.0


def test_epf_labels_from_real_slip():
    assert match_field("EPF Corporation Contribution")[0] == "epf_employer"
    assert match_field("ETF Corporation Contribution")[0] == "etf"
    assert match_field("E.P.F Gross") is None and match_field("ETF Gross") is None   # base amounts, not contributions


def test_issuer_is_never_a_field_or_personal_line():
    rows = [("Name : Sample Person", ""), ("EPF Corporation Contribution", "2,000.00"), ("Gross Salary", "50,000.00")]
    assert "issuer" not in extract_fields(page(rows))
