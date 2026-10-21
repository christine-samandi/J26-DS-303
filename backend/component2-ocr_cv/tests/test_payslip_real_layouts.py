"""Why: eight real payslips (two employers, English and Sinhala, phone photos) showed layouts the
synthetic slips never had: labels like 'Consolidated Salary' and 'A P I T', a label masked as a person's
name, two printed rows merged into one OCR line, a page photographed sideways. These tests rebuild those
situations with made-up numbers (no personal data) and pin the fixes."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import pytest

from entity_extraction.extractor import extract_fields, find_issuer, find_pay_period, group_lines
from entity_extraction.keywords import match_field, normalise
from entity_extraction.payslip import extract_document
from privacy import mask_tokens

SAMPLES = Path(__file__).resolve().parents[1] / "sample_documents"


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows, conf=0.93, amount_x=760, dy=0):
    """rows: (label, amount). dy shifts the amount column down (half-row offset on real photos)."""
    toks = []
    for i, (label, amount) in enumerate(rows):
        y = 100 + 40 * i
        x = 40
        for w in label.split():
            toks.append(Tok(w, conf, [x, y, 14 * len(w) + 4, 20]))
            x += 14 * len(w) + 14
        if amount:
            toks.append(Tok(amount, conf, [amount_x, y + dy, 12 * len(amount), 20]))
    return toks


@pytest.mark.parametrize("label,field", [
    ("A P I T", "apit"), ("E.T.F.CONTRIBUTION 3%", "etf"), ("E.P.F. EMPLOYEE CONTRIBUTION", "epf_employee"),
    ("E.P.F. EMPLOYER CONTRIBUTION", "epf_employer"), ("TOTAL DEDUCTION", "total_deductions"),
    ("Consolidated Salary", "basic_salary"), ("CLA Adjustment", "cola"),
    ("Total gross remuneration - LKR as per Pay Sheet", "gross_income"), ("Amount of tax deducted - LKR", "apit"),
])
def test_real_slip_labels(label, field):
    assert match_field(label)[0] == field


def test_not_fields():
    assert match_field("TOTAL FOR P.A.Y.E. TAX") is None            # the taxable total, not the tax withheld
    assert match_field("Fuel Allowance for Managers") is None       # one allowance is not the allowances total
    assert normalise("A P I T") == "apit"


def test_field_label_is_never_masked_as_a_name():
    toks = page([("SALARY", "71,925.00"), ("E.P.F. EMPLOYEE CONTRIBUTION", "14,010.48"),
                 ("W/W&OP CONTRIBUTION - AFTER 96", "3,596.25")])
    masked, rep = mask_tokens(toks)
    text = " ".join(t.text for t in masked)
    assert "EMPLOYEE" in text and text.count("CONTRIBUTION") == 2 and not rep.counts


def test_amount_without_separators_on_money_line_is_not_an_id():
    masked, rep = mask_tokens(page([("Total gross remuneration - LKR", "433951165")]))
    assert "433951165" in " ".join(t.text for t in masked)
    masked, rep = mask_tokens(page([("Account No", "1234567890")]))
    assert "1234567890" not in " ".join(t.text for t in masked)         # a real account number still is


def test_two_printed_rows_merged_into_one_line_are_split():
    # amount column printed half a row lower: 'Paye Tax' and 'Total Deductions' chain into one line
    toks = [Tok("Paye", .9, [40, 100, 60, 20]), Tok("Tax", .9, [110, 100, 40, 20]),
            Tok("18,357.00", .9, [760, 112, 110, 20]),
            Tok("Total", .9, [40, 124, 60, 20]), Tok("Deductions", .9, [110, 124, 120, 20])]
    lines = [ln.text for ln in group_lines(toks)]
    assert "Total Deductions" in lines and any(l.startswith("Paye Tax") for l in lines)


def deductions_slip(apit="4,357.00", total="22,367.48"):
    return page([("LKR", ""), ("SALARY", "71,925.00"), ("RENT ALLOWANCE", "14,385.00"),
                 ("COST OF LIVING ALLOWANCE", "69,336.36"), ("TOTAL EARNINGS", "155,646.36"),
                 ("E.P.F. EMPLOYEE CONTRIBUTION", "12,451.71"), ("CBEU FEE", "500.00"),
                 ("STAFF LOAN", "5,033.77"), ("STAMP DUTY", "25.00"), ("A P I T", apit),
                 ("TOTAL DEDUCTION", total), ("NET SALARY", "133,278.88")])


def test_deduction_lines_sum_verifies_apit():
    out, report = extract_document(deductions_slip(), "T")
    f = {x.field_name: x for x in out.extracted_fields}
    check = next(c for c in report["checks"] if c["rule"] == "deduction lines add up to total deductions")
    assert check["passed"]
    assert f["apit"].value == 4357.0 and not f["apit"].needs_review         # verified by the sum
    assert f["epf_employee"].value == 12451.71 and "issuer" not in f        # 'LKR' is not the employer


def test_deduction_sum_points_at_a_misread_line_but_verifies_nothing():
    out, report = extract_document(deductions_slip(apit="4,857.00", total="22,367.48"), "T")
    f = {x.field_name: x for x in out.extracted_fields}
    check = next(c for c in report["checks"] if c["rule"] == "deduction lines add up to total deductions")
    assert check["passed"] is True and "one digit misread" in check["explanation"]  # 4,857 -> 4,357 is the only fix
    assert f["apit"].needs_review                                     # the misread line itself is not 'verified'


def test_pay_period_from_glued_title_and_title_is_not_issuer():
    toks = page([("Pay Slip For The Month ofsuly2026", ""), ("Consolidated Salary", "179,045.79")])
    lines = group_lines(toks)
    pp = find_pay_period(lines)
    assert pp.value == "2026-07"
    assert find_issuer(lines) is None


@pytest.mark.skipif(not (SAMPLES / "SYN_payslip_clean.png").exists(), reason="synthetic sample missing")
def test_sideways_photo_is_turned_upright():
    from api.pipeline import run_any_orientation
    img = cv2.rotate(cv2.imread(str(SAMPLES / "SYN_payslip_clean.png")), cv2.ROTATE_90_COUNTERCLOCKWISE)
    out, report, upright = run_any_orientation(img, "SYN", lang="eng", engine="tesseract")
    assert out.document_type == "payslip" and report["orientation"]["rotated"] == 90
    assert upright.shape[0] > upright.shape[1] or upright.shape != img.shape
