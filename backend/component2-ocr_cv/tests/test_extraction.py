"""Why: extraction turns OCR text into tax data; these tests pin down label matching, amount
cleaning, layout-based value finding, the fiscal self-correction and the final JSON contract."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import pytest

from entity_extraction.amounts import parse_amount
from entity_extraction.confidence import field_confidence
from entity_extraction.extractor import extract_fields
from entity_extraction.fiscal import misreadings, validate_and_correct
from entity_extraction.keywords import match_field
from entity_extraction.payslip import extract_payslip, to_contract_json

REPO = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO / "contracts/schemas/ocr_output_schema.json").read_text(encoding="utf-8"))
GOOD = dict(basic_salary=120000.0, allowances=30000.0, gross_income=150000.0, epf_employee=12000.0,
            apit=3500.0, total_deductions=15500.0, net_amount=134500.0)


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list


def page(rows, conf=0.9, dy=60):
    toks = []
    for r, words in enumerate(rows):
        x = 50
        for w in words:
            toks.append(Tok(w, conf, [x, 100 + r * dy, len(w) * 15, 30]))
            x += len(w) * 15 + 20
    return toks


PAYSLIP = [["LANKA", "TEXTILES", "(PVT)", "LTD"], ["PAY", "SLIP", "-", "SEPTEMBER", "2026"],
           ["Basic", "Salary", "120,000.00"], ["Allowances", "30,000.00"], ["Gross", "Salary", "150,000.00"],
           ["EPF", "Employee", "8%", "12,000.00"], ["APIT", "3,500.00"], ["Total", "Deductions", "15,500.00"],
           ["Net", "Salary", "134,500.00"]]


# ---------- amounts
@pytest.mark.parametrize("text,value,corrected", [
    ("Rs. 150,000.00", 150000.0, False), ("30,000,00", 30000.0, True), ("l2,OOO.OO", 12000.0, True),
    ("134,500.00", 134500.0, False), ("3500", 3500.0, False), ("12.000.00", 12000.0, False)])
def test_parse_amount(text, value, corrected):
    p = parse_amount(text)
    assert p.value == value and p.corrected == corrected


@pytest.mark.parametrize("text", ["8%", "Basic", "", "--"])
def test_parse_amount_rejects_non_amounts(text):
    assert parse_amount(text) is None


# ---------- labels
@pytest.mark.parametrize("label,field", [
    ("Gross Salary", "gross_income"), ("Net Safary", "net_amount"), ("EPF Employer 12%", "epf_employer"),
    ("EPF Employee 8%", "epf_employee"), ("ETF 3%", "etf"), ("Total Deductions", "total_deductions"),
    ("APIT", "apit"), ("දළ වැටුප", "gross_income"), ("நிகர சம்பளம்", "net_amount")])
def test_match_field(label, field):
    assert match_field(label)[0] == field


@pytest.mark.parametrize("label", ["PAY SLIP - SEPTEMBER 2026", "LANKA TEXTILES (PVT) LTD", "Signature"])
def test_non_labels_are_not_matched(label):
    assert match_field(label) is None


# ---------- layout extraction
def test_extract_fields_same_line():
    f = extract_fields(page(PAYSLIP))
    assert f["gross_income"].value == 150000.0 and f["net_amount"].value == 134500.0
    assert f["pay_period"].value == "2026-09" and "LANKA" in f["issuer"].value
    assert "apit" in f and f["apit"].value == 3500.0


def test_value_below_label():
    toks = [Tok("Net", 0.9, [50, 100, 45, 30]), Tok("Salary", 0.9, [110, 100, 90, 30]),
            Tok("134,500.00", 0.9, [60, 150, 150, 30])]
    assert extract_fields(toks)["net_amount"].value == 134500.0


def test_year_is_not_an_amount():
    assert "apit" not in extract_fields(page([["PAY", "SLIP", "SEPTEMBER", "2026"]]))


# ---------- fiscal self-correction (own algorithm)
def test_consistent_payslip_is_unchanged():
    r = validate_and_correct(GOOD)
    assert not r.corrections and all(c.passed for c in r.checks)


def test_extra_digit_in_gross_is_corrected():
    r = validate_and_correct(dict(GOOD, gross_income=4150000.0))
    assert r.corrections == {"gross_income": (4150000.0, 150000.0)}


def test_lookalike_digit_in_net_is_corrected():
    r = validate_and_correct(dict(GOOD, net_amount=184500.0))
    assert r.corrections["net_amount"][1] == 134500.0


def test_weak_evidence_is_not_corrected():
    # only one SOFT rule breaks -> do not guess; the field is flagged instead
    r = validate_and_correct(dict(GOOD, apit=8500.0))
    assert not r.corrections


def test_other_deductions_do_not_trigger_false_correction():
    r = validate_and_correct(dict(GOOD, total_deductions=17500.0, net_amount=132500.0))   # e.g. a loan
    assert not r.corrections


def test_misreadings_include_dropped_digit():
    assert (150000.0, 1) in misreadings(4150000.0)


# ---------- confidence
def test_confidence_rules():
    assert field_confidence(0.9, verified=True, corrected=False, contradicted=False) >= 0.95
    assert field_confidence(0.99, verified=False, corrected=True, contradicted=False) <= 0.90
    assert field_confidence(0.9, verified=False, corrected=False, contradicted=True) < 0.9


# ---------- end to end
def test_extract_payslip_output_matches_contract():
    toks = page([r if r[0] != "Gross" else ["Gross", "Salary", "4150,000.00"] for r in PAYSLIP])
    out, report = extract_payslip(toks, doc_id="TEST_001")
    data = json.loads(to_contract_json(out))
    jsonschema.validate(data, SCHEMA)
    by = {f["field_name"]: f for f in data["extracted_fields"]}
    assert by["gross_income"]["value"] == 150000.0 and by["gross_income"]["needs_review"] is True
    assert by["net_amount"]["confidence"] >= 0.95 and by["net_amount"]["tax_entity"] == "NET_PAY"
    assert "basic_salary" not in by                       # helper fields stay internal
    assert report["corrections"]["gross_income"]["corrected_value"] == 150000.0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_real_ocr_on_clean_synthetic_payslip():
    import cv2
    from ocr_engines.tesseract_engine import run_tesseract
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "sample_documents" / "SYN_payslip_clean.png"))
    out, _ = extract_payslip(run_tesseract(img, lang="eng"), doc_id="SYN")
    by = {f.field_name: f.value for f in out.extracted_fields}
    assert by.get("gross_income") == 150000.0 and by.get("net_amount") == 134500.0
