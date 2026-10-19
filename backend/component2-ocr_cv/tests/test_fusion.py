"""Why: fusion is an own method; these tests pin down its rules - arithmetic decides disputed
amounts, agreement raises confidence, unresolved disagreement is flagged - and show the case that
motivates it: each engine misreads a DIFFERENT field, and neither can be fixed on its own."""
import importlib.util
import shutil
from dataclasses import dataclass

import pytest

from entity_extraction.extractor import Candidate
from entity_extraction.fusion import AGREE_BOOST, DISAGREE_FACTOR, extract_fused, fuse_candidates


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


LINES = [("LANKA TEXTILES (PVT) LTD", None), ("PAY SLIP SEPTEMBER 2026", None),
         ("Basic Salary", "120,000.00"), ("Allowances", "30,000.00"), ("Gross Salary", "150,000.00"),
         ("EPF Employee 8%", "12,000.00"), ("APIT", "3,500.00"), ("Total Deductions", "15,500.00"),
         ("Net Salary", "134,500.00")]


def page(overrides=None, conf=0.9):
    """Tokens of a clean payslip; overrides = {label: misread amount}."""
    overrides = overrides or {}
    toks = []
    for i, (label, amount) in enumerate(LINES):
        y = 100 + 40 * i
        for j, w in enumerate(label.split()):
            toks.append(Tok(w, conf, [20 + 90 * j, y, 80, 20], f"l{i}"))
        if amount:
            toks.append(Tok(overrides.get(label, amount), conf, [600, y, 120, 20], f"l{i}"))
    return toks


def cand(field, value, conf):
    return Candidate(field, value, str(value), ocr_confidence=conf, bbox=[0, 0, 10, 10])


def test_arithmetic_decides_disputed_amount():
    t = {"gross_income": cand("gross_income", 150000.0, 0.9), "total_deductions": cand("total_deductions", 15500.0, 0.9),
         "net_amount": cand("net_amount", 184500.0, 0.97)}            # confident but wrong
    p = {"gross_income": cand("gross_income", 150000.0, 0.9), "total_deductions": cand("total_deductions", 15500.0, 0.9),
         "net_amount": cand("net_amount", 134500.0, 0.80)}            # less confident but right
    fused, decisions = fuse_candidates({"tesseract": t, "paddle": p})
    assert fused["net_amount"].value == 134500.0
    d = {x.field: x for x in decisions}
    assert d["net_amount"].reason == "arithmetic" and d["net_amount"].chosen == "paddle"
    assert d["net_amount"].options == {"tesseract": 184500.0, "paddle": 134500.0}


def test_agreement_raises_confidence():
    fused, decisions = fuse_candidates({"tesseract": {"apit": cand("apit", 3500.0, 0.8)},
                                        "paddle": {"apit": cand("apit", 3500.0, 0.7)}})
    assert fused["apit"].ocr_confidence == pytest.approx(1 - 0.2 * AGREE_BOOST)
    assert decisions[0].reason == "agreed" and decisions[0].chosen == "tesseract+paddle"


def test_unresolved_disagreement_is_penalised():
    fused, decisions = fuse_candidates({"tesseract": {"issuer": cand("issuer", "LANKA TEXTILES", 0.9)},
                                        "paddle": {"issuer": cand("issuer", "LANKA TEXT1LES", 0.6)}})
    assert fused["issuer"].value == "LANKA TEXTILES"
    assert fused["issuer"].ocr_confidence == pytest.approx(0.9 * DISAGREE_FACTOR)
    assert decisions[0].reason == "higher confidence"


def test_single_engine_field_is_kept():
    fused, decisions = fuse_candidates({"tesseract": {}, "paddle": {"etf": cand("etf", 4500.0, 0.88)}})
    assert fused["etf"].ocr_confidence == 0.88 and decisions[0].reason == "only paddle"


def test_fusion_fixes_what_neither_engine_can():
    # Tesseract dropped a digit in net, Paddle dropped one in gross: single-digit self-correction
    # cannot ADD a digit back, so each engine alone leaves a wrong value.
    tess = page({"Net Salary": "13,450.00"})
    padd = page({"Gross Salary": "15,000.00"})
    out, report = extract_fused({"tesseract": tess, "paddle": padd}, doc_id="FUS")
    by = {f.field_name: f.value for f in out.extracted_fields}
    assert out.document_type == "payslip"
    assert by["gross_income"] == 150000.0 and by["net_amount"] == 134500.0
    reasons = {d["field"]: d["reason"] for d in report["fusion"]["decisions"]}
    assert reasons["gross_income"] == "arithmetic" and reasons["net_amount"] == "arithmetic"
    assert reasons["apit"] == "agreed"
    net_rule = [c for c in report["checks"] if c["hard"]]
    assert net_rule and all(c["passed"] for c in net_rule)


def test_non_payslip_is_not_fused():
    letter = [Tok(w, 0.9, [20 + 60 * i, 100, 50, 20], "l0") for i, w in enumerate(
        "Dear Sir I am writing to request leave for two days Yours faithfully".split())]
    out, report = extract_fused({"tesseract": letter, "paddle": letter}, doc_id="L")
    assert out.document_type == "unknown" and out.extracted_fields == []
    assert "no field fusion" in report["fusion"]["note"]


@pytest.mark.skipif(shutil.which("tesseract") is None or importlib.util.find_spec("paddleocr") is None,
                    reason="needs Tesseract and PaddleOCR")
def test_fusion_engine_through_pipeline():
    from pathlib import Path
    import cv2
    from api.pipeline import analyse
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "sample_documents" / "SYN_payslip_dark_skew.jpg"))
    out, report, _ = analyse(img, "SYN", lang="eng", engine="fusion")
    by = {f.field_name: f.value for f in out.extracted_fields}
    assert out.engine.startswith("fusion") and by.get("net_amount") == 134500.0
    assert report["fusion"]["decisions"]
