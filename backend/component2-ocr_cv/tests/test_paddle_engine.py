"""Why: PaddleOCR must hand the pipeline the SAME token format as Tesseract (words, 0-1 confidence,
[x, y, w, h] boxes, line ids), otherwise extraction and benchmarking silently get worse.
The format tests run everywhere; the real read is skipped when Paddle is not installed."""
import importlib.util
from pathlib import Path

import cv2
import pytest

from ocr_engines.paddle_engine import assign_line_ids, paddle_lang, split_phrase

COMPONENT = Path(__file__).resolve().parents[1]


def test_language_mapping_has_no_sinhala():
    assert paddle_lang("sin+tam+eng") == "ta"
    assert paddle_lang("sin+eng") == "en"           # Sinhala stays with Tesseract
    assert paddle_lang("eng") == "en"


def test_phrase_is_split_into_word_tokens():
    toks = split_phrase("Basic Salary 120,000.00", 0.97, [100, 50, 330, 70])
    assert [t.text for t in toks] == ["Basic", "Salary", "120,000.00"]
    assert all(t.confidence == 0.97 and t.bbox[1] == 50 and t.bbox[3] == 20 for t in toks)
    assert toks[0].bbox[0] == 100 and toks[-1].bbox[0] + toks[-1].bbox[2] == 330   # spans the phrase box
    assert toks[0].bbox[0] < toks[1].bbox[0] < toks[2].bbox[0]


def test_line_ids_follow_rows():
    toks = (split_phrase("Net Salary", 0.9, [10, 300, 120, 320]) + split_phrase("134,500.00", 0.9, [400, 302, 500, 321])
            + split_phrase("Gross Salary 150,000.00", 0.9, [10, 100, 300, 120]))
    toks = assign_line_ids(toks)
    by = {t.text: t.line_id for t in toks}
    assert by["Net"] == by["134,500.00"] != by["Gross"]
    assert by["Gross"] == "p_0"                        # top row first


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="PaddleOCR not installed")
def test_paddle_reads_synthetic_payslip():
    from ocr_engines.paddle_engine import run_paddle
    from entity_extraction.payslip import extract_document
    img = cv2.imread(str(COMPONENT / "sample_documents" / "SYN_payslip_clean.png"))
    tokens = run_paddle(img, lang="en")
    assert tokens and all(0 <= t.confidence <= 1 and len(t.bbox) == 4 and t.line_id for t in tokens)
    out, _ = extract_document(tokens, doc_id="SYN", engine="paddleocr-3")
    by = {f.field_name: f.value for f in out.extracted_fields}
    assert out.document_type == "payslip" and by.get("net_amount") == 134500.0
