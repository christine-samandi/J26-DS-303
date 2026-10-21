"""Why: the default language is now 'auto'. Loading Sinhala and Tamil on an English page damaged
amounts on a real bank statement ('26 215 08' -> '08 08'). These tests fake the OCR passes to check
the decision: English pages use English only (one pass); Sinhala words switch on Sinhala, not Tamil."""
from dataclasses import dataclass

import numpy as np

import api.pipeline as pipeline


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = "1"


def fake_ocr(readings):
    calls = []

    def run(img, lang="eng", psm=6):
        calls.append(lang)
        return readings[lang]
    return run, calls


ENGLISH = [Tok(w, 0.92, [10 * i, 0, 8, 8]) for i, w in enumerate("Net Salary 121,180.82 Total Deductions".split())]
DOUBTFUL = [Tok(w, 0.3, [10 * i, 0, 8, 8]) for i, w in enumerate("Xq mlp 121,180.82 wq zz".split())]
SINHALA = [Tok(w, 0.9, [10 * i, 0, 8, 8]) for i, w in enumerate("ශුද්ධ වැටුප මූලික වැටුප දළ".split())]
STRAY_TAMIL = [Tok(w, 0.9, [10 * i, 0, 8, 8]) for i, w in enumerate("௦௦ ஒஒ".split())]   # 2 stray fragments


def test_english_page_uses_english_only(monkeypatch):
    run, calls = fake_ocr({"eng": ENGLISH})
    monkeypatch.setattr(pipeline, "run_tesseract", run)
    monkeypatch.setattr(pipeline, "available_languages", lambda: ["eng", "sin", "tam"])
    tokens, used, reason = pipeline.choose_language(np.zeros((10, 10), np.uint8))
    assert used == "eng" and calls == ["eng"] and "clean" in reason


def test_sinhala_page_switches_on_sinhala_but_not_tamil(monkeypatch):
    run, calls = fake_ocr({"eng": DOUBTFUL, "sin+tam+eng": SINHALA + ENGLISH, "sin+eng": SINHALA + ENGLISH})
    monkeypatch.setattr(pipeline, "run_tesseract", run)
    monkeypatch.setattr(pipeline, "available_languages", lambda: ["eng", "sin", "tam"])
    tokens, used, reason = pipeline.choose_language(np.zeros((10, 10), np.uint8))
    assert used == "sin+eng" and calls == ["eng", "sin+tam+eng", "sin+eng"] and "Sinhala" in reason


def test_doubtful_english_without_other_scripts_stays_english(monkeypatch):
    run, calls = fake_ocr({"eng": DOUBTFUL, "sin+tam+eng": ENGLISH})
    monkeypatch.setattr(pipeline, "run_tesseract", run)
    monkeypatch.setattr(pipeline, "available_languages", lambda: ["eng", "sin", "tam"])
    tokens, used, _ = pipeline.choose_language(np.zeros((10, 10), np.uint8))
    assert used == "eng" and tokens is DOUBTFUL


def test_missing_language_packs_fall_back_to_english(monkeypatch):
    run, calls = fake_ocr({"eng": DOUBTFUL})
    monkeypatch.setattr(pipeline, "run_tesseract", run)
    monkeypatch.setattr(pipeline, "available_languages", lambda: ["eng"])
    assert pipeline.choose_language(np.zeros((10, 10), np.uint8))[1] == "eng"


def test_stray_characters_do_not_switch_on_a_script(monkeypatch):
    # Real statement: table rules produced a few Tamil-looking fragments in the 3-language reading
    run, calls = fake_ocr({"eng": DOUBTFUL, "sin+tam+eng": ENGLISH * 3 + STRAY_TAMIL})
    monkeypatch.setattr(pipeline, "run_tesseract", run)
    monkeypatch.setattr(pipeline, "available_languages", lambda: ["eng", "sin", "tam"])
    tokens, used, reason = pipeline.choose_language(np.zeros((10, 10), np.uint8))
    assert used == "eng" and tokens is DOUBTFUL and "English kept" in reason
