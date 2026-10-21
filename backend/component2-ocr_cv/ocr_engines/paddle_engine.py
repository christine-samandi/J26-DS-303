"""PaddleOCR wrapper returning the shared Token format (text, confidence 0-1, bbox, line_id).

Why this module exists:
- A second, independently trained reader. Tesseract (LSTM) and PaddleOCR (deep detection +
  recognition) make DIFFERENT mistakes, especially on phone photos; comparing them on the same
  documents is part of the evaluation, and combining them is the basis for multi-engine fusion.
- PaddleOCR has no Sinhala model, so it covers English and Tamil only; Sinhala text stays with
  Tesseract (the Engine Dispatcher's job, decided per language).

Adaptation to the shared format:
- Paddle returns one box per text phrase ("Basic Salary 120,000.00"). Extraction expects words,
  so each phrase is split into words with boxes proportional to character count, all carrying the
  phrase's recognition score.
- line_id is assigned by grouping boxes on the same visual row, as for every other engine.
- The Paddle model is loaded once per language (it takes seconds) and reused.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

# Speed (Paddle on CPU took ~5 minutes for one 4000-px phone photo):
#  - skip PaddleX's network check of the model hosters on every model load,
#  - detect text on a copy whose longest side is at most DET_MAX_SIDE pixels (boxes come back in the
#    original coordinates), and
#  - use the lighter mobile text detector; recognition keeps the language's default model.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
DET_MAX_SIDE = int(os.environ.get("C2_PADDLE_DET_MAX_SIDE", "1600"))
DET_MODEL = os.environ.get("C2_PADDLE_DET_MODEL", "PP-OCRv5_mobile_det")
# Lighter text readers per language (the default 'medium' reader dominated CPU time: ~100 text lines
# per payslip). Set C2_PADDLE_FAST=0 to go back to Paddle's default models.
REC_MODEL = {"en": "en_PP-OCRv5_mobile_rec", "ta": "ta_PP-OCRv5_mobile_rec"}
FAST = os.environ.get("C2_PADDLE_FAST", "1") != "0"

import numpy as np

from ocr_engines.tesseract_engine import Token

TESS_TO_PADDLE = {"eng": "en", "tam": "ta"}      # no Sinhala model in PaddleOCR


def paddle_lang(tesseract_lang: str) -> str:
    """Paddle runs ONE recognition model per call. English is preferred whenever it is requested:
    amounts, labels and most payslip/bill text are printed in English, and Tesseract still covers
    Sinhala and Tamil words in fusion. The Tamil model is used only for Tamil-only documents."""
    codes = tesseract_lang.split("+")
    return "ta" if ("tam" in codes and "eng" not in codes) else "en"


def split_phrase(text: str, score: float, box: Sequence[float]) -> List[Token]:
    """Split one recognised phrase into word tokens; box = [x1, y1, x2, y2]."""
    words = text.split()
    if not words:
        return []
    x1, y1, x2, y2 = [float(v) for v in box]
    width, total = max(x2 - x1, 1.0), len(text)
    tokens, pos = [], 0
    for w in words:
        start = text.index(w, pos)
        end = start + len(w)
        pos = end
        wx1 = x1 + width * start / total
        wx2 = x1 + width * end / total
        tokens.append(Token(text=w, confidence=round(float(score), 4),
                            bbox=[int(round(wx1)), int(round(y1)), max(1, int(round(wx2 - wx1))), max(1, int(round(y2 - y1)))],
                            line_id=""))
    return tokens


def assign_line_ids(tokens: List[Token]) -> List[Token]:
    """Give tokens on the same visual row the same line_id, in reading order (top to bottom, left to right)."""
    from entity_extraction.extractor import group_lines
    out: List[Token] = []
    for i, line in enumerate(sorted(group_lines(tokens), key=lambda ln: min(t.bbox[1] for t in ln.tokens))):
        for t in line.tokens:
            t.line_id = f"p_{i}"
            out.append(t)
    return out


@lru_cache(maxsize=2)
def _model(lang: str):
    from paddleocr import PaddleOCR   # imported lazily: the rest of the pipeline works without Paddle
    # enable_mkldnn=False: Paddle 3.x on Windows CPUs crashes inside oneDNN with the PP-OCRv6 models
    # ("ConvertPirAttribute2RuntimeAttribute not support"); plain CPU inference is slower but stable.
    base = dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, enable_mkldnn=False)
    fast = dict(text_det_limit_side_len=DET_MAX_SIDE, text_det_limit_type="max")
    light = dict(fast, text_detection_model_name=DET_MODEL)
    options = [dict(light, text_recognition_model_name=REC_MODEL[lang])] if lang in REC_MODEL else []
    options += [light, fast, {}] if FAST else [{}]
    for extra in options:                                                       # fall back if a setting
        try:                                                                    # is not supported
            return PaddleOCR(**base, **extra)
        except Exception:  # noqa: BLE001  any load error: try the next, simpler setting
            continue
    return PaddleOCR(**base)


def warm_up(lang: str = "en") -> bool:
    """Load the Paddle model once (e.g. when the API starts), so the first upload is not slow."""
    try:
        _model(lang)
        return True
    except Exception:
        return False


def _field(result, key):
    try:
        return result[key]
    except (KeyError, TypeError):
        return result.json["res"][key]


def run_paddle(image: np.ndarray, lang: str = "en") -> List[Token]:
    """image: numpy array (BGR or gray). Our preprocessing already deskews, so Paddle's own
    orientation/unwarping models are switched off (faster, and boxes stay in our coordinates)."""
    if image.ndim == 2:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tokens: List[Token] = []
    for page in _model(lang).predict(image):
        texts, scores, boxes = _field(page, "rec_texts"), _field(page, "rec_scores"), _field(page, "rec_boxes")
        for text, score, box in zip(texts, scores, boxes):
            tokens += split_phrase(str(text), float(score), list(box))
    return assign_line_ids(tokens)
