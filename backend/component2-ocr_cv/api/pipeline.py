"""The complete Component 2 pipeline as one function, shared by the API and the scripts.

Why this module exists: the web API, the demo script and the benchmark must all run exactly
the same steps (deterministic: same input -> same output). Keeping the order in one place
means a change to the pipeline is made once and tested once.

    upload bytes -> image -> preprocessing -> OCR engine (tesseract | paddle | fusion of both) ->
    document classification -> field extraction -> fiscal self-correction -> confidence ->
    contract JSON (+ explanation report)
Everything happens in memory; nothing is written to disk (NFR4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from api.schema import OCROutput
from entity_extraction.payslip import extract_document
from ocr_engines.tesseract_engine import run_tesseract
from preprocessing import preprocess

Engine = Literal["tesseract", "paddle", "fusion"]
ENGINE_LABELS = {"tesseract": "tesseract-5", "paddle": "paddleocr-3", "fusion": "fusion(tesseract-5+paddleocr-3)"}

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_BYTES = 15 * 1024 * 1024


class UnsupportedFile(ValueError):
    """The upload is not an image or PDF we can read."""


class EngineUnavailable(RuntimeError):
    """The requested OCR engine is not installed on this machine (e.g. PaddleOCR)."""


@dataclass
class PipelineResult:
    output: OCROutput
    report: dict
    image: np.ndarray          # the (first) page as decoded, for drawing boxes on
    pages: int = 1


def decode_upload(data: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """Bytes -> BGR image. PDFs use their first page (multi-page support comes with
    multi-document consolidation, Gap 5). Raises UnsupportedFile with a readable reason."""
    if not data:
        raise UnsupportedFile("the file is empty")
    if len(data) > MAX_BYTES:
        raise UnsupportedFile(f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    name = filename.lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(data, dpi=300)
        except Exception as e:   # Poppler missing or a broken PDF
            raise UnsupportedFile(f"could not read the PDF ({type(e).__name__}); is Poppler installed?") from e
        if not pages:
            raise UnsupportedFile("the PDF has no pages")
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR), len(pages)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedFile("the file is not a readable image (JPG, PNG, TIFF, BMP, WEBP) or PDF")
    return img, 1


def read_tokens(img: np.ndarray, engine: str, lang: str) -> list:
    """Run ONE reading engine. lang is Tesseract-style (sin+tam+eng); Paddle maps it to its own model."""
    if engine == "tesseract":
        return run_tesseract(img, lang=lang)
    if engine == "paddle":
        try:
            from ocr_engines.paddle_engine import paddle_lang, run_paddle
            return run_paddle(img, lang=paddle_lang(lang))
        except ImportError as e:
            raise EngineUnavailable("PaddleOCR is not installed (pip install -r requirements-paddle.txt)") from e
    raise ValueError(f"unknown engine '{engine}'")


def analyse(img: np.ndarray, doc_id: str, lang: str = "sin+tam+eng", use_preprocessing: bool = True,
            engine: str = "tesseract") -> Tuple[OCROutput, dict, Sequence]:
    """The whole pipeline for one image. Returns (contract output, explanation report, tokens used
    for the text). The benchmark uses the tokens for CER/WER; the API only needs the first two."""
    from entity_extraction.fusion import extract_fused

    t0 = time.time()
    to_original = None
    page = img
    if use_preprocessing:
        res = preprocess(img)
        page = res.image                      # every engine reads the SAME straightened image,
        to_original = res.to_original_bbox    # so their boxes can be compared and fused

    if engine == "fusion":
        by_engine = {"tesseract": read_tokens(page, "tesseract", lang), "paddle": read_tokens(page, "paddle", lang)}
        out, report = extract_fused(by_engine, doc_id, to_original=to_original)
        tokens = by_engine["tesseract"]
    else:
        tokens = read_tokens(page, engine, lang)
        out, report = extract_document(tokens, doc_id=doc_id, engine=ENGINE_LABELS.get(engine, engine),
                                      to_original=to_original)
    out.processing_ms = int((time.time() - t0) * 1000)
    report["preprocessing"] = res.steps_applied if use_preprocessing else []
    report["engine"] = engine
    return out, report, tokens


def run_pipeline(img: np.ndarray, doc_id: str, lang: str = "sin+tam+eng", use_preprocessing: bool = True,
                 engine: str = "tesseract") -> Tuple[OCROutput, dict]:
    out, report, _ = analyse(img, doc_id, lang=lang, use_preprocessing=use_preprocessing, engine=engine)
    return out, report


def process_upload(data: bytes, filename: str, doc_id: Optional[str] = None, lang: str = "sin+tam+eng",
                   use_preprocessing: bool = True, engine: str = "tesseract") -> PipelineResult:
    img, pages = decode_upload(data, filename)
    doc = doc_id or (filename.rsplit(".", 1)[0] if filename else "upload")
    out, report = run_pipeline(img, doc, lang=lang, use_preprocessing=use_preprocessing, engine=engine)
    if pages > 1:
        report["note"] = f"PDF has {pages} pages; only page 1 was processed"
    return PipelineResult(output=out, report=report, image=img, pages=pages)


def draw_fields(img: np.ndarray, out: OCROutput) -> np.ndarray:
    """Draw each extracted field's box on the uploaded image: green = trusted, orange = needs
    review. Shows WHERE each value came from (source traceability, Gap 7)."""
    vis = img.copy()
    scale = max(0.5, min(img.shape[:2]) / 1200)
    for f in out.extracted_fields:
        x, y, w, h = f.bbox
        colour = (40, 160, 40) if not f.needs_review else (0, 140, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, max(2, int(3 * scale)))
        value = f"{f.value:,.2f}" if isinstance(f.value, float) else str(f.value)
        label = f"{f.field_name}: {value[:24]} ({f.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = max(th + 4, y - 6)
        cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty + 2), colour, -1)
        cv2.putText(vis, label, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def available_languages() -> List[str]:
    import pytesseract
    return sorted(pytesseract.get_languages(config=""))
