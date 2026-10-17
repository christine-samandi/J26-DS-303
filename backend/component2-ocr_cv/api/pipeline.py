"""The complete Component 2 pipeline as one function, shared by the API and the scripts.

Why this module exists: the web API, the demo script and the benchmark must all run exactly
the same steps (deterministic: same input -> same output). Keeping the order in one place
means a change to the pipeline is made once and tested once.

    upload bytes -> image -> preprocessing -> Tesseract -> field extraction ->
    fiscal self-correction -> confidence -> contract JSON (+ explanation report)
Everything happens in memory; nothing is written to disk (NFR4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from api.schema import OCROutput
from entity_extraction.payslip import extract_payslip
from ocr_engines.tesseract_engine import run_tesseract
from preprocessing import preprocess

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_BYTES = 15 * 1024 * 1024


class UnsupportedFile(ValueError):
    """The upload is not an image or PDF we can read."""


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


def run_pipeline(img: np.ndarray, doc_id: str, lang: str = "sin+tam+eng", use_preprocessing: bool = True,
                 engine_name: str = "tesseract-5") -> Tuple[OCROutput, dict]:
    t0 = time.time()
    to_original = None
    if use_preprocessing:
        res = preprocess(img)
        tokens = run_tesseract(res.image, lang=lang)    # boxes in straightened coordinates
        to_original = res.to_original_bbox              # final boxes mapped back to the upload
    else:
        tokens = run_tesseract(img, lang=lang)
    out, report = extract_payslip(tokens, doc_id=doc_id, engine=engine_name,
                                  processing_ms=int((time.time() - t0) * 1000), to_original=to_original)
    report["preprocessing"] = res.steps_applied if use_preprocessing else []
    return out, report


def process_upload(data: bytes, filename: str, doc_id: Optional[str] = None, lang: str = "sin+tam+eng",
                   use_preprocessing: bool = True) -> PipelineResult:
    img, pages = decode_upload(data, filename)
    doc = doc_id or (filename.rsplit(".", 1)[0] if filename else "upload")
    out, report = run_pipeline(img, doc, lang=lang, use_preprocessing=use_preprocessing)
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
