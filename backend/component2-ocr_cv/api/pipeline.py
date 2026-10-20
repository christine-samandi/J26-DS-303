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

import re
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


AUTO = "auto"
LOW_CONF = 0.5          # a word below this OCR confidence is "doubtful"
CLEAN_SHARE = 0.15      # English pass with at most 15% doubtful words = an English document
SCRIPT_MIN_WORDS = 5    # a script is "on the page" only with this many confident words ...
SCRIPT_MIN_SHARE = 0.10  # ... making up at least this share of all words (stray characters from
                         # table rules or logos must not switch on Tamil/Sinhala)
SCRIPTS = {"sin": ("\u0D80", "\u0DFF", "Sinhala"), "tam": ("\u0B80", "\u0BFF", "Tamil")}


def script_words(tokens, lo: str, hi: str) -> int:
    """Confident words (2+ characters) written mostly in the script range [lo, hi]."""
    n = 0
    for t in tokens:
        letters = [ch for ch in t.text if ch.isalpha() or lo <= ch <= hi]
        if t.confidence >= 0.6 and len(t.text) >= 2 and letters and \
                sum(lo <= ch <= hi for ch in letters) / len(letters) >= 0.5:
            n += 1
    return n


def choose_language(img: np.ndarray) -> Tuple[list, str, str]:
    """Script-aware language selection (lang='auto').

    Why: loading Sinhala and Tamil models on an English page makes Tesseract try Sinhala/Tamil
    shapes on English text (a real statement's '26 215 08' became '08 08' and a false 'ta' language
    appeared). Loading only Sinhala on an English+Sinhala payslip avoids Tamil confusion, and so on.
      1. Read in English. If few words are doubtful, it is an English document: done (fast path).
      2. Otherwise read with every installed script and count confident words per script.
      3. Re-read with ONLY the scripts that are really on the page, plus English.
    Returns (tokens, tesseract language used, reason)."""
    eng = run_tesseract(img, lang="eng")
    words = [t for t in eng if len(t.text) >= 2]
    doubtful = sum(t.confidence < LOW_CONF for t in words) / len(words) if words else 1.0
    if words and doubtful <= CLEAN_SHARE:
        return eng, "eng", f"English reading is clean ({doubtful:.0%} doubtful words)"
    extra = [c for c in ("sin", "tam") if c in available_languages()]
    if not extra:
        return eng, "eng", "Sinhala/Tamil language packs are not installed"
    all_lang = "+".join(extra + ["eng"])
    mixed = run_tesseract(img, lang=all_lang)
    total = max(1, sum(len(t.text) >= 2 for t in mixed))
    counts = {c: script_words(mixed, SCRIPTS[c][0], SCRIPTS[c][1]) for c in extra}
    chosen = [c for c in extra if counts[c] >= SCRIPT_MIN_WORDS and counts[c] / total >= SCRIPT_MIN_SHARE]
    if not chosen:
        found = ", ".join(f"{SCRIPTS[c][2]} {counts[c]}" for c in extra)
        return eng, "eng", f"no real Sinhala/Tamil text ({found} confident words of {total}); English kept"
    used = "+".join(chosen + ["eng"])
    reason = ", ".join(f"{SCRIPTS[c][2]}: {counts[c]} confident words of {total}" for c in chosen)
    return (mixed if used == all_lang else run_tesseract(img, lang=used)), used, reason


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


# --- Self-checked reading (own method) -------------------------------------------------------
# Why: the binarised page (adaptive threshold + Tesseract 'single block' mode) is best for most
# photos (PS_001: 6/6 vs 3/6), but on some slips it erases thin decimal points ('179,045.79' ->
# '179,045 79') and drops boxed header rows entirely (so the NIC/name were never read, and never
# masked). No single setting wins on every document, so the pipeline checks its own reading:
# when the first reading looks broken it reads again from the greyscale page with layout analysis
# (psm 4) and keeps whichever reading the tax arithmetic and amount format support better.
MONEY_TOKEN = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d{2})?$|^\d+\.\d{2}$")
GOOD_DECIMAL_SHARE = 0.5            # under half the amounts with '.dd' cents = decimal points lost
ALT_PSM = 4                         # single column of variable-size text: keeps boxed header rows


def reading_quality(tokens: Sequence, report: dict, n_fields: int) -> Tuple[int, int, float]:
    """(hard checks passed, fields found, share of amounts with decimal cents): compared as a tuple."""
    hard = sum(1 for c in report.get("checks", []) if c.get("hard") and c.get("passed"))
    money = [t.text.strip(".,:;") for t in tokens if MONEY_TOKEN.match(t.text.strip(".,:;"))]
    share = sum("." in m for m in money) / len(money) if money else 0.0
    return hard, n_fields, round(share, 2)


def looks_broken(quality: Tuple[int, int, float]) -> bool:
    hard, _, share = quality
    return hard == 0 or share < GOOD_DECIMAL_SHARE


def analyse(img: np.ndarray, doc_id: str, lang: str = "auto", use_preprocessing: bool = True,
            engine: str = "tesseract", mask_pii: bool = True) -> Tuple[OCROutput, dict, Sequence]:
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

    language = {"requested": lang, "used": lang, "reason": "chosen by the caller"}
    tess_tokens = None
    if lang == AUTO and engine in ("tesseract", "fusion"):
        tess_tokens, used, reason = choose_language(page)
        language = {"requested": AUTO, "used": used, "reason": reason}
        lang = used
    elif lang == AUTO:
        lang = "eng"
        language = {"requested": AUTO, "used": "en (PaddleOCR)", "reason": "PaddleOCR reads English/Tamil models only"}

    from privacy import mask_tokens
    pii_counts, pii_boxes = {}, []

    def guard(toks):
        """Mask personal identifiers right after OCR, before any other step sees them (NFR4)."""
        if not mask_pii:
            return toks
        masked, rep = mask_tokens(toks)
        for k, v in rep.counts.items():
            pii_counts[k] = max(pii_counts.get(k, 0), v)
        pii_boxes.extend(rep.regions)
        return masked

    if engine == "fusion":
        by_engine = {"tesseract": guard(tess_tokens if tess_tokens is not None else read_tokens(page, "tesseract", lang)),
                     "paddle": guard(read_tokens(page, "paddle", lang))}
        out, report = extract_fused(by_engine, doc_id, to_original=to_original)
        tokens = by_engine["tesseract"]
    else:
        tokens = guard(tess_tokens if tess_tokens is not None else read_tokens(page, engine, lang))
        out, report = extract_document(tokens, doc_id=doc_id, engine=ENGINE_LABELS.get(engine, engine),
                                      to_original=to_original)
        if engine == "tesseract" and use_preprocessing and "threshold" in res.steps_applied:
            first = reading_quality(tokens, report, len(out.extracted_fields))
            reading = {"used": "binarised page", "first": first, "second": None,
                       "reason": "first reading passed the self-check"}
            if looks_broken(first):
                # Same steps minus threshold: threshold does not move pixels, so boxes still map back.
                grey = preprocess(img, steps=[st for st in res.steps_applied if st != "threshold"]).image
                alt_tokens = guard(run_tesseract(grey, lang=lang, psm=ALT_PSM))
                alt_out, alt_report = extract_document(alt_tokens, doc_id=doc_id,
                                                       engine=ENGINE_LABELS.get(engine, engine),
                                                       to_original=to_original)
                second = reading_quality(alt_tokens, alt_report, len(alt_out.extracted_fields))
                reading.update(second=second)
                if second > first:
                    tokens, out, report = alt_tokens, alt_out, alt_report
                    reading.update(used="greyscale page (layout mode)",
                                   reason="second reading passes more tax checks / keeps decimal points")
                else:
                    reading["reason"] = "second reading was not better; first kept"
            report["reading"] = reading
    out.processing_ms = int((time.time() - t0) * 1000)
    report["preprocessing"] = res.steps_applied if use_preprocessing else []
    report["engine"] = engine
    report["ocr_language"] = language
    report["privacy"] = {"enabled": mask_pii, "masked": dict(sorted(pii_counts.items())),
                         "boxes": [[int(v) for v in (to_original(b) if to_original else b)] for b in pii_boxes],
                         "note": "personal identifiers masked right after OCR; values are never returned"}
    return out, report, tokens


def run_pipeline(img: np.ndarray, doc_id: str, lang: str = "auto", use_preprocessing: bool = True,
                 engine: str = "tesseract") -> Tuple[OCROutput, dict]:
    out, report, _ = analyse(img, doc_id, lang=lang, use_preprocessing=use_preprocessing, engine=engine)
    return out, report


def process_upload(data: bytes, filename: str, doc_id: Optional[str] = None, lang: str = "auto",
                   use_preprocessing: bool = True, engine: str = "tesseract") -> PipelineResult:
    img, pages = decode_upload(data, filename)
    doc = doc_id or (filename.rsplit(".", 1)[0] if filename else "upload")
    out, report = run_pipeline(img, doc, lang=lang, use_preprocessing=use_preprocessing, engine=engine)
    if pages > 1:
        report["note"] = f"PDF has {pages} pages; only page 1 was processed"
    return PipelineResult(output=out, report=report, image=img, pages=pages)


def blackout(img: np.ndarray, boxes) -> np.ndarray:
    """Cover masked personal data on the image (solid black boxes, a little padding)."""
    vis = img.copy()
    for x, y, w, h in boxes:
        pad = max(2, h // 5)
        cv2.rectangle(vis, (max(0, x - pad), max(0, y - pad)), (x + w + pad, y + h + pad), (0, 0, 0), -1)
    return vis


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
