"""FastAPI service for Component 2 - the single entry point other components call (FR10).

Why this module exists: Component 1 (the agentic orchestrator) treats OCR as one tool. This
service exposes the whole pipeline behind one HTTP call that returns the shared JSON
contract, so the other components never depend on Python internals.

Run (from backend/component2-ocr_cv, with the virtual environment active):
    uvicorn api.main:app --reload
Then open http://localhost:8000/docs to try it in the browser.

Privacy (NFR4): uploads are processed in memory and never written to disk or logged.
"""
from __future__ import annotations

from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from api.pipeline import (Engine, EngineUnavailable, UnsupportedFile, available_languages, draw_fields,
                          process_upload)
from api.schema import OCROutput

app = FastAPI(
    title="Component 2 - OCR & Computer Vision",
    version="0.1.0",
    description="Sri Lankan financial documents (photo, scan or PDF) -> structured, tax-ready JSON "
                "with per-field confidence and bounding boxes. Part of J26-DS-303.",
)

LANG_HELP = ("auto (default: detects which of English/Sinhala/Tamil are on the page) or Tesseract "
             "languages joined with '+', e.g. eng, sin+eng, sin+tam+eng")


def _check_lang(lang: str) -> None:
    if lang == "auto":
        return
    try:
        installed = set(available_languages())
    except Exception as e:
        raise HTTPException(503, f"Tesseract is not available on this machine: {type(e).__name__}") from e
    missing = [code for code in lang.split("+") if code not in installed]
    if missing:
        raise HTTPException(400, f"Tesseract language(s) not installed: {', '.join(missing)}. "
                                 f"Installed: {', '.join(sorted(installed))}")


ENGINE_HELP = ("tesseract (default, reads Sinhala/Tamil/English) | paddle (English/Tamil) | "
               "fusion (both engines; fiscal rules decide disagreements)")


def _run(file: UploadFile, lang: str, preprocess: bool, doc_id: Optional[str], engine: str = "tesseract"):
    _check_lang(lang)
    data = file.file.read()
    try:
        return process_upload(data, file.filename or "", doc_id=doc_id, lang=lang, use_preprocessing=preprocess,
                              engine=engine)
    except UnsupportedFile as e:
        raise HTTPException(415, str(e)) from e
    except EngineUnavailable as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health() -> dict:
    """Is the service up, and which OCR languages can it read?"""
    try:
        import pytesseract
        return {"status": "ok", "tesseract": str(pytesseract.get_tesseract_version()),
                "languages": available_languages()}
    except Exception as e:
        return {"status": "degraded", "error": f"Tesseract not available: {type(e).__name__}"}


@app.post("/extract", response_model=OCROutput, response_model_exclude_none=True)
def extract(file: UploadFile = File(..., description="Photo, scan or PDF of a payslip"),
            lang: str = Query("auto", description=LANG_HELP),
            preprocess: bool = Query(True, description="Turn off only to compare with raw OCR"),
            doc_id: Optional[str] = Query(None, description="Optional id; defaults to the file name"),
            engine: Engine = Query("tesseract", description=ENGINE_HELP)) -> OCROutput:
    """Upload one document and receive the contract JSON (contracts/schemas/ocr_output_schema.json)."""
    return _run(file, lang, preprocess, doc_id, engine).output


@app.post("/extract/explain")
def extract_explain(file: UploadFile = File(...), lang: str = Query("auto", description=LANG_HELP),
                    preprocess: bool = Query(True), doc_id: Optional[str] = Query(None),
                    engine: Engine = Query("tesseract", description=ENGINE_HELP)) -> dict:
    """Same as /extract, plus the explanation: every arithmetic check, every correction, and the
    preprocessing steps applied (and, for fusion, which engine won each field and why).
    For reviewers and debugging, not part of the shared contract."""
    r = _run(file, lang, preprocess, doc_id, engine)
    return {"result": r.output.model_dump(exclude_none=True), "explanation": r.report}


@app.post("/extract/visual", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def extract_visual(file: UploadFile = File(...), lang: str = Query("auto", description=LANG_HELP),
                   preprocess: bool = Query(True), doc_id: Optional[str] = Query(None),
                   engine: Engine = Query("tesseract", description=ENGINE_HELP)) -> Response:
    """Returns the uploaded image with a box on every extracted field
    (green = trusted, orange = needs review): shows where each value came from."""
    r = _run(file, lang, preprocess, doc_id, engine)
    ok, png = cv2.imencode(".png", draw_fields(r.image, r.output))
    if not ok:
        raise HTTPException(500, "could not draw the result image")
    return Response(content=png.tobytes(), media_type="image/png",
                    headers={"X-Overall-Confidence": str(r.output.overall_confidence),
                             "X-Needs-Review": str(r.output.needs_review).lower()})
