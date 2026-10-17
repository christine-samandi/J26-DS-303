"""End-to-end payslip extraction: OCR tokens -> schema-valid JSON (Objectives 2.3 and 2.4).

Why this module exists: it joins the steps (label/value finding, fiscal self-correction,
confidence, ontology) into one call that returns exactly the shared output contract, plus a
human-readable report of every check and correction (the explanation behind each value).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from api.schema import ExtractedField, OCROutput
from entity_extraction.confidence import field_confidence, needs_review
from entity_extraction.extractor import extract_fields
from entity_extraction.fiscal import FiscalReport, validate_and_correct
from entity_extraction.keywords import HELPER_FIELDS

ONTOLOGY = json.loads((Path(__file__).parent / "ontology" / "tax_ontology.json").read_text(encoding="utf-8"))["fields"]


def detect_languages(tokens: Sequence) -> List[str]:
    text = "".join(t.text for t in tokens)
    langs = []
    if any("඀" <= ch <= "෿" for ch in text):
        langs.append("si")
    if any("஀" <= ch <= "௿" for ch in text):
        langs.append("ta")
    if any(ch.isascii() and ch.isalpha() for ch in text):
        langs.append("en")
    return langs or ["en"]


def extract_payslip(tokens: Sequence, doc_id: str, engine: str = "tesseract-5",
                    processing_ms: Optional[int] = None,
                    to_original: Optional[Callable[[List[int]], List[int]]] = None) -> Tuple[OCROutput, dict]:
    """tokens must be in the coordinates of the image OCR ran on (the straightened image), because
    lines are rebuilt from box positions and a tilted photo puts a label and its amount at
    different heights. `to_original` maps each final bbox back to the uploaded photo."""
    cands = extract_fields(tokens)
    amounts = {f: c.value for f, c in cands.items() if isinstance(c.value, float)}
    fiscal: FiscalReport = validate_and_correct(amounts)

    out_fields: List[ExtractedField] = []
    for f, c in cands.items():
        if f in HELPER_FIELDS:
            continue
        value = fiscal.values.get(f, c.value)
        conf = field_confidence(
            c.ocr_confidence,
            verified=fiscal.verified(f) and f not in fiscal.corrections,
            corrected=f in fiscal.corrections,
            contradicted=fiscal.contradicted(f),
            cleaned=c.cleaned,
            match_score=c.match_score,
        )
        out_fields.append(ExtractedField(
            field_name=f, value=value, raw_text=c.raw_text, confidence=conf,
            bbox=[int(x) for x in (to_original(c.bbox) if to_original else c.bbox)], tax_entity=ONTOLOGY.get(f, {}).get("tax_entity"),
            needs_review=needs_review(conf)))

    overall = round(sum(x.confidence for x in out_fields) / len(out_fields), 3) if out_fields else 0.0
    result = OCROutput(doc_id=doc_id, document_type="payslip", languages=detect_languages(tokens),
                       extracted_fields=out_fields, overall_confidence=overall,
                       needs_review=(not out_fields) or any(x.needs_review for x in out_fields),
                       engine=engine, processing_ms=processing_ms)
    report = {
        "checks": [{"rule": c.rule, "hard": c.hard, "passed": c.passed, "explanation": c.explanation}
                   for c in fiscal.checks],
        "corrections": {f: {"ocr_value": a, "corrected_value": b} for f, (a, b) in fiscal.corrections.items()},
        "notes": {f: c.notes for f, c in cands.items() if c.notes},
        "helper_values": {f: cands[f].value for f in HELPER_FIELDS if f in cands},
    }
    return result, report


def to_contract_json(result: OCROutput, indent: Optional[int] = 2) -> str:
    """Serialise for other components. Optional fields that are empty are LEFT OUT instead of
    written as null, because the shared JSON schema types them as number/string, not null."""
    return result.model_dump_json(indent=indent, exclude_none=True)
