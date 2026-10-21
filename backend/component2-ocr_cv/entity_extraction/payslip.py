"""End-to-end payslip extraction: OCR tokens -> schema-valid JSON (Objectives 2.3 and 2.4).

Why this module exists: it joins the steps (label/value finding, fiscal self-correction,
confidence, ontology) into one call that returns exactly the shared output contract, plus a
human-readable report of every check and correction (the explanation behind each value).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from api.schema import ExtractedField, OCROutput
from entity_extraction.confidence import field_confidence, needs_review
from entity_extraction.alignment import extract_aligned
from entity_extraction.fiscal import FiscalReport, validate_and_correct
from entity_extraction.keywords import HELPER_FIELDS

ONTOLOGY = json.loads((Path(__file__).parent / "ontology" / "tax_ontology.json").read_text(encoding="utf-8"))["fields"]


def _script_share(word: str, lo: str, hi: str) -> float:
    letters = [ch for ch in word if ch.isalpha() or lo <= ch <= hi]
    return sum(lo <= ch <= hi for ch in letters) / len(letters) if letters else 0.0


def detect_languages(tokens: Sequence, min_words: int = 3, min_conf: float = 0.6) -> List[str]:
    """A language is reported only with real evidence: at least `min_words` words of 2+ characters,
    mostly in that script, read with confidence >= `min_conf`. (One stray character read from a
    logo or a table line used to be enough, which wrongly reported Sinhala/Tamil on English PDFs.)"""
    counts = {"si": 0, "ta": 0, "en": 0}
    for t in tokens:
        w = t.text.strip()
        if len(w) < 2 or t.confidence < min_conf:
            continue
        if _script_share(w, "\u0D80", "\u0DFF") >= 0.5:
            counts["si"] += 1
        elif _script_share(w, "\u0B80", "\u0BFF") >= 0.5:
            counts["ta"] += 1
        elif sum(ch.isascii() and ch.isalpha() for ch in w) >= 2:
            counts["en"] += 1
    langs = [code for code in ("si", "ta", "en") if counts[code] >= min_words]
    return langs or ["en"]


SUM_TOLERANCE = 0.011
MIN_DEDUCTION_LINES = 3


def deduction_sum_check(tokens: Sequence, cands: dict) -> Optional[dict]:
    """Own method: every amount printed between TOTAL EARNINGS and TOTAL DEDUCTIONS is a deduction
    line, so they must add up to the printed total. One equation then verifies APIT, EPF employee and
    every other deduction at once - fields no other rule can check (APIT was always 'unverified').
    Uses the amount column only, so it works even when labels and amounts are paired a row apart.
    OCR slips inside the list are handled the same way as elsewhere: a number printed without any
    separator on a slip whose amounts carry cents lost its decimal point ('53000' = 530.00), a row
    number stuck to an amount is dropped ('1 1,000.00'), and the sum may be repaired by changing ONE
    digit of ONE line (reported; the line is not an output field)."""
    from entity_extraction.amounts import parse_amount
    from entity_extraction.extractor import group_lines, split_amount
    from entity_extraction.invoice_checks import digit_close

    g, td = cands.get("gross_income"), cands.get("total_deductions")
    if not g or not td or g.amount_line < 0 or td.amount_line <= g.amount_line:
        return None
    gross, target = float(g.value), float(td.value)
    rows = []
    for ln in group_lines(tokens)[g.amount_line + 1: td.amount_line]:
        _, amt = split_amount(ln)
        if not amt:
            continue
        if len(amt) >= 2 and amt[0].text.isdigit() and len(amt[0].text) == 1 and "," in amt[1].text:
            amt = amt[1:]                         # 'recovery 1' + '1,000.00'
        rows.append(amt)
    if len(rows) < MIN_DEDUCTION_LINES:
        return None
    has_cents = sum(bool(re.search(r"[.,\s]\d{2}$", " ".join(t.text for t in a))) for a in rows) >= 0.6 * len(rows)
    amounts, notes = [], []
    for amt in rows:
        raw = " ".join(t.text for t in amt)
        p = parse_amount(raw)
        if p is None:
            return None                           # an unreadable amount: the sum cannot be trusted either way
        v = p.value
        if has_cents and len(amt) == 1 and raw.isdigit() and len(raw) >= 3:
            v = round(v / 100, 2)                 # decimal point lost
            notes.append(f"'{raw}' read as {v:,.2f} (decimal point lost)")
        if abs(v - gross) <= SUM_TOLERANCE:
            continue                              # the TOTAL EARNINGS line repeated under its label
        amounts.append(v)
    total = round(sum(amounts), 2)
    passed = abs(total - target) <= SUM_TOLERANCE
    repaired = False
    if not passed:
        fixes = [(i, round(a + target - total, 2)) for i, a in enumerate(amounts)
                 if a + target - total > 0 and digit_close(a, round(a + target - total, 2))]
        if len(fixes) == 1:
            i, new = fixes[0]
            notes.append(f"line {amounts[i]:,.2f} looks like {new:,.2f} (one digit misread)")
            passed = True
            repaired = True
    inside = {f for f, c in cands.items() if g.amount_line < c.amount_line < td.amount_line}
    # a sum that only holds after repairing a line verifies nothing: that line may be one of the fields
    verified = set() if repaired else inside | {"total_deductions"}
    return {"passed": passed, "lines": len(amounts), "sum": total, "fields": verified,
            "check": {"rule": "deduction lines add up to total deductions", "hard": False, "passed": passed,
                      "explanation": f"{len(amounts)} deduction lines sum to {total:,.2f}; total deductions "
                                     f"{target:,.2f}" + ("; " + "; ".join(notes) if notes else "")}}


def extract_payslip(tokens: Sequence, doc_id: str, engine: str = "tesseract-5",
                    processing_ms: Optional[int] = None,
                    to_original: Optional[Callable[[List[int]], List[int]]] = None,
                    document_type: str = "payslip", candidates: Optional[dict] = None) -> Tuple[OCROutput, dict]:
    """tokens must be in the coordinates of the image OCR ran on (the straightened image), because
    lines are rebuilt from box positions and a tilted photo puts a label and its amount at
    different heights. `to_original` maps each final bbox back to the uploaded photo.
    `candidates` lets multi-engine fusion pass already-chosen field values; the fiscal checks,
    confidence and ontology steps below are then identical to the single-engine path."""
    alignment = {"applied": False, "reason": "fields supplied by fusion"}
    if candidates is not None:
        cands = candidates
    else:
        cands, alignment = extract_aligned(tokens)
    td, net, gross = cands.get("total_deductions"), cands.get("net_amount"), cands.get("gross_income")
    if td and net and td.value == net.value and not (gross and abs(gross.value - 2 * net.value) < 0.011):
        # one printed amount cannot be both (two merged rows on a glare photo); net is the bold line
        cands = {f: c for f, c in cands.items() if f != "total_deductions"}
    amounts = {f: c.value for f, c in cands.items() if isinstance(c.value, float)}
    fiscal: FiscalReport = validate_and_correct(amounts)
    dsum = deduction_sum_check(tokens, cands) if candidates is None else None
    sum_verified = dsum["fields"] if dsum and dsum["passed"] else set()

    out_fields: List[ExtractedField] = []
    for f, c in cands.items():
        if f in HELPER_FIELDS:
            continue
        value = fiscal.values.get(f, c.value)
        conf = field_confidence(
            c.ocr_confidence,
            verified=(fiscal.verified(f) or f in sum_verified) and f not in fiscal.corrections,
            corrected=f in fiscal.corrections or c.realigned,
            contradicted=fiscal.contradicted(f),
            cleaned=c.cleaned,
            match_score=c.match_score,
        )
        out_fields.append(ExtractedField(
            field_name=f, value=value, raw_text=c.raw_text, confidence=conf,
            bbox=[int(x) for x in (to_original(c.bbox) if to_original else c.bbox)], tax_entity=ONTOLOGY.get(f, {}).get("tax_entity"),
            needs_review=needs_review(conf)))

    overall = round(sum(x.confidence for x in out_fields) / len(out_fields), 3) if out_fields else 0.0
    result = OCROutput(doc_id=doc_id, document_type=document_type, languages=detect_languages(tokens),
                       extracted_fields=out_fields, overall_confidence=overall,
                       needs_review=(not out_fields) or any(x.needs_review for x in out_fields),
                       engine=engine, processing_ms=processing_ms)
    report = {
        "checks": [{"rule": c.rule, "hard": c.hard, "passed": c.passed, "explanation": c.explanation}
                   for c in fiscal.checks] + ([dsum["check"]] if dsum else []),
        "corrections": {f: {"ocr_value": a, "corrected_value": b} for f, (a, b) in fiscal.corrections.items()},
        "notes": {f: c.notes for f, c in cands.items() if c.notes},
        "helper_values": {f: cands[f].value for f in HELPER_FIELDS if f in cands},
        "alignment": alignment,
    }
    return result, report


def to_contract_json(result: OCROutput, indent: Optional[int] = 2) -> str:
    """Serialise for other components. Optional fields that are empty are LEFT OUT instead of
    written as null, because the shared JSON schema types them as number/string, not null."""
    return result.model_dump_json(indent=indent, exclude_none=True)


def extract_document(tokens: Sequence, doc_id: str, engine: str = "tesseract-5",
                     processing_ms: Optional[int] = None,
                     to_original: Optional[Callable[[List[int]], List[int]]] = None) -> Tuple[OCROutput, dict]:
    """Classify first, then extract with the right extractor (FR8 before FR4).
    Payslips, bank statements, invoices, receipts and quotations are extracted; non-financial
    documents return 'unknown' - never forced into payslip fields."""
    from document_classifier import classify_tokens

    cls = classify_tokens(tokens)
    classification = {"document_type": cls.document_type, "confidence": cls.confidence,
                      "scores": cls.scores, "evidence": cls.evidence, "reason": cls.reason}
    if cls.document_type == "payslip":
        out, report = extract_payslip(tokens, doc_id, engine=engine, processing_ms=processing_ms,
                                      to_original=to_original, document_type="payslip")
        report["classification"] = classification
        return out, report
    if cls.document_type == "bank_statement":
        from entity_extraction.bank_statement import extract_bank_statement
        out, report = extract_bank_statement(tokens, doc_id, engine=engine, processing_ms=processing_ms,
                                             to_original=to_original, languages=detect_languages(tokens))
        report["classification"] = classification
        return out, report

    if cls.document_type in ("invoice", "receipt", "quotation"):
        from entity_extraction.invoice import extract_invoice
        out, report = extract_invoice(tokens, doc_id, engine=engine, processing_ms=processing_ms,
                                      to_original=to_original, document_type=cls.document_type,
                                      languages=detect_languages(tokens))
        report["classification"] = classification
        return out, report

    out = OCROutput(doc_id=doc_id, document_type=cls.document_type, languages=detect_languages(tokens),
                    extracted_fields=[], overall_confidence=0.0, needs_review=True,
                    engine=engine, processing_ms=processing_ms)
    note = ("not recognised as a financial document; no tax fields extracted"
            if cls.document_type == "unknown"
            else f"{cls.document_type} detected; field extraction for this type is not implemented yet")
    return out, {"checks": [], "corrections": {}, "notes": {"document": [note]}, "helper_values": {},
                 "classification": classification}
