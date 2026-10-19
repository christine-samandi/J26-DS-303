"""Find tax fields in OCR output using labels + layout (Objective 2.3, Gap 1).

Why this module exists:
- OCR gives words and boxes, not meaning. On a payslip the meaning comes from layout: a value
  sits to the RIGHT of its label on the same line, or directly BELOW it. This module rebuilds
  lines from word boxes, finds the label on each line (keywords.py) and takes the amount next
  to it (amounts.py), keeping the amount's box for traceability.
- It works on any engine's tokens (anything with .text, .confidence, .bbox), so Tesseract,
  PaddleOCR or a PDF text layer can all feed it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from entity_extraction.amounts import looks_numeric, parse_amount
from entity_extraction.keywords import match_field

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
PERSONAL_FIELD = re.compile(r"^\W*(name|nic|number|no\b|employee|emp\b|department|designation|address|"
                            r"account|date|period|month)\b", re.I)
ISSUER_HINTS = re.compile(r"\b(pvt|ltd|limited|plc|company|co\.|bank|holdings|corporation|\(pvt\))", re.I)


@dataclass
class Candidate:
    field: str
    value: object                 # float for amounts, str for issuer / pay_period
    raw_text: str                 # exactly as OCR read it
    label_text: str = ""
    ocr_confidence: float = 0.0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    match_score: float = 1.0
    cleaned: bool = False         # amount text was corrected by amounts.py
    notes: List[str] = field(default_factory=list)
    line_index: int = -1          # line of the label (for row-alignment checks)
    amount_line: int = -1         # line the amount was taken from
    realigned: bool = False       # value taken from a neighbouring row (see alignment.py)
    phrase_words: int = 1         # words in the matched label phrase: 'net salary' (2) is more specific than 'net' (1)


@dataclass
class Line:
    tokens: list

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def top(self) -> int:
        return min(t.bbox[1] for t in self.tokens)

    @property
    def bottom(self) -> int:
        return max(t.bbox[1] + t.bbox[3] for t in self.tokens)


def union_bbox(tokens: Sequence) -> List[int]:
    x0 = min(t.bbox[0] for t in tokens)
    y0 = min(t.bbox[1] for t in tokens)
    x1 = max(t.bbox[0] + t.bbox[2] for t in tokens)
    y1 = max(t.bbox[1] + t.bbox[3] for t in tokens)
    return [x0, y0, x1 - x0, y1 - y0]


def group_lines(tokens: Sequence) -> List[Line]:
    """Words whose vertical centres fall inside each other's height belong to one line.
    Done from boxes (not the engine's line ids) so every engine is treated the same."""
    toks = sorted((t for t in tokens if t.text.strip()), key=lambda t: t.bbox[1] + t.bbox[3] / 2)
    lines: List[List] = []
    for t in toks:
        cy = t.bbox[1] + t.bbox[3] / 2
        for ln in lines:
            ref = ln[-1]
            rcy = ref.bbox[1] + ref.bbox[3] / 2
            if abs(cy - rcy) <= 0.5 * max(t.bbox[3], ref.bbox[3]):
                ln.append(t)
                break
        else:
            lines.append([t])
    return [Line(sorted(ln, key=lambda t: t.bbox[0])) for ln in lines]


def _is_year(text: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d", text.strip()))


def split_amount(line: Line):
    """Return (label_tokens, amount_tokens): the trailing run of numeric-looking tokens is the
    amount (OCR sometimes splits '150, 000.00' into pieces, so they are joined)."""
    toks = line.tokens
    # Drop trailing punctuation-only fragments (table rules read as ' | \ ) so they do not hide
    # the amount: "CLOSING 26 215 08 '" must still give the amount 26 215 08.
    while toks and not any(ch.isalnum() for ch in toks[-1].text):
        toks = toks[:-1]
    # A table rule read as 1-2 letters right after an amount ('26 215 08 if') is noise, not a label.
    while len(toks) >= 2 and len(toks[-1].text) <= 2 and not looks_numeric(toks[-1].text) \
            and looks_numeric(toks[-2].text):
        toks = toks[:-1]
    i = len(toks)
    while i > 0 and looks_numeric(toks[i - 1].text):
        i -= 1
    amount = toks[i:]
    if amount and len(amount) == 1 and _is_year(amount[0].text):
        return toks, []
    return toks[:i], amount


def _amount_candidate(field_name, label_toks, amount_toks, score) -> Optional[Candidate]:
    raw = " ".join(t.text for t in amount_toks)     # spaces kept: '28 775 29' -> cents rule
    parsed = parse_amount(raw)
    if parsed is None:
        return None
    conf = min(t.confidence for t in amount_toks)
    c = Candidate(field=field_name, value=parsed.value, raw_text=" ".join(t.text for t in amount_toks),
                  label_text=" ".join(t.text for t in label_toks), ocr_confidence=round(conf, 4),
                  bbox=union_bbox(amount_toks), match_score=score, cleaned=parsed.corrected)
    if parsed.corrected:
        c.notes.append(f"amount text cleaned: '{c.raw_text}' -> {parsed.value:,.2f}")
    return c


def find_pay_period(lines: List[Line]) -> Optional[Candidate]:
    pat = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*[-/,]?\s*((?:19|20)\d\d)\b", re.I)
    num = re.compile(r"\b((?:19|20)\d\d)[-/.](0?[1-9]|1[0-2])\b|\b(0?[1-9]|1[0-2])[-/.]((?:19|20)\d\d)\b")
    for ln in lines:
        m = pat.search(ln.text)
        if m:
            value = f"{m.group(2)}-{MONTHS[m.group(1)[:3].lower()]:02d}"
        else:
            m2 = num.search(ln.text)
            if not m2:
                continue
            y, mo = (m2.group(1), m2.group(2)) if m2.group(1) else (m2.group(4), m2.group(3))
            value = f"{y}-{int(mo):02d}"
        conf = sum(t.confidence for t in ln.tokens) / len(ln.tokens)
        return Candidate("pay_period", value, ln.text, ocr_confidence=round(conf, 4), bbox=union_bbox(ln.tokens))
    return None


def find_issuer(lines: List[Line]) -> Optional[Candidate]:
    """Employer / company name: a line with a company word (Ltd, PVT, PLC, Bank ...) near the top,
    otherwise the first text line. The fallback gets lower confidence."""
    # Lines that are field labels, carry an amount or hold personal key-value data (Name :, NIC ...)
    # are never the issuer: 'EPF Corporation Contribution 28,775.29' contains 'Corporation' but is a field.
    top = [ln for ln in lines[: max(3, len(lines) // 4)]
           if not split_amount(ln)[1] and not match_field(ln.text) and not PERSONAL_FIELD.match(ln.text)]
    for ln in top:
        if ISSUER_HINTS.search(ln.text):
            conf = sum(t.confidence for t in ln.tokens) / len(ln.tokens)
            return Candidate("issuer", ln.text.strip(" -|"), ln.text, ocr_confidence=round(conf, 4),
                             bbox=union_bbox(ln.tokens))
    for ln in top:
        if sum(c.isalpha() for c in ln.text) >= 3 and not split_amount(ln)[1]:
            conf = sum(t.confidence for t in ln.tokens) / len(ln.tokens) * 0.8
            c = Candidate("issuer", ln.text.strip(" -|"), ln.text, ocr_confidence=round(conf, 4),
                          bbox=union_bbox(ln.tokens))
            c.notes.append("issuer guessed from the first text line (no company word found)")
            return c
    return None


def extract_fields(tokens: Sequence) -> Dict[str, Candidate]:
    """Return the best candidate per field (amount fields, plus issuer and pay_period)."""
    lines = group_lines(tokens)
    found: Dict[str, Candidate] = {}

    def keep(c: Optional[Candidate]):
        if c is None:
            return
        old = found.get(c.field)
        # A more specific label wins ('Net Salary' over 'Net bonus payment', both containing 'net'),
        # then the closer match, then OCR confidence.
        if old is None or (c.match_score + 0.05 * c.phrase_words, c.ocr_confidence) > \
                (old.match_score + 0.05 * old.phrase_words, old.ocr_confidence):
            found[c.field] = c

    for i, ln in enumerate(lines):
        label_toks, amount_toks = split_amount(ln)
        if not label_toks:
            continue
        m = match_field(" ".join(t.text for t in label_toks))
        if not m:
            continue
        fname, score, phrase = m
        words = len(phrase.split())
        if amount_toks:
            c = _amount_candidate(fname, label_toks, amount_toks, score)
            if c:
                c.line_index, c.amount_line, c.phrase_words = i, i, words
            keep(c)
            continue
        # Label without a value on its line: take an amount-only line directly below it. The amount
        # may sit under the label or far to the right in an amount column (right-aligned slips).
        lx0 = label_toks[0].bbox[0]
        for j, nxt in enumerate(lines[i + 1: i + 3], start=i + 1):
            n_label, n_amount = split_amount(nxt)
            if n_amount and not n_label:
                ax1 = n_amount[-1].bbox[0] + n_amount[-1].bbox[2]
                if ax1 >= lx0 - 50 and nxt.top - ln.bottom < 3 * (ln.bottom - ln.top):
                    c = _amount_candidate(fname, label_toks, n_amount, score)
                    if c:
                        c.notes.append("value found below its label")
                        c.line_index, c.amount_line, c.phrase_words = i, j, words
                    keep(c)
                    break

    keep(find_pay_period(lines))
    keep(find_issuer(lines))
    return found
