"""Read a PDF page by page and sort every text line into header, footer, margin note or body.

Sri Lankan Acts (documents.gov.lk / IRD) share one printed layout:
- a running title at the top of each page and a page number at the top or bottom,
- marginal notes (the short section titles) in the outer margin beside each section,
- one body column of numbered sections, then the schedules.

This module only does the geometry. Turning body lines into sections is in `segment.py`.
`python -m data_collection.inspect_pdf <pdf>` prints what this module decided, line by line, so
the thresholds below can be tuned against the real documents.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Tunable thresholds (fractions of page height / width)
HEADER_BAND = 0.08  # top 8% of the page
FOOTER_BAND = 0.08  # bottom 8%
REPEAT_SHARE = 0.3  # a header/footer text appears on >= 30% of pages
SAME_ROW_TOL = 2.5  # points: lines this close vertically are one visual row

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_NUMBER_TOKEN = re.compile(r"^\s*(?:\d{1,3}[A-Z]?\.|\(\d{1,2}[A-Z]?\)|\([a-z]{1,4}\))\s*$")


@dataclass
class Line:
    page: int  # 1-based
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    text: str
    kind: str = "body"  # body | header | footer | margin

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Page:
    number: int
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)


def _norm_repeat(text: str) -> str:
    """Header text with digits masked, so 'Act, No. 2 of 2025 3' matches across pages."""
    return re.sub(r"\d+", "#", " ".join(text.split())).lower()


def read_pages(pdf_path: Path) -> list[Page]:
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(pdf_path) as doc:
        for index, pdf_page in enumerate(doc, start=1):
            page = Page(index, pdf_page.rect.width, pdf_page.rect.height)
            for block in pdf_page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for ln in block["lines"]:
                    text = "".join(span["text"] for span in ln["spans"])
                    if not text.strip():
                        continue
                    size = max(span["size"] for span in ln["spans"])
                    x0, y0, x1, y1 = ln["bbox"]
                    page.lines.append(Line(index, x0, y0, x1, y1, size, text.strip()))
            page.lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
            pages.append(page)
    return pages


def _body_column(pages: list[Page]) -> tuple[float, float]:
    """Left and right edges of the main text column, from the wide lines."""
    wide = [ln for p in pages for ln in p.lines if ln.width > 0.4 * p.width]
    if not wide:
        all_lines = [ln for p in pages for ln in p.lines]
        return min(ln.x0 for ln in all_lines), max(ln.x1 for ln in all_lines)
    left = Counter(round(ln.x0 / 5) * 5 for ln in wide).most_common(1)[0][0]
    rights = sorted(ln.x1 for ln in wide)
    right = rights[int(0.95 * (len(rights) - 1))]
    return float(left), float(right)


def classify(pages: list[Page]) -> tuple[float, float]:
    """Mark each line's kind in place. Returns the body column (left, right)."""
    # 1. running headers / footers: repeated text in the top or bottom band, and page numbers
    band_texts: Counter[str] = Counter()
    for p in pages:
        seen = set()
        for ln in p.lines:
            if ln.y1 <= HEADER_BAND * p.height or ln.y0 >= (1 - FOOTER_BAND) * p.height:
                seen.add(_norm_repeat(ln.text))
        band_texts.update(seen)
    min_repeats = max(2, int(REPEAT_SHARE * len(pages)))
    repeated = {t for t, n in band_texts.items() if n >= min_repeats}

    for p in pages:
        for ln in p.lines:
            in_top = ln.y1 <= HEADER_BAND * p.height
            in_bottom = ln.y0 >= (1 - FOOTER_BAND) * p.height
            if (in_top or in_bottom) and (
                _PAGE_NUMBER.match(ln.text) or _norm_repeat(ln.text) in repeated
            ):
                ln.kind = "header" if in_top else "footer"

    # 2. marginal notes: outside the body column, and not a hanging "52." or "(1)" number
    left, right = _body_column(pages)
    for p in pages:
        for ln in p.lines:
            if ln.kind != "body" or _NUMBER_TOKEN.match(ln.text):
                continue
            if ln.x0 >= right - 2 or ln.x1 <= left + 2:
                ln.kind = "margin"
    return left, right


@dataclass
class BodyRow:
    """One visual row of body text (lines on the same baseline merged left to right)."""

    page: int
    y0: float
    text: str


@dataclass
class MarginNote:
    page: int
    y0: float
    y1: float
    text: str


def body_rows(pages: list[Page]) -> list[BodyRow]:
    rows: list[BodyRow] = []
    for p in pages:
        current: list[Line] = []
        for ln in (ln for ln in p.lines if ln.kind == "body"):
            if current and abs(ln.y0 - current[0].y0) > SAME_ROW_TOL:
                rows.append(_merge(current))
                current = []
            current.append(ln)
        if current:
            rows.append(_merge(current))
    return rows


def _merge(lines: list[Line]) -> BodyRow:
    lines = sorted(lines, key=lambda ln: ln.x0)
    return BodyRow(lines[0].page, min(ln.y0 for ln in lines), " ".join(ln.text for ln in lines))


def margin_notes(pages: list[Page]) -> list[MarginNote]:
    """Group consecutive margin lines on a page into notes."""
    notes: list[MarginNote] = []
    for p in pages:
        group: list[Line] = []
        margin = [ln for ln in p.lines if ln.kind == "margin"]
        gap = statistics.median([ln.y1 - ln.y0 for ln in margin]) * 1.6 if margin else 0
        for ln in margin:
            if group and ln.y0 - group[-1].y1 > gap:
                notes.append(_note(group))
                group = []
            group.append(ln)
        if group:
            notes.append(_note(group))
    return notes


def _note(lines: list[Line]) -> MarginNote:
    text = " ".join(ln.text for ln in lines)
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # re-join words hyphenated across note lines
    return MarginNote(lines[0].page, lines[0].y0, lines[-1].y1, " ".join(text.split()))
