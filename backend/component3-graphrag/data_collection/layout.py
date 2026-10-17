"""Read a PDF page by page and sort every text line into header, footer, margin note or body.

Sri Lankan Acts (documents.gov.lk / IRD) share one printed layout, checked against all seven
Inland Revenue Acts in the corpus:
- a running header on each page (page number + short title, sometimes on two lines), placed
  well below the top edge of the paper, so headers are found as the first lines of a page that
  repeat across pages (or are a bare page number), not by a fixed band; the same test is
  applied to the last lines of a page, for gazettes that number pages at the foot;
- printer codes such as "2 PL 005318-2,962 (07/2017)" at the foot of some pages;
- marginal notes (the short section titles) beside each section, in a smaller font (8pt beside
  10pt text), on the right of some pages and the left of others (the pages are mirrored, and
  heavily indented pages give no reliable column edge);
- schedules printed entirely in 8pt, the same size as the notes.

So a marginal note is a line that is (a) smaller than the main text and (b) wholly left or right
of every main-size line on its page. The main size is the Act's (10pt) on pages that carry
it, otherwise the page's own (schedule pages: 8pt, so their table cells are never notes).

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

# Tunable thresholds (points, or fractions of page size / page count)
HEADER_ZONE = 16.0  # lines starting within this distance of a page's first line may be header
REPEAT_SHARE = 0.2  # a header text (digits masked) appears on >= 20% of pages
SAME_ROW_TOL = 2.5  # lines this close vertically are one visual row
NOTE_SMALLER = 1.0  # a note's font is at least this much smaller than the main text

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_NUMBER_TOKEN = re.compile(r"^\s*(?:\d{1,3}[A-Z]?\.|\(\d{1,2}[A-Z]?\)|\([a-z]{1,4}\))\s*$")
_PRINTER_CODE = re.compile(r"\bPL\s*\d{3,}")  # "2 PL 005318—–2,962 (07/2017)"


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
                    text = "".join(span["text"] for span in _dedup(ln["spans"]))
                    if not text.strip():
                        continue
                    size = max(span["size"] for span in ln["spans"])
                    x0, y0, x1, y1 = ln["bbox"]
                    page.lines.append(Line(index, x0, y0, x1, y1, size, text.strip()))
            page.lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
            pages.append(page)
    return pages


def _dedup(spans: list[dict]) -> list[dict]:
    """Drop a span that the next span repeats at the same position.

    Some PDFs draw the section number twice ("5." then "5. Section 52 ..." at the same x),
    which would otherwise read as "5.5. Section 52".
    """
    kept = []
    for i, span in enumerate(spans):
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        if (
            nxt is not None
            and span["text"].strip()
            and abs(nxt["bbox"][0] - span["bbox"][0]) < 1.0
            and nxt["text"].startswith(span["text"])
        ):
            continue
        kept.append(span)
    return kept


def _top_lines(page: Page) -> list[Line]:
    if not page.lines:
        return []
    first = min(ln.y0 for ln in page.lines)
    return [ln for ln in page.lines if ln.y0 <= first + HEADER_ZONE]


def _bottom_lines(page: Page) -> list[Line]:
    if not page.lines:
        return []
    last = max(ln.y0 for ln in page.lines)
    return [ln for ln in page.lines if ln.y0 >= last - HEADER_ZONE]


def _is_header_text(text: str, repeated: set[str]) -> bool:
    if _PAGE_NUMBER.match(text):
        return True
    norm = _norm_repeat(text)
    return norm in repeated and len(norm) >= 6 and any(c.isalpha() for c in norm)


def classify(pages: list[Page]) -> dict[int, float]:
    """Mark each line's kind in place. Returns the main text font size of each page."""
    # 1. running headers/footers: page numbers, and text repeated at the top (or bottom) of
    #    many pages; printer codes anywhere
    min_repeats = max(2, int(REPEAT_SHARE * len(pages)))
    for edge, kind in ((_top_lines, "header"), (_bottom_lines, "footer")):
        seen: Counter[str] = Counter()
        for p in pages:
            seen.update({_norm_repeat(ln.text) for ln in edge(p)})
        repeated = {t for t, n in seen.items() if n >= min_repeats}
        for p in pages:
            for ln in edge(p):
                if ln.kind == "body" and _is_header_text(ln.text, repeated):
                    ln.kind = kind
    for p in pages:
        for ln in p.lines:
            if _PRINTER_CODE.search(ln.text):
                ln.kind = "footer"

    # 2. marginal notes: smaller than the main text, and wholly outside the span of the
    #    main-size text on their page (main text never sits in the margin; table cells do not
    #    sit outside the column)
    doc_main = _main_size([ln for p in pages for ln in p.lines if ln.kind == "body"])
    sizes: dict[int, float] = {}
    for p in pages:
        body = [ln for ln in p.lines if ln.kind == "body"]
        if not body:
            continue
        at_doc_main = [ln for ln in body if abs(ln.size - doc_main) <= 0.5]
        carries_doc_main = len(at_doc_main) >= 2 and (
            max(ln.x1 for ln in at_doc_main) - min(ln.x0 for ln in at_doc_main) >= 0.4 * p.width
        )
        main = sizes[p.number] = doc_main if carries_doc_main else _main_size(body)
        mains = [ln for ln in body if ln.size >= main - 0.5]
        left = min(ln.x0 for ln in mains)
        right = max(ln.x1 for ln in mains)
        for ln in body:
            if ln.size > main - NOTE_SMALLER or _NUMBER_TOKEN.match(ln.text):
                continue
            if ln.x1 <= left + 2 or ln.x0 >= right - 2:
                ln.kind = "margin"
    return sizes


def _main_size(lines: list[Line]) -> float:
    """The font size carrying the most characters."""
    weight: Counter[float] = Counter()
    for ln in lines:
        weight[round(ln.size * 2) / 2] += len(ln.text)
    return weight.most_common(1)[0][0] if weight else 0.0


@dataclass
class BodyRow:
    """One visual row of body text (lines on the same baseline merged left to right)."""

    page: int
    y0: float
    text: str
    size: float = 0.0


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
    return BodyRow(
        lines[0].page,
        min(ln.y0 for ln in lines),
        " ".join(ln.text for ln in lines),
        max(ln.size for ln in lines),
    )


def margin_notes(pages: list[Page]) -> list[MarginNote]:
    """Group consecutive margin lines on a page into notes."""
    notes: list[MarginNote] = []
    for p in pages:
        group: list[Line] = []
        margin = [ln for ln in p.lines if ln.kind == "margin"]
        gap = statistics.median([ln.y1 - ln.y0 for ln in margin]) * 0.5 if margin else 0
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
    text = re.sub(r"(\w)- ([A-Z])", r"\1-\2", text)  # "Non- Governmental" -> "Non-Governmental"
    text = re.sub(r"(\w)- ([a-z])", r"\1\2", text)  # re-join "speci- fied", keep "Non-Governmental"
    return MarginNote(lines[0].page, lines[0].y0, lines[-1].y1, " ".join(text.split()))
