"""Split an Act's body text into sections, subsections and schedule paragraphs.

Input: body rows and margin notes from `layout.py`. Output: a ParsedDocument whose segments
carry their number, schedule, page range, marginal note, text and subsections. These segments
are the units Phase 2 extracts entities from and Phase 4 embeds for vector search.

Section numbers must run in order (1, 2, 3 ... with 52A after 52). A line that looks like a
section start but breaks the order (e.g. "2025. The amount ...") is treated as ordinary text.
That one rule removes most false splits without any document-specific patterns.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from data_collection.layout import BodyRow, MarginNote

MAX_SECTION_JUMP = 6  # repealed sections leave small gaps in numbering

_ORDINALS = "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH"
_SCHEDULE = re.compile(rf"^\s*({_ORDINALS})\s+SCHEDULE\b\s*(?:\[[^\]]*\])?\s*[.:]?\s*$")
_SECTION = re.compile(r"^(?P<num>\d{1,3})(?P<suffix>[A-Z]{0,2})\.\s+(?P<rest>[(\"“A-Z].*)$")
_SUBSECTION = re.compile(r"(?:^|\n)\((?P<num>\d{1,2})(?P<suffix>[A-Z]{0,2})\)\s+")
_CERTIFIED = re.compile(
    r"\[\s*Certified\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]+),?\s+(\d{4})\s*\]",
    re.IGNORECASE,
)
_TITLE = re.compile(r"^.*ACT,?\s+No\.?\s*\d+\s+OF\s+\d{4}\s*$", re.IGNORECASE)


@dataclass
class Subsection:
    label: str
    text: str


@dataclass
class Segment:
    segment_id: str
    kind: str  # preamble | section | schedule_paragraph
    number: str | None
    schedule: str | None  # e.g. "FIFTH SCHEDULE"
    marginal_note: str | None
    page_start: int
    page_end: int
    text: str
    subsections: list[Subsection] = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_id: str
    source_file: str
    sha256: str
    page_count: int
    title_detected: str | None
    certified_date_detected: str | None
    segments: list[Segment]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# --- text clean-up ------------------------------------------------------------------------


def join_rows(rows: list[str]) -> str:
    """Join visual rows into flowing text, re-joining words hyphenated at a line end.

    A newline is kept before "(2)"-style markers so subsections can still be found.
    """
    out = ""
    for row in rows:
        row = " ".join(row.split())
        if not out:
            out = row
        elif out.endswith("-") and row[:1].islower():
            out = out[:-1] + row
        elif re.match(r"^\(\d{1,2}[A-Z]{0,2}\)\s", row):
            out += "\n" + row
        else:
            out += " " + row
    return out


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- numbering ----------------------------------------------------------------------------


def _key(num: str, suffix: str) -> tuple[int, str]:
    return int(num), suffix


def _follows(
    prev: tuple[int, str] | None, cur: tuple[int, str], max_jump: int = MAX_SECTION_JUMP
) -> bool:
    """Is `cur` a plausible next number after `prev`? (1 first; 52 -> 52A or 53..58)

    Sections may skip a few numbers (repealed sections); subsections use max_jump=1.
    """
    if prev is None:
        return cur == (1, "")
    if cur[0] == prev[0]:
        return cur[1] > prev[1]
    return 0 < cur[0] - prev[0] <= max_jump and cur[1] == ""


def split_subsections(text: str) -> tuple[str, list[Subsection]]:
    """Leading text, then (1), (2), (2A) ... in order. Out-of-order markers stay in the text."""
    starts: list[tuple[int, int, str]] = []  # (marker start, text start, label)
    prev: tuple[int, str] | None = None
    for m in _SUBSECTION.finditer(text):
        cur = _key(m["num"], m["suffix"])
        if _follows(prev, cur, max_jump=1):
            starts.append((m.start(), m.end(), f"{m['num']}{m['suffix']}"))
            prev = cur
    if not starts:
        return _flat(text), []
    lead = _flat(text[: starts[0][0]])
    subs = []
    for i, (_, begin, label) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        subs.append(Subsection(label, _flat(text[begin:end])))
    return lead, subs


# --- metadata -----------------------------------------------------------------------------


def detect_certified_date(text: str) -> str | None:
    m = _CERTIFIED.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m[1]} {m[2][:3]} {m[3]}", "%d %b %Y").date().isoformat()
    except ValueError:
        return None


def detect_title(rows: list[BodyRow]) -> str | None:
    for row in rows[:40]:
        if _TITLE.match(row.text.strip()):
            return _flat(row.text)
    return None


# --- segmentation -------------------------------------------------------------------------


def _short_id(document_id: str) -> str:
    return document_id.removeprefix("DOC-")


def segment(
    document_id: str,
    rows: list[BodyRow],
    notes: list[MarginNote],
) -> tuple[list[Segment], list[str]]:
    short = _short_id(document_id)
    segments: list[Segment] = []
    warnings: list[str] = []

    schedule: str | None = None
    prev: tuple[int, str] | None = None
    current: dict | None = None
    preamble_rows: list[BodyRow] = []

    def close() -> None:
        if current is None:
            return
        body = join_rows(current["rows"])
        lead, subs = split_subsections(body)
        number = current["number"]
        if current["schedule"]:
            sched_no = _ORDINALS.split("|").index(current["schedule"].split()[0]) + 1
            seg_id = f"SEG-{short}-SCH{sched_no}-P{int(re.sub(r'[A-Z]', '', number)):03d}"
            seg_id += re.sub(r"\d", "", number)
        else:
            seg_id = f"SEG-{short}-S{int(re.sub(r'[A-Z]', '', number)):03d}"
            seg_id += re.sub(r"\d", "", number)
        segments.append(
            Segment(
                segment_id=seg_id,
                kind="schedule_paragraph" if current["schedule"] else "section",
                number=number,
                schedule=current["schedule"],
                marginal_note=_note_for(notes, current["rows_meta"][0]),
                page_start=current["rows_meta"][0].page,
                page_end=current["rows_meta"][-1].page,
                text=lead if subs else _flat(body),
                subsections=subs,
            )
        )

    for row in rows:
        text = row.text.strip()
        sched = _SCHEDULE.match(text.upper()) if text.isupper() or "SCHEDULE" in text else None
        if sched and text == text.upper():
            close()
            current = None
            schedule = f"{sched[1].upper()} SCHEDULE"
            prev = None
            continue

        m = _SECTION.match(text)
        if m and _follows(prev, _key(m["num"], m["suffix"])):
            close()
            prev = _key(m["num"], m["suffix"])
            current = {
                "number": f"{m['num']}{m['suffix']}",
                "schedule": schedule,
                "rows": [m["rest"]],
                "rows_meta": [row],
            }
            continue

        if current is None:
            preamble_rows.append(row)
        else:
            current["rows"].append(text)
            current["rows_meta"].append(row)
    close()

    if preamble_rows:
        segments.insert(
            0,
            Segment(
                segment_id=f"SEG-{short}-PRE",
                kind="preamble",
                number=None,
                schedule=None,
                marginal_note=None,
                page_start=preamble_rows[0].page,
                page_end=preamble_rows[-1].page,
                text=join_rows([r.text for r in preamble_rows]),
            ),
        )
    if not any(s.kind == "section" for s in segments):
        warnings.append(
            "no numbered sections found: check the layout with data_collection.inspect_pdf"
        )
    return segments, warnings


def _note_for(notes: list[MarginNote], start: BodyRow) -> str | None:
    """The margin note beside a section's first row (same page, overlapping vertically)."""
    best = None
    for note in notes:
        if note.page != start.page:
            continue
        distance = abs(note.y0 - start.y0)
        if distance <= 14 and (best is None or distance < abs(best.y0 - start.y0)):
            best = note
    return best.text if best else None


def check_against_registry(
    parsed: ParsedDocument, certified: date, expected_title_number: str | None
) -> None:
    """Add warnings when what the PDF says disagrees with the corpus registry."""
    if parsed.certified_date_detected is None:
        parsed.warnings.append("certified date not found in the PDF text")
    elif parsed.certified_date_detected != certified.isoformat():
        parsed.warnings.append(
            f"certified date in PDF ({parsed.certified_date_detected}) differs from "
            f"registry ({certified.isoformat()})"
        )
    if expected_title_number and parsed.title_detected:
        number, year = expected_title_number.split(" of ")
        pattern = rf"No\.?\s*0*{int(number)}\s+OF\s+{year}"
        if not re.search(pattern, parsed.title_detected, re.IGNORECASE):
            parsed.warnings.append(
                f"title '{parsed.title_detected}' does not match Act No. {expected_title_number}"
            )
