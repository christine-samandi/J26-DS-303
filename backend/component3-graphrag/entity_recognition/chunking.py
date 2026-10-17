"""Cut parsed Acts (Phase 1 JSON) into retrieval chunks.

A chunk is the unit that is searched, cited and annotated: one subsection of a section, or one
schedule paragraph (or one of its sub-paragraphs). Every chunk keeps where it came from (Act,
section, subsection, schedule, page, Part/Chapter heading, marginal note) so an answer can cite
it exactly.

The chunk text is VERBATIM from the parsed Act (only whitespace is normalised): Component 4
checks quoted evidence against it, so nothing is rewritten here. Context for search (marginal
note, heading) is kept in separate fields.

Very long subsections (e.g. the interpretation section) are split between list items
"(a)", "(b)" ... so no chunk exceeds MAX_CHARS where a clean split exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1800  # long enough for a full rate table, short enough for precise evidence
MIN_LEAD_CHARS = 40  # shorter text before "(1)" is merged into the first subsection

# a list item "(a)", "(ii)", "(aa)" starting right after ";" ":" "—" or "-" (the Acts' item breaks)
_ITEM_BREAK = re.compile(r"(?<=[;:—–-])\s+(?=\((?:[a-z]{1,4}|[ivx]{1,5})\)\s)")


@dataclass
class Chunk:
    chunk_id: str  # CHUNK-IRA-2017-S052-2, CHUNK-IRA-2017-SCH5-P002
    document_id: str
    segment_id: str
    kind: str  # section | schedule_paragraph | schedule_text | preamble
    section: str | None
    subsection: str | None
    schedule: str | None
    paragraph: str | None
    part: int  # 1, 2 ... when a long text was split; 1 otherwise
    heading: str | None
    marginal_note: str | None
    page_start: int
    page_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str) -> list[str]:
    """Split text longer than MAX_CHARS between list items; greedy packing, verbatim pieces."""
    if len(text) <= MAX_CHARS:
        return [text]
    pieces, start = [], 0
    for m in _ITEM_BREAK.finditer(text):
        pieces.append(text[start : m.start()])
        start = m.end()
    pieces.append(text[start:])
    out: list[str] = []
    for piece in pieces:
        if out and len(out[-1]) + 1 + len(piece) <= MAX_CHARS:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _id_part(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def chunks_from_document(parsed: dict) -> list[Chunk]:
    doc_id = parsed["document_id"]
    out: list[Chunk] = []
    for seg in parsed["segments"]:
        base = seg["segment_id"].replace("SEG-", "CHUNK-", 1)
        is_schedule = seg["kind"] in ("schedule_paragraph", "schedule_text")
        units: list[tuple[str | None, str]] = []  # (subsection label, text)
        lead = seg["text"].strip()
        subs = seg.get("subsections") or []
        if subs:
            first = True
            for sub in subs:
                text = sub["text"].strip()
                if first and lead and len(lead) < MIN_LEAD_CHARS:
                    text = f"{lead} ({sub['label']}) {text}"
                elif first and lead:
                    units.append((None, lead))
                if not (first and lead and len(lead) < MIN_LEAD_CHARS):
                    text = f"({sub['label']}) {text}"
                units.append((sub["label"], text))
                first = False
        elif lead:
            units.append((None, lead))

        for label, text in units:
            parts = _split_long(text)
            for n, part_text in enumerate(parts, start=1):
                chunk_id = base
                if label:
                    chunk_id += f"-{_id_part(label)}"
                elif subs:
                    chunk_id += "-0"  # text before the first subsection
                if len(parts) > 1:
                    chunk_id += f"-P{n}"
                out.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        segment_id=seg["segment_id"],
                        kind=seg["kind"],
                        section=None if is_schedule else seg["number"],
                        subsection=label if not is_schedule else None,
                        schedule=seg["schedule"],
                        paragraph=seg["number"] if is_schedule else None,
                        part=n,
                        heading=seg.get("heading"),
                        marginal_note=seg["marginal_note"],
                        page_start=seg["page_start"],
                        page_end=seg["page_end"],
                        text=part_text,
                    )
                )
    return out


def load_chunks(processed_dir: Path, only: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(processed_dir.glob("DOC-*.json")):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if only and parsed["document_id"] != only:
            continue
        chunks.extend(chunks_from_document(parsed))
    return chunks
