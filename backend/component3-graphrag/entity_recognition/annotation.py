"""Build the hand-annotation workbook used to measure extraction precision and recall.

    python -m entity_recognition.annotation          # -> ground_truth/entity_annotations.xlsx

The sample (fixed, reproducible):
- CORE: every chunk that states or changes the two PP1 rules (personal relief, individual tax
  bands) in each Act, because those are the facts the system must get right;
- RANDOM: 20 other chunks drawn with a fixed seed, 8 from the 2017 Act and 2 from each
  amendment Act, 120-500 characters long, so the score is not only about the PP1 rules.

The annotator sees ONLY the chunk text, never the system's output (blind annotation), and
marks each chunk "done" when finished; evaluation uses done chunks only.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import get_args

from entity_recognition.chunking import Chunk, load_chunks
from j26_contracts.retrieval import EntityLabel

SEED = 2026
CORE = [
    "CHUNK-IRA-2017-S052-2",  # reliefs are deducted (resident individuals)
    "CHUNK-IRA-2017-SCH5-P002",  # personal relief Rs. 500,000
    "CHUNK-IRA-2017-SCH1-P001-1",  # 2017 tax bands
    "CHUNK-IRA-AMEND-10-2021-S051-1-P1",  # 2020 tax bands
    "CHUNK-IRA-AMEND-10-2021-S055-2-P1",  # personal relief Rs. 3,000,000
    "CHUNK-IRA-AMEND-45-2022-S033-1C",  # 2023 tax bands
    "CHUNK-IRA-AMEND-45-2022-S036-2",  # personal relief Rs. 1,200,000
    "CHUNK-IRA-AMEND-02-2025-S001-2",  # operation date April 1, 2025
    "CHUNK-IRA-AMEND-02-2025-S003-1-P1",  # 2025 tax bands
    "CHUNK-IRA-AMEND-02-2025-S005-3",  # personal relief Rs. 1,800,000
]
RANDOM_PER_DOC = {"DOC-IRA-2017": 8}  # every other document: 2
RANDOM_DEFAULT = 2
MIN_CHARS, MAX_CHARS = 120, 500

LABELS = list(get_args(EntityLabel))
SHEET = "Annotate"
DONE = "done"

GUIDE = [
    "HOW TO ANNOTATE (about 1-2 hours; you can stop and continue later)",
    "",
    "Purpose: this is the answer key for the entity extractor. Mark what a careful human sees,",
    "not what you think the program finds. Do not look at the program's output while annotating.",
    "",
    "For each chunk on the 'Annotate' sheet:",
    "  1. Read the chunk text (the tall row).",
    "  2. For every entity, use one empty row below it: choose the LABEL in column B (dropdown)",
    "     and copy-paste the EXACT words from the text into column C. One entity per row.",
    "     If the same words appear twice in the chunk, add two rows.",
    "  3. When the chunk is finished, type  done  in column D of its grey header row.",
    "     Chunks without 'done' are ignored, so you can annotate in several sittings.",
    "  4. Not enough rows? Insert rows inside the chunk's block (column A must show its chunk ID).",
    "",
    "LABELS (what to mark, and the exact words to copy)",
    "  AMOUNT       money: 'Rs. 1,800,000', 'Rs.0', 'Rs. 3 million', 'sixty thousand rupees'",
    "  RATE         a percentage: '24%', '15 per centum', 'six per centum', 'Zero percent'",
    "  DATE         a calendar date with a year: 'April 1, 2025', '01.04.2025',",
    "               '24th of October, 2017'",
    "               (not 'thirty first day of March' with no year; do not include 'on or after')",
    "  SECTION_REF  a reference to law, the whole chain as ONE entity:",
    "               'paragraph 2 of the Fifth Schedule to the principal enactment',",
    "               'Section 52 of the principal enactment', 'subsection (2)', 'this section',",
    "               'item (iii) of that subparagraph', 'Inland Revenue Act, No. 24 of 2017'",
    "  TAX_SLAB     a rate-table band: 'Exceeding Rs. 600,000 but not exceeding Rs. 1,200,000',",
    "               'Not exceeding Rs. 500,000'. ALSO mark the amounts inside it as AMOUNT.",
    "  RELIEF       'personal relief', 'relief', 'reliefs'",
    "  DEDUCTION    'deduction(s)', 'qualifying payment(s)', 'allowable deduction'",
    "  EXEMPTION    'exempt amount(s)', 'exemption', 'exempt from tax'",
    "  INCOME_TYPE  'employment income', 'income from employment', 'business income',",
    "               'income from an investment', 'rental income', 'other income'",
    "  TAXPAYER_CATEGORY  who a rule applies to: 'resident individual',",
    "               'individual who is resident in Sri Lanka',",
    "               'resident or non-resident individual',",
    "               'senior citizen', 'individual', 'company', 'partnership', 'trust',",
    "               'charitable institution' (take the longest phrase)",
    "",
    "RULES",
    "  - Copy text exactly (same spelling, commas, 'Rs.'); spaces at the ends do not matter.",
    "  - Longest phrase wins: 'personal relief' (not also 'relief');",
    "    a reference chain is one entity.",
    "  - Only TAX_SLAB may contain other entities (its amounts).",
    "  - Unsure? Mark it and write a short note in column D.",
]


def select_chunks(chunks: list[Chunk]) -> list[Chunk]:
    by_id = {c.chunk_id: c for c in chunks}
    missing = [cid for cid in CORE if cid not in by_id]
    if missing:
        raise SystemExit(f"Core chunks not found (re-run parse/extract?): {missing}")
    picked = [by_id[cid] for cid in CORE]
    rng = random.Random(SEED)
    for doc_id in sorted({c.document_id for c in chunks}):
        pool = [
            c
            for c in chunks
            if c.document_id == doc_id
            and c.kind != "preamble"
            and c.chunk_id not in CORE
            and MIN_CHARS <= len(c.text) <= MAX_CHARS
        ]
        k = min(len(pool), RANDOM_PER_DOC.get(doc_id, RANDOM_DEFAULT))
        picked.extend(rng.sample(pool, k))
    return picked


def citation(c: Chunk) -> str:
    where = f"{c.schedule} para {c.paragraph}" if c.schedule else f"section {c.section}"
    if c.subsection:
        where += f"({c.subsection})"
    return f"{c.document_id}, {where}, page {c.page_start}"


def build_workbook(chunks: list[Chunk], path: Path) -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    guide = wb.active
    guide.title = "Guide"
    for i, line in enumerate(GUIDE, start=1):
        guide.cell(row=i, column=1, value=line).font = Font(name="Consolas", bold=i == 1)
    guide.column_dimensions["A"].width = 110

    ws = wb.create_sheet(SHEET)
    ws.append(["chunk_id", "label", "entity text (copy exactly)", "note / done"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for col, width in zip("ABCD", (34, 22, 70, 28), strict=True):
        ws.column_dimensions[col].width = width
    labels = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True)
    ws.add_data_validation(labels)

    grey = PatternFill("solid", fgColor="DDDDDD")
    for n, c in enumerate(chunks, start=1):
        ws.append([c.chunk_id, f"CHUNK {n}/{len(chunks)}", citation(c), ""])
        for cell in ws[ws.max_row]:
            cell.fill = grey
            cell.font = Font(bold=True)
        ws.append(["", "", c.text, ""])
        row = ws.max_row
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=4)
        ws.cell(row=row, column=2, value=c.text).alignment = Alignment(
            wrap_text=True, vertical="top"
        )
        ws.cell(row=row, column=3, value=None)
        ws.row_dimensions[row].height = min(409, 15 * math.ceil(len(c.text) / 120) + 6)
        for _ in range(min(40, 4 + len(c.text) // 50)):
            ws.append([c.chunk_id, "", "", ""])
            ws.cell(row=ws.max_row, column=1).font = Font(color="999999")
            labels.add(f"B{ws.max_row}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return len(chunks)


def read_annotations(path: Path) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Return ({chunk_id: text} for chunks marked done, {chunk_id: [(label, text), ...]})."""
    from openpyxl import load_workbook

    ws = load_workbook(path)[SHEET]
    texts: dict[str, str] = {}
    done: set[str] = set()
    gold: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for a, b, c, d in ws.iter_rows(min_row=2, max_col=4, values_only=True):
        if a and isinstance(b, str) and b.startswith("CHUNK "):  # grey header row
            current = a
            if str(d or "").strip().lower() == DONE:
                done.add(a)
            continue
        if not a and current and b and current not in texts:  # the chunk text row
            texts[current] = b
            continue
        if a and b and c is not None and str(c).strip():
            if b not in LABELS:
                raise SystemExit(f"Unknown label {b!r} for {a} in {path.name}")
            gold.setdefault(a, []).append((b, " ".join(str(c).split())))
    return {cid: texts[cid] for cid in texts if cid in done}, {
        cid: gold.get(cid, []) for cid in done
    }


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import COMPONENT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Create the entity annotation workbook.")
    default = COMPONENT_DIR / "ground_truth" / "entity_annotations.xlsx"
    parser.add_argument("--out", type=Path, default=default)
    parser.add_argument("--force", action="store_true", help="overwrite an existing workbook")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists (it may hold your annotations). Use --force to replace.")
        return 1
    chunks = select_chunks(load_chunks(get_settings().processed_dir))
    n = build_workbook(chunks, args.out)
    print(f"{n} chunks ({len(CORE)} core + {n - len(CORE)} random) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
