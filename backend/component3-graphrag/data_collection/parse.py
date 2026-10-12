"""Parse the downloaded Acts into structured JSON (sections, subsections, schedules).

Usage (after `python -m data_collection.download`):

    python -m data_collection.parse                     # every document in manifest.csv
    python -m data_collection.parse --only DOC-IRA-AMEND-02-2025

Writes data/processed/<document_id>.json and prints a summary per document. The SHA-256 in
manifest.csv is checked first, so a JSON file always matches the exact PDF it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from c3_common.config import get_settings
from data_collection.layout import body_rows, classify, margin_notes, read_pages
from data_collection.manifest import read_manifest, sha256_of
from data_collection.segment import (
    ParsedDocument,
    check_against_registry,
    detect_certified_date,
    detect_title,
    segment,
)
from data_collection.sources import load_registry


def parse_pdf(pdf_path: Path, document_id: str) -> ParsedDocument:
    pages = read_pages(pdf_path)
    classify(pages)
    rows = body_rows(pages)
    notes = margin_notes(pages)
    segments, warnings = segment(document_id, rows, notes)
    all_text = "\n".join(ln.text for p in pages for ln in p.lines)
    return ParsedDocument(
        document_id=document_id,
        source_file=pdf_path.name,
        sha256=sha256_of(pdf_path),
        page_count=len(pages),
        title_detected=detect_title(rows),
        certified_date_detected=detect_certified_date(all_text),
        segments=segments,
        warnings=warnings,
    )


def parse_corpus(raw_dir: Path, out_dir: Path, only: str | None = None) -> list[ParsedDocument]:
    registry = load_registry()
    manifest = read_manifest(raw_dir)
    if not manifest:
        raise SystemExit(f"No manifest in {raw_dir}. Run: python -m data_collection.download")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for doc_id, row in sorted(manifest.items()):
        if only and doc_id != only:
            continue
        pdf = raw_dir / row.filename
        parsed = parse_pdf(pdf, doc_id)
        if parsed.sha256 != row.sha256:
            parsed.warnings.append("file differs from manifest.csv: re-run the download step")
        source = registry.get(doc_id)
        check_against_registry(parsed, source.certified_date, source.act_number)
        (out_dir / f"{doc_id}.json").write_text(
            json.dumps(parsed.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results.append(parsed)
    return results


def summarise(parsed: ParsedDocument) -> str:
    kinds = [s.kind for s in parsed.segments]
    schedules = list(dict.fromkeys(s.schedule for s in parsed.segments if s.schedule))
    noted = sum(1 for s in parsed.segments if s.marginal_note)
    lines = [
        f"{parsed.document_id}: {parsed.page_count} pages, "
        f"{kinds.count('section')} sections, {kinds.count('schedule_paragraph')} schedule "
        f"paragraphs, {noted} margin notes matched",
        f"  title: {parsed.title_detected}",
        f"  certified: {parsed.certified_date_detected}",
    ]
    if schedules:
        lines.append(f"  schedules: {', '.join(schedules)}")
    lines += [f"  WARNING: {w}" for w in parsed.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse downloaded Acts into JSON segments.")
    parser.add_argument("--only", metavar="DOC_ID")
    args = parser.parse_args(argv)
    settings = get_settings()
    results = parse_corpus(settings.raw_docs_dir, settings.processed_dir, args.only)
    for parsed in results:
        print(summarise(parsed) + "\n")
    print(f"JSON written to {settings.processed_dir}")
    return 1 if any(p.warnings for p in results) else 0


if __name__ == "__main__":
    sys.exit(main())
