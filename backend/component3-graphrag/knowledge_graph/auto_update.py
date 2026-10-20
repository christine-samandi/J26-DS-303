"""Automatic update: a new Act found by the law monitor is added to the graph by itself,
but only when it passes every safety check. Nobody has to check it by hand.

The checks replace the person: each one stops a way a document could make the tax answers
wrong. If any check fails, NOTHING changes (the registry, the files and the graph stay as they
were, so the old law keeps answering) and the reason is recorded for the maintainer.

    1  is a PDF                     the link really returned a PDF
    2  is an income tax Act         the cover page says "Inland Revenue (Amendment) Act, No. N
                                    of YYYY" and has a certified date (not VAT, not a gazette,
                                    not a Sinhala or Tamil copy)
    3  not already in the corpus    the same Act under another link is skipped
    4  reads correctly              the parser finds its sections, without warnings
    5  no rule disappears           every rule version in the graph is still in the timeline
    6  history unchanged            every hand-verified version (ground_truth/) is still rebuilt
                                    exactly; only an open-ended version may now get an end date
    7  values make sense            amounts and caps above 0 and below Rs. 1,000 million, rates
                                    above 0 and at most 100%, tax bands in increasing order

Then it is written to Neo4j (knowledge_graph.update), the change is recorded (GET
/notifications) and the running API is told to reload. Used by data_collection.monitor; also
on its own:

    python -m knowledge_graph.auto_update <PDF-URL>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

_TITLE = re.compile(r"INLAND\s+REVENUE\s+\(AMENDMENT\)\s+ACT,?\s+NO\.?\s*(\d+)\s+OF\s+(\d{4})")
MAX_LKR = 1_000_000_000


@dataclass
class Outcome:
    status: str  # added | held | skipped
    reason: str = ""
    document_id: str | None = None
    checks: list[str] = field(default_factory=list)  # each passed check, for the log
    changes: int = 0


class Held(Exception):
    """A safety check failed: the document is not added."""


def identify(title: str | None, certified: str | None) -> tuple[str, str, str, str]:
    """(document_id, title, act_number, filename) from the cover page, or Held."""
    m = _TITLE.search((title or "").upper())
    if not m or not certified:
        raise Held(f"not an Inland Revenue (Amendment) Act (cover page: {title!r})")
    number, year = int(m[1]), m[2]
    return (f"DOC-IRA-AMEND-{number:02d}-{year}",
            f"Inland Revenue (Amendment) Act, No. {number} of {year}", f"{number} of {year}",
            f"IRA_AMEND_{number:02d}_{year}_E.pdf")  # fmt: skip


def check_history(versions: dict, gold: dict[str, list[dict]]) -> list[str]:
    """Check 6: problems with the hand-verified versions (empty = all still exact)."""
    problems = []
    for key, rows in gold.items():
        built = [v.to_dict() for v in versions.get(key, [])]
        for i, g in enumerate(rows):
            v = built[i] if i < len(built) else None
            if v is None:
                problems.append(f"{key} V{i + 1} is missing")
                continue
            same = (v["value"] == g["value"] and v["valid_from"] == g["valid_from"]
                    and v["source_document_id"] == g["source"])  # fmt: skip
            end_ok = v["valid_to"] == g["valid_to"] or (
                g["valid_to"] is None
                and v["valid_to"] is not None
                and v["valid_to"] >= g["valid_from"]
            )  # an open version may be closed by the new Act, nothing else may change
            if not (same and end_ok):
                problems.append(f"{key} V{i + 1} would change")
    return problems


def check_values(versions: dict) -> list[str]:
    """Check 7: values outside what a tax rule can be."""
    bad = []
    for vs in versions.values():
        for v in vs:
            if v.value_kind in {"amount", "cap"} and not 0 < v.value < MAX_LKR:
                bad.append(f"{v.rule_id}: amount {v.value}")
            elif v.value_kind == "rate" and not 0 < v.value <= 100:
                bad.append(f"{v.rule_id}: rate {v.value}%")
            elif v.value_kind == "bands":
                lowers = [b[0] for b in v.value]
                if lowers != sorted(lowers) or not all(0 < b[2] <= 100 for b in v.value):
                    bad.append(f"{v.rule_id}: bands out of order or rates outside 0-100%")
    return bad


def _version_rows(versions: dict) -> list[dict]:
    rows = []
    for vs in versions.values():
        for v in vs:
            d = v.to_dict()
            rows.append({"rule_id": v.rule_id, "rule_key": v.rule_key, "status": v.status,
                         "valid_from": d["valid_from"], "valid_to": d["valid_to"],
                         "source": v.source_document_id,
                         "amount": v.value if v.value_kind == "amount" else None,
                         "cap": v.value if v.value_kind == "cap" else None,
                         "rate_percent": v.value if v.value_kind == "rate" else None,
                         "bands": v.value if v.value_kind == "bands" else None})  # fmt: skip
    return rows


def ingest(url: str, fetch_bytes, apply=None, reload=None, now: datetime | None = None
           ) -> Outcome:  # fmt: skip
    """Run the checks on the PDF at `url` and add it when they all pass."""
    from c3_common.config import COMPONENT_DIR, get_settings
    from data_collection.download import _record, _row
    from data_collection.manifest import read_manifest, write_manifest
    from data_collection.parse import parse_corpus, parse_pdf
    from data_collection.sources import SourceDocument, add_document, load_registry, registry_path
    from entity_recognition.chunking import load_chunks
    from entity_recognition.extract import extract_corpus
    from knowledge_graph.update import apply_update, version_changes
    from knowledge_graph.versions import _gold, build_versions

    apply = apply or (lambda ids: apply_update(ids, skip_parse=True))
    now = now or datetime.now(UTC)
    settings = get_settings()
    raw, processed, extracted = (
        settings.raw_docs_dir,
        settings.processed_dir,
        settings.extracted_dir,
    )
    out = Outcome("held")
    incoming = raw / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / "candidate.pdf"
    backups: dict[Path, bytes | None] = {}
    written: list[Path] = []

    def backup(path: Path) -> None:
        backups.setdefault(path, path.read_bytes() if path.is_file() else None)

    try:
        content = fetch_bytes(url)
        if not content.startswith(b"%PDF"):
            raise Held("the link did not return a PDF")
        out.checks.append("1 is a PDF")
        tmp.write_bytes(content)

        parsed = parse_pdf(tmp, "DOC-INCOMING")
        doc_id, title, act_number, filename = identify(parsed.title_detected,
                                                       parsed.certified_date_detected)  # fmt: skip
        out.document_id = doc_id
        out.checks.append(f"2 is an income tax Act: {title}")
        if doc_id in {d.document_id for d in load_registry().documents}:
            out.status, out.reason = "skipped", f"{title} is already in the corpus"
            return out
        out.checks.append("3 not already in the corpus")
        if not any(s.kind == "section" for s in parsed.segments) or parsed.warnings:
            raise Held(
                f"the PDF did not read cleanly ({'; '.join(parsed.warnings) or 'no sections'})"
            )
        out.checks.append(
            f"4 reads correctly: {sum(s.kind == 'section' for s in parsed.segments)} sections"
        )

        # register it (undone below if a later check fails)
        doc = SourceDocument(
            document_id=doc_id, title=title, document_type="amendment_act",
            act_number=act_number,
            certified_date=date.fromisoformat(parsed.certified_date_detected),
            operative_from=None, url=url, filename=filename,
            relevance=f"added automatically by the law monitor on {now.date()}",
        )  # fmt: skip
        reg, manifest = registry_path(), raw / "manifest.csv"
        for p in (reg, manifest):
            backup(p)
        target = raw / filename
        shutil.move(str(tmp), target)
        written.append(target)
        add_document(doc, reg)
        rows = read_manifest(raw)
        _record(rows, _row(doc, target, "monitor"))
        write_manifest(raw, rows)
        parse_corpus(raw, processed, doc_id)
        written.append(processed / f"{doc_id}.json")
        extract_corpus(processed, extracted, doc_id)
        written.append(extracted / f"{doc_id}.json")

        chunks = load_chunks(processed)
        before = build_versions([c for c in chunks if c.document_id != doc_id])
        after = build_versions(chunks)
        changes = version_changes(_version_rows(before), after)
        if removed := [c.rule_id for c in changes if c.kind == "-"]:
            raise Held(f"rule versions would disappear: {', '.join(removed)}")
        out.checks.append("5 no rule disappears")
        if problems := check_history(after, _gold(COMPONENT_DIR / "ground_truth")):
            raise Held("the verified history would change: " + "; ".join(problems))
        out.checks.append("6 history unchanged")
        if bad := check_values(after):
            raise Held("values outside what a tax rule can be: " + "; ".join(bad))
        out.checks.append("7 values make sense")

        try:
            applied = apply([doc_id])
        except Exception as e:  # Neo4j down: keep the old state, try again next month
            raise Held(f"the graph could not be updated ({type(e).__name__}: {e})") from e
        out.status, out.changes = "added", len(applied or [])
        out.reason = f"{title}: {out.changes} rule version(s) changed"
        if reload:
            reload()
        return out
    except Exception as e:  # a failed check (Held) or anything unexpected: change nothing
        out.status = "held"
        out.reason = str(e) if isinstance(e, Held) else f"{type(e).__name__}: {e}"
        for path, data in backups.items():  # put everything back as it was
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        for path in written:
            path.unlink(missing_ok=True)
        load_registry.cache_clear()
        return out
    finally:
        tmp.unlink(missing_ok=True)


def reload_api(base_url: str | None = None) -> None:
    """Tell a running C3 API to read the new graph (best effort)."""
    import os
    import urllib.request

    base = (base_url or os.getenv("C3_API_URL") or "http://localhost:8003").rstrip("/")
    try:
        urllib.request.urlopen(urllib.request.Request(f"{base}/reload", method="POST"), timeout=30)
        print(f"  API reloaded ({base})")
    except OSError as e:
        print(f"  API not reloaded ({e}); restart it or POST {base}/reload")


def download(url: str) -> bytes:
    from data_collection.download import TIMEOUT_S, make_session

    response = make_session().get(url, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add a new Act automatically, if it is safe.")
    parser.add_argument("url")
    args = parser.parse_args(argv)
    out = ingest(args.url, download, reload=reload_api)
    for c in out.checks:
        print(f"  ok  {c}")
    print(f"{out.status.upper()}: {out.reason}")
    return 0 if out.status != "held" else 2


if __name__ == "__main__":
    sys.exit(main())
