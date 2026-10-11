"""Download the corpus documents and record them in manifest.csv.

Usage (from backend/component3-graphrag, with the [ingest] extra installed):

    python -m data_collection.download                 # fetch every document in the registry
    python -m data_collection.download --only DOC-IRA-2017
    python -m data_collection.download --force         # re-download even if already present

If an official site blocks scripts, download the PDF in a browser and register it by hand:

    python -m data_collection.download --register DOC-IRA-2017 path\\to\\file.pdf

Each file is checked to really be a PDF, saved under raw_regulatory_docs/, fingerprinted with
SHA-256 and written to manifest.csv. Re-running is safe: unchanged files are left alone, and a
file that changed upstream is reported as CHANGED.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from c3_common.config import get_settings
from data_collection.manifest import ManifestRow, read_manifest, sha256_of, write_manifest
from data_collection.sources import SourceDocument, load_registry

USER_AGENT = "J26-DS-303-research/0.1 (SLIIT final-year research; regulatory corpus collection)"
TIMEOUT_S = 60


class NotAPdfError(ValueError):
    """The server answered, but not with a PDF (e.g. an HTML error or login page)."""


@dataclass
class Outcome:
    document_id: str
    status: str  # "new" | "unchanged" | "changed" | "failed"
    detail: str = ""


def make_session():
    """A requests session with retries for flaky government sites."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _row(doc: SourceDocument, path: Path, method: str) -> ManifestRow:
    return ManifestRow(
        document_id=doc.document_id,
        title=doc.title,
        document_type=doc.document_type,
        act_number=doc.act_number or "",
        certified_date=doc.certified_date.isoformat(),
        operative_from=doc.operative_from.isoformat(),
        source_url=doc.url,
        filename=doc.filename,
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        retrieved_at=_now(),
        method=method,
    )


def _record(rows: dict[str, ManifestRow], new: ManifestRow) -> str:
    old = rows.get(new.document_id)
    if old is not None and old.sha256 == new.sha256:
        return "unchanged"  # keep the original retrieved_at: it is the evidence of first collection
    rows[new.document_id] = new
    return "new" if old is None else "changed"


def fetch(doc: SourceDocument, raw_dir: Path, session) -> Path:
    """Download one document to raw_dir, atomically. Raises on HTTP errors or non-PDF content."""
    response = session.get(doc.url, timeout=TIMEOUT_S)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        kind = response.headers.get("Content-Type", "unknown")
        raise NotAPdfError(f"expected a PDF from {doc.url}, got {kind} ({len(content)} bytes)")
    target = raw_dir / doc.filename
    tmp = target.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(target)
    return target


def download_all(
    raw_dir: Path, *, only: str | None = None, force: bool = False, session=None
) -> list[Outcome]:
    registry = load_registry()
    docs = [registry.get(only)] if only else registry.documents
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(raw_dir)
    session = session or make_session()
    outcomes: list[Outcome] = []
    for doc in docs:
        existing = raw_dir / doc.filename
        if existing.is_file() and doc.document_id in rows and not force:
            outcomes.append(Outcome(doc.document_id, "unchanged", "already downloaded"))
            continue
        try:
            path = fetch(doc, raw_dir, session)
        except Exception as exc:  # report and continue with the other documents
            outcomes.append(Outcome(doc.document_id, "failed", f"{type(exc).__name__}: {exc}"))
            continue
        status = _record(rows, _row(doc, path, "download"))
        outcomes.append(Outcome(doc.document_id, status, f"{path.stat().st_size:,} bytes"))
    write_manifest(raw_dir, rows)
    return outcomes


def register_local(document_id: str, pdf_path: Path, raw_dir: Path) -> Outcome:
    """Add a PDF downloaded by hand (fallback when a site blocks scripts)."""
    doc = load_registry().get(document_id)
    if not pdf_path.read_bytes()[:4] == b"%PDF":
        raise NotAPdfError(f"{pdf_path} is not a PDF")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / doc.filename
    if pdf_path.resolve() != target.resolve():
        shutil.copyfile(pdf_path, target)
    rows = read_manifest(raw_dir)
    status = _record(rows, _row(doc, target, "manual"))
    write_manifest(raw_dir, rows)
    return Outcome(document_id, status, f"registered from {pdf_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the C3 regulatory corpus.")
    parser.add_argument("--only", metavar="DOC_ID", help="download just this document")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--register", nargs=2, metavar=("DOC_ID", "PDF"), help="register a hand-downloaded PDF"
    )
    args = parser.parse_args(argv)
    raw_dir = get_settings().raw_docs_dir

    if args.register:
        outcomes = [register_local(args.register[0], Path(args.register[1]), raw_dir)]
    else:
        outcomes = download_all(raw_dir, only=args.only, force=args.force)

    for o in outcomes:
        print(f"{o.status.upper():10} {o.document_id:24} {o.detail}")
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        print(
            f"\n{len(failed)} failed. If a site blocks scripts, download in a browser and run:\n"
            "  python -m data_collection.download --register <DOC_ID> <path-to-pdf>",
            file=sys.stderr,
        )
    print(f"\nManifest: {raw_dir / 'manifest.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
