"""manifest.csv: the evidence trail for every raw document in the corpus.

One row per document: where it came from, when it was fetched, and its SHA-256 fingerprint.
If a later download produces a different fingerprint, the official file has changed, which
is exactly the kind of event versioning (and C4's drift monitoring) needs to know about.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MANIFEST_NAME = "manifest.csv"


@dataclass
class ManifestRow:
    document_id: str
    title: str
    document_type: str
    act_number: str
    certified_date: str
    operative_from: str  # empty when sections take effect on different dates
    source_url: str
    filename: str
    sha256: str
    size_bytes: int
    retrieved_at: str  # ISO 8601, UTC
    method: str  # "download" (script) or "manual" (registered by hand)


COLUMNS = [f.name for f in fields(ManifestRow)]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(raw_dir: Path) -> dict[str, ManifestRow]:
    path = raw_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            ManifestRow(**{**r, "size_bytes": int(r["size_bytes"])}) for r in csv.DictReader(fh)
        ]
    return {r.document_id: r for r in rows}


def write_manifest(raw_dir: Path, rows: dict[str, ManifestRow]) -> Path:
    path = raw_dir / MANIFEST_NAME
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for doc_id in sorted(rows):
            writer.writerow(asdict(rows[doc_id]))
    tmp.replace(path)
    return path
