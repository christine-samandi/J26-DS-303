"""Dataset bookkeeping for the Component 2 benchmark (Objective 2.5, Gap 6).

Why this module exists:
- Results must be reported per capture condition, language, document type and source
  (real vs synthetic). That is only possible if every image is described in one manifest.
- The train/test split must be made per DOCUMENT, never per image; otherwise photos of the
  same payslip land on both sides and the test score is inflated (data leakage).
- Ground-truth files are typed by hand, so they are checked automatically (schema, filled
  values, payslip arithmetic) before any accuracy number is computed from them.

File naming convention (data/images/):
    <DOC_ID>_<capture>.<ext>    e.g. PS_001_photo_dark.jpg, IN_001_digital.pdf
    DOC_ID  = PS (payslip) | BS (bank statement) | IN (invoice) | RC (receipt) + _ + 3 digits
    capture = digital | scan | photo_good | photo_dark | photo_skew | photo_other
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

COMPONENT = Path(__file__).resolve().parents[1]
DATA = COMPONENT / "data"
GT_SCHEMA = COMPONENT / "docs" / "ground_truth_schema.json"

DOC_TYPES = {"PS": "payslip", "BS": "bank_statement", "IN": "invoice", "RC": "receipt"}
CAPTURES = {"digital", "scan", "photo_good", "photo_dark", "photo_skew", "photo_other"}
FILE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
NAME_RE = re.compile(r"^(?P<doc_id>(?P<prefix>PS|BS|IN|RC)_\d{3})_(?P<capture>[a-z_]+?)(?:_\d+)?$")

MANIFEST_COLUMNS = ["image_file", "doc_id", "doc_type", "source", "language", "capture", "device", "split", "notes"]
MANUAL_COLUMNS = ["source", "language", "device", "split", "notes"]   # never overwritten once set

# Fields expected per document type (the rest stay out of that type's ground truth).
FIELDS_BY_TYPE: Dict[str, List[str]] = {
    "payslip": ["issuer", "pay_period", "gross_income", "epf_employee", "epf_employer", "etf", "apit",
                "total_deductions", "net_amount"],
    "bank_statement": ["issuer", "pay_period", "document_date", "total_amount"],
    "invoice": ["issuer", "document_date", "vat", "sscl", "total_amount"],
    "receipt": ["issuer", "document_date", "vat", "total_amount"],
}


@dataclass
class ParsedName:
    doc_id: str
    doc_type: str
    capture: str


def parse_image_name(filename: str) -> Optional[ParsedName]:
    """'PS_001_photo_dark.jpg' -> ParsedName('PS_001', 'payslip', 'photo_dark'); None if it does not match."""
    p = Path(filename)
    if p.suffix.lower() not in FILE_EXT:
        return None
    m = NAME_RE.match(p.stem)
    if not m:
        return None
    return ParsedName(m["doc_id"], DOC_TYPES[m["prefix"]], m["capture"])


# ---------------------------------------------------------------- manifest

def read_manifest(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image_file"]))


def assign_splits(doc_types: Dict[str, str], existing: Dict[str, str], test_every: int = 3) -> Dict[str, str]:
    """Per-document split, stratified by document type and deterministic.
    Within each type (doc ids sorted), every `test_every`-th document goes to 'test', the rest to
    'dev'; a type with >= 2 documents always gets at least one test document. Splits already in
    the manifest are kept, so adding documents later never moves existing ones."""
    splits = dict(existing)
    by_type = defaultdict(list)
    for d, t in doc_types.items():
        by_type[t].append(d)
    for t, docs in by_type.items():
        docs.sort()
        new = [d for d in docs if d not in splits]
        for i, d in enumerate(docs):
            if d in new:
                splits[d] = "test" if (i % test_every == test_every - 1) else "dev"
        if len(docs) >= 2 and not any(splits[d] == "test" for d in docs):
            candidates = [d for d in docs if d in new] or docs
            splits[candidates[-1]] = "test"
    return splits


def build_manifest(images_dir: Path, existing_rows: List[dict]) -> tuple[List[dict], List[str], List[str]]:
    """Return (rows, skipped_files, removed_images). Manual columns of existing rows are kept."""
    old = {r["image_file"]: r for r in existing_rows}
    rows, skipped = [], []
    for f in sorted(p for p in images_dir.iterdir() if p.is_file() and p.name != ".gitkeep"):
        parsed = parse_image_name(f.name)
        if parsed is None:
            skipped.append(f.name)
            continue
        row = {c: "" for c in MANIFEST_COLUMNS}
        row.update(image_file=f.name, doc_id=parsed.doc_id, doc_type=parsed.doc_type, capture=parsed.capture,
                   source="real")
        prev = old.get(f.name, {})
        for c in MANUAL_COLUMNS:
            if prev.get(c):
                row[c] = prev[c]
        rows.append(row)

    existing_splits = {r["doc_id"]: r["split"] for r in rows if r["split"]}
    splits = assign_splits({r["doc_id"]: r["doc_type"] for r in rows}, existing_splits)
    for r in rows:
        r["split"] = splits[r["doc_id"]]
    removed = sorted(set(old) - {r["image_file"] for r in rows})
    return rows, skipped, removed


# ---------------------------------------------------------------- ground truth

def ground_truth_skeleton(doc_id: str, doc_type: str, source: str = "real", languages: Optional[List[str]] = None) -> dict:
    return {
        "doc_id": doc_id,
        "document_type": doc_type,
        "source": source or "real",
        "languages": languages or ["en"],
        "fields": {f: None for f in FIELDS_BY_TYPE[doc_type]},
        "full_text": "",
        "checked": False,
        "notes": "Amounts as plain numbers (150000.00, no commas). pay_period YYYY-MM, dates YYYY-MM-DD. "
                 "null = not on the document. Set checked=true after re-checking against the paper.",
    }


@dataclass
class CheckReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counts: Dict[str, Counter] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def payslip_arithmetic(fields: dict, tol: float = 1.0) -> List[str]:
    """Catch typing mistakes in hand-made ground truth using the same rules the pipeline uses."""
    issues = []
    g, d, n = fields.get("gross_income"), fields.get("total_deductions"), fields.get("net_amount")
    if None not in (g, d, n) and abs(g - d - n) > tol:
        issues.append(f"gross - total_deductions = {g - d:.2f} but net_amount = {n:.2f}")
    return issues


def check_dataset(images_dir: Path, manifest_path: Path, gt_dir: Path, schema_path: Path = GT_SCHEMA) -> CheckReport:
    import jsonschema

    rep = CheckReport()
    rows = read_manifest(manifest_path)
    if not rows:
        rep.errors.append(f"manifest is missing or empty: {manifest_path}")
        return rep

    files = {p.name for p in images_dir.iterdir() if p.is_file() and p.name != ".gitkeep"} if images_dir.exists() else set()
    listed = {r["image_file"] for r in rows}
    for f in sorted(files - listed):
        rep.errors.append(f"image not in manifest: {f}")
    for f in sorted(listed - files):
        rep.errors.append(f"manifest row has no image: {f}")

    split_of = defaultdict(set)
    for r in rows:
        split_of[r["doc_id"]].add(r["split"])
        if r["split"] not in {"dev", "test"}:
            rep.errors.append(f"{r['image_file']}: split must be dev or test, got '{r['split']}'")
        if r["capture"] not in CAPTURES:
            rep.warnings.append(f"{r['image_file']}: unusual capture '{r['capture']}'")
        if not r.get("language"):
            rep.warnings.append(f"{r['image_file']}: language is empty (use en, si, ta, si+en, ...)")
    for d, s in split_of.items():
        if len(s) > 1:
            rep.errors.append(f"{d}: images in both dev and test (data leakage)")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    doc_type = {r["doc_id"]: r["doc_type"] for r in rows}
    for d in sorted(doc_type):
        p = gt_dir / f"{d}.json"
        if not p.exists():
            rep.errors.append(f"{d}: ground truth missing ({p.name})")
            continue
        try:
            gt = json.loads(p.read_text(encoding="utf-8"))
            jsonschema.validate(gt, schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            rep.errors.append(f"{p.name}: invalid - {getattr(e, 'message', e)}")
            continue
        if gt["doc_id"] != d or gt["document_type"] != doc_type[d]:
            rep.errors.append(f"{p.name}: doc_id/document_type does not match the manifest")
        empty = [k for k, v in gt["fields"].items() if v is None]
        if len(empty) == len(gt["fields"]):
            rep.warnings.append(f"{p.name}: no values filled in yet")
        elif empty:
            rep.warnings.append(f"{p.name}: null fields {empty} (fine if not on the document)")
        if not gt["checked"]:
            rep.warnings.append(f"{p.name}: checked=false (re-check against the paper, then set true)")
        if gt["document_type"] == "payslip":
            for issue in payslip_arithmetic(gt["fields"]):
                rep.warnings.append(f"{p.name}: {issue} - typo, or the payslip has other deductions?")
    for p in sorted(gt_dir.glob("*.json")) if gt_dir.exists() else []:
        if p.stem not in doc_type:
            rep.warnings.append(f"{p.name}: ground truth for a document with no images")

    docs = {r["doc_id"]: r for r in rows}
    rep.counts = {
        "images by capture": Counter(r["capture"] for r in rows),
        "images by language": Counter(r["language"] or "?" for r in rows),
        "documents by type": Counter(r["doc_type"] for r in docs.values()),
        "documents by split": Counter(r["split"] for r in docs.values()),
        "documents by source": Counter(r["source"] for r in docs.values()),
    }
    return rep
