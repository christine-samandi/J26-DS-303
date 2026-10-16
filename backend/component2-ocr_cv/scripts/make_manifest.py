"""Build or update data/manifest.csv from the image file names in data/images/.

Why: one row per image (document type, capture condition, language, split) is what lets the
benchmark report results per condition and per language. Re-running is safe: columns you
filled by hand (source, language, device, split, notes) are kept.

Usage (from backend/component2-ocr_cv):
    python scripts/make_manifest.py
Then open data/manifest.csv and fill the `language` column (en, si, ta, si+en, ta+en, si+ta+en)
and `device` (phone model or scanner).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset import DATA, build_manifest, read_manifest, write_manifest  # noqa: E402


def main() -> int:
    images, manifest = DATA / "images", DATA / "manifest.csv"
    images.mkdir(parents=True, exist_ok=True)
    rows, skipped, removed = build_manifest(images, read_manifest(manifest))
    if not rows:
        print(f"No correctly named images in {images}. Example name: PS_001_photo_dark.jpg")
        return 1
    write_manifest(manifest, rows)
    docs = {r["doc_id"]: r for r in rows}
    print(f"Wrote {manifest} : {len(rows)} images, {len(docs)} documents")
    print("  documents by split:", dict(Counter(r["split"] for r in docs.values())))
    print("  images by capture: ", dict(Counter(r["capture"] for r in rows)))
    for s in skipped:
        print(f"  SKIPPED (name does not follow DOC_ID_capture.ext): {s}")
    for r in removed:
        print(f"  REMOVED (image no longer in data/images): {r}")
    empty_lang = sum(1 for r in rows if not r["language"])
    if empty_lang:
        print(f"  TODO: fill the language column for {empty_lang} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
