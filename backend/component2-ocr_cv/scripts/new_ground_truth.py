"""Create an empty ground-truth JSON for every document in the manifest that has none yet.

Why: the ground truth is the answer key the pipeline is scored against. Generating the file
with the right fields for each document type avoids typos in field names; you only type values.
Existing files are never overwritten.

Usage (from backend/component2-ocr_cv):
    python scripts/new_ground_truth.py
Then open each data/ground_truth/<DOC_ID>.json, type the values from the paper document,
and set "checked": true after re-checking.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset import DATA, ground_truth_skeleton, read_manifest  # noqa: E402


def lang_list(value: str) -> list:
    langs = [x for x in value.replace(" ", "").split("+") if x in {"en", "si", "ta"}]
    return langs or ["en"]


def main() -> int:
    rows = read_manifest(DATA / "manifest.csv")
    if not rows:
        print("Run scripts/make_manifest.py first.")
        return 1
    gt_dir = DATA / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for doc_id, r in sorted({r["doc_id"]: r for r in rows}.items()):
        p = gt_dir / f"{doc_id}.json"
        if p.exists():
            continue
        gt = ground_truth_skeleton(doc_id, r["doc_type"], r["source"], lang_list(r["language"]))
        p.write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created += 1
        print(f"  created {p.name}")
    print(f"{created} new ground-truth files in {gt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
