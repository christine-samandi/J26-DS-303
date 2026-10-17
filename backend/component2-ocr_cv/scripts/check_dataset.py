"""Check the dataset before any accuracy number is computed.

Why: a wrong ground-truth value or a document split across dev and test silently corrupts
every result. This prints errors (must fix), warnings (should look at) and the counts you
will report in the PP1 dataset slide.

Usage (from backend/component2-ocr_cv):
    python scripts/check_dataset.py
Exit code 0 = no errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset import DATA, check_dataset  # noqa: E402


def main() -> int:
    rep = check_dataset(DATA / "images", DATA / "manifest.csv", DATA / "ground_truth")
    for name, counter in rep.counts.items():
        print(f"{name:20s} " + ", ".join(f"{k}={v}" for k, v in sorted(counter.items())))
    if rep.warnings:
        print(f"\nWARNINGS ({len(rep.warnings)})")
        for w in rep.warnings:
            print("  -", w)
    if rep.errors:
        print(f"\nERRORS ({len(rep.errors)})")
        for e in rep.errors:
            print("  x", e)
    print("\nDataset OK" if rep.ok else "\nFix the errors above")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
