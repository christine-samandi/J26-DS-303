"""Run the full Component 2 pipeline on one document and show the JSON plus the explanation.

Why: one command that demonstrates every stage - preprocessing, OCR, field extraction, fiscal
self-correction and confidence - on any image. Used for checking and for the PP1 demo.

Usage (from backend/component2-ocr_cv):
    python scripts/extract_demo.py sample_documents/SYN_payslip_dark_skew.jpg
    python scripts/extract_demo.py data/images/PS_001_photo_dark.jpg --lang sin+eng --raw
Writes outputs/extract_demo/<name>.json (contract output) and <name>_report.json (checks).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entity_extraction.payslip import extract_payslip, to_contract_json  # noqa: E402
from ocr_engines.tesseract_engine import run_tesseract  # noqa: E402
from preprocessing import preprocess  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--lang", default="sin+tam+eng")
    ap.add_argument("--raw", action="store_true", help="skip preprocessing (for comparison)")
    ap.add_argument("--out", default="outputs/extract_demo")
    args = ap.parse_args()

    path = Path(args.image)
    img = cv2.imread(str(path))
    if img is None:
        print(f"Cannot read image: {path}")
        return 1
    t0 = time.time()
    to_original = None
    if args.raw:
        tokens = run_tesseract(img, lang=args.lang)
    else:
        res = preprocess(img)
        tokens = run_tesseract(res.image, lang=args.lang)   # boxes stay in straightened coordinates
        to_original = res.to_original_bbox                  # final boxes are mapped back to the photo
    result, report = extract_payslip(tokens, doc_id=path.stem, processing_ms=int((time.time() - t0) * 1000),
                                     to_original=to_original)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{path.stem}.json").write_text(to_contract_json(result), encoding="utf-8")
    (out / f"{path.stem}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{path.name}  ({'raw' if args.raw else 'preprocessed'}, {result.processing_ms} ms)")
    print(f"{'field':18s}{'value':>16s}  {'conf':>5s}  review  OCR text")
    for f in result.extracted_fields:
        v = f"{f.value:,.2f}" if isinstance(f.value, float) else str(f.value)
        print(f"{f.field_name:18s}{v[:16]:>16s}  {f.confidence:5.2f}  {'YES' if f.needs_review else ' - ':6s}  {f.raw_text}")
    print("\nArithmetic checks:")
    for c in report["checks"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['rule']}{'  (hard rule)' if c['hard'] else ''}")
    for f, c in report["corrections"].items():
        print(f"  CORRECTED {f}: OCR read {c['ocr_value']:,.2f} -> {c['corrected_value']:,.2f}")
    print(f"\noverall confidence {result.overall_confidence}  needs review: {result.needs_review}")
    print(f"saved {out / (path.stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
