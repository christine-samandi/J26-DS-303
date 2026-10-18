"""Benchmark the full pipeline against hand-checked ground truth (Objective 2.5, Gap 6).

Why this script exists:
- PP1 needs measured accuracy on REAL documents, not impressions from one demo image.
- It answers the research questions directly: does preprocessing help, and on which capture
  conditions (scan vs dark/skewed phone photo)? Which engine reads better? Is the confidence
  honest enough that "needs_review" catches the wrong values?
- Every engine runs through the SAME preprocessing, classifier, extraction and fiscal checks, so
  differences come from the reading step only.

Usage (from backend/component2-ocr_cv):
    python scripts/benchmark.py                         # all images, tesseract, raw + preprocessed
    python scripts/benchmark.py --split test            # only the held-out test documents
    python scripts/benchmark.py --engines tesseract --modes pre --lang eng
Outputs (git-ignored): outputs/benchmark/per_field.csv, per_image.csv, summary.csv, summary.md.
The console shows only counts and scores, never document values (NFR4).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.pipeline import UnsupportedFile, available_languages, decode_upload  # noqa: E402
from entity_extraction.payslip import extract_document  # noqa: E402
from ocr_engines.tesseract_engine import run_tesseract, tokens_to_text  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from scripts.dataset import DATA, read_manifest  # noqa: E402
from scripts.metrics import (FieldResult, cer_wer, expected_calibration_error,  # noqa: E402
                             precision_recall_f1, review_quality, score_fields)

# engine name -> function(image, lang) -> tokens. PaddleOCR is added here later; nothing else changes.
ENGINES: Dict[str, Callable] = {"tesseract": lambda img, lang: run_tesseract(img, lang=lang)}
MODES = ("raw", "pre")
LANG_CODES = {"en": "eng", "si": "sin", "ta": "tam"}


def tesseract_lang(manifest_language: str, installed: List[str]) -> str:
    """'si+en' -> 'sin+eng'. English is always added: amounts and many labels are printed in English."""
    codes = [LANG_CODES[c] for c in (manifest_language or "").replace(",", "+").split("+") if c in LANG_CODES]
    codes = list(dict.fromkeys(codes + ["eng"]))
    return "+".join(c for c in codes if c in installed) or "eng"


def run_one(img, engine: str, mode: str, lang: str, doc_id: str):
    """Same steps as api.pipeline.run_pipeline, but keeps the tokens for CER/WER."""
    t0 = time.time()
    to_original = None
    if mode == "pre":
        res = preprocess(img)
        tokens = ENGINES[engine](res.image, lang)
        to_original = res.to_original_bbox
    else:
        tokens = ENGINES[engine](img, lang)
    ms = int((time.time() - t0) * 1000)
    out, _ = extract_document(tokens, doc_id=doc_id, engine=engine, processing_ms=ms, to_original=to_original)
    return out, tokens, ms


def summarise(results: List[FieldResult], images: List[dict]) -> dict:
    s = precision_recall_f1(results)
    pairs = [(r.confidence, r.status == "correct") for r in results if r.confidence is not None]
    s["ece"] = expected_calibration_error(pairs)
    s.update(review_quality(results))
    s["images"] = len(images)
    s["doc_type_acc"] = round(sum(i["type_ok"] for i in images) / len(images), 4) if images else None
    cers = [i["cer"] for i in images if i["cer"] != ""]
    wers = [i["wer"] for i in images if i["wer"] != ""]
    s["cer"] = round(sum(cers) / len(cers), 4) if cers else None
    s["wer"] = round(sum(wers) / len(wers), 4) if wers else None
    s["mean_ms"] = int(sum(i["ms"] for i in images) / len(images)) if images else None
    return s


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default="outputs/benchmark")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--engines", default="tesseract", help="comma list, e.g. tesseract,paddle")
    ap.add_argument("--modes", default="raw,pre", help="raw (no preprocessing), pre (preprocessing on)")
    ap.add_argument("--lang", default="auto", help="'auto' = from the manifest language column, or e.g. sin+tam+eng")
    args = ap.parse_args(argv)

    data = Path(args.data)
    rows = read_manifest(data / "manifest.csv")
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        print(f"No images in {data / 'manifest.csv'} (split={args.split}). Run make_manifest.py first.")
        return 1
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        print(f"Unknown engine(s) {unknown}; available: {sorted(ENGINES)}")
        return 1
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    installed = available_languages()

    field_rows, image_rows = [], []
    by_key: Dict[tuple, List[FieldResult]] = defaultdict(list)
    images_by_key: Dict[tuple, List[dict]] = defaultdict(list)

    for r in rows:
        gt_path = data / "ground_truth" / f"{r['doc_id']}.json"
        if not gt_path.exists():
            print(f"skip {r['image_file']}: no ground truth")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        try:
            img, _ = decode_upload((data / "images" / r["image_file"]).read_bytes(), r["image_file"])
        except (UnsupportedFile, FileNotFoundError) as e:
            print(f"skip {r['image_file']}: {e}")
            continue
        lang = tesseract_lang(r.get("language", ""), installed) if args.lang == "auto" else args.lang

        for engine in engines:
            for mode in modes:
                out, tokens, ms = run_one(img, engine, mode, lang, r["doc_id"])
                pred = {f.field_name: (f.value, f.confidence, f.needs_review) for f in out.extracted_fields}
                results = score_fields(pred, gt["fields"])
                cer, wer = cer_wer(gt.get("full_text", ""), tokens_to_text(tokens))
                img_row = {"image_file": r["image_file"], "doc_id": r["doc_id"], "doc_type": r["doc_type"],
                           "capture": r["capture"], "source": r.get("source", ""), "split": r["split"],
                           "engine": engine, "mode": mode, "lang": lang,
                           "predicted_type": out.document_type, "type_ok": out.document_type == r["doc_type"],
                           "fields_correct": sum(x.status == "correct" for x in results),
                           "fields_expected": sum(v is not None for v in gt["fields"].values()),
                           "cer": "" if cer is None else cer, "wer": "" if wer is None else wer, "ms": ms}
                image_rows.append(img_row)
                for x in results:
                    field_rows.append({"image_file": r["image_file"], "doc_id": r["doc_id"], "engine": engine,
                                       "mode": mode, "capture": r["capture"], "field": x.field, "status": x.status,
                                       "confidence": x.confidence, "needs_review": x.needs_review,
                                       "predicted": x.predicted, "expected": x.expected})
                for key in [(engine, mode, "all", "all"), (engine, mode, "capture", r["capture"]),
                            (engine, mode, "doc_type", r["doc_type"]), (engine, mode, "source", r.get("source", ""))]:
                    by_key[key] += results
                    images_by_key[key].append(img_row)
                print(f"{r['image_file']:32s} {engine:10s} {mode:4s} type={'ok ' if img_row['type_ok'] else 'NO '}"
                      f"fields {img_row['fields_correct']}/{img_row['fields_expected']}  {ms} ms")

    if not image_rows:
        print("Nothing was benchmarked.")
        return 1

    summary = []
    for key in sorted(by_key):
        s = summarise(by_key[key], images_by_key[key])
        summary.append({"engine": key[0], "mode": key[1], "group": key[2], "value": key[3], **s})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data_rows in [("per_field.csv", field_rows), ("per_image.csv", image_rows), ("summary.csv", summary)]:
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data_rows[0].keys()))
            w.writeheader()
            w.writerows(data_rows)

    cols = ["engine", "mode", "group", "value", "images", "precision", "recall", "f1", "doc_type_acc",
            "cer", "wer", "ece", "trusted_wrong", "mean_ms"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if s[c] is None else str(s[c]) for c in cols) + " |" for s in summary]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nOverall (field-level):")
    for s in summary:
        if s["group"] == "all":
            print(f"  {s['engine']:10s} {s['mode']:4s} P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} "
                  f"type_acc={s['doc_type_acc']} CER={s['cer']} ECE={s['ece']} trusted_but_wrong={s['trusted_wrong']}")
    print(f"\nSaved {len(image_rows)} runs to {out_dir}/ (summary.md is ready for slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
