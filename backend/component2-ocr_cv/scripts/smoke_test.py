"""Smoke test: Tesseract on every image, without and with preprocessing.

Why: (1) proves the install and the Sinhala/Tamil language packs work,
     (2) produces the before/after evidence for PP1 (raw OCR vs preprocessed OCR, Gap 3),
     (3) shows which documents and capture conditions are hardest.
It reports OCR *confidence*, not accuracy; accuracy (CER/WER) needs ground truth and
comes from the benchmark script.

Usage:
    python scripts/smoke_test.py --images data/images --out outputs/smoke --lang sin+tam+eng --mode both
Outputs per image and mode (raw | pre):
    <name>_<mode>.txt         recognised text
    <name>_<mode>_boxes.jpg   word boxes drawn on the ORIGINAL photo (green >=0.95, orange >=0.70, red <0.70)
    <name>_pre_clean.png      the preprocessed image (for before/after slides)
    summary.csv               words, mean confidence, share of low-confidence words, seconds, steps applied
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_engines.tesseract_engine import run_tesseract, tokens_to_text  # noqa: E402
from preprocessing import DEFAULT_STEPS, preprocess  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(path: Path):
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path  # needs Poppler; only imported for PDFs
        for k, page in enumerate(convert_from_path(str(path), dpi=300), 1):
            yield f"{path.stem}_p{k}", cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    elif path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is not None:
            yield path.stem, img


def colour(conf):
    if conf >= 0.95:
        return (0, 160, 0)
    if conf >= 0.70:
        return (0, 165, 255)
    return (0, 0, 255)


def run_one(img, mode, lang):
    """Return tokens with bboxes in ORIGINAL-image coordinates, plus the cleaned image."""
    if mode == "raw":
        return run_tesseract(img, lang=lang), None, []
    res = preprocess(img, steps=DEFAULT_STEPS)
    tokens = run_tesseract(res.image, lang=lang)
    for t in tokens:
        t.bbox = res.to_original_bbox(t.bbox)
    return tokens, res.image, res.steps_applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--lang", default="sin+tam+eng")
    ap.add_argument("--mode", choices=["raw", "pre", "both"], default="both")
    args = ap.parse_args()

    images = Path(args.images)
    images.mkdir(parents=True, exist_ok=True)   # data/ and outputs/ are git-ignored
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images.iterdir() if p.is_file() and p.name != ".gitkeep")
    if not files:
        sys.exit(f"No files found in {images} - add images there and re-run")
    modes = ["raw", "pre"] if args.mode == "both" else [args.mode]

    rows = []
    for f in files:
        for name, img in load_images(f):
            for mode in modes:
                t0 = time.time()
                tokens, clean, steps = run_one(img, mode, args.lang)
                secs = time.time() - t0
                (out / f"{name}_{mode}.txt").write_text(tokens_to_text(tokens), encoding="utf-8")
                vis = img.copy()
                for t in tokens:
                    x, y, w, h = t.bbox
                    cv2.rectangle(vis, (x, y), (x + w, y + h), colour(t.confidence), 2)
                cv2.imwrite(str(out / f"{name}_{mode}_boxes.jpg"), vis)
                if clean is not None:
                    cv2.imwrite(str(out / f"{name}_pre_clean.png"), clean)
                n = len(tokens)
                mean = sum(t.confidence for t in tokens) / n if n else 0
                low = sum(t.confidence < 0.70 for t in tokens) / n if n else 0
                rows.append({"image": name, "mode": mode, "words": n, "mean_conf": round(mean, 3),
                             "low_conf_share": round(low, 3), "seconds": round(secs, 2),
                             "steps": "+".join(steps)})
                print(f"{name:32s} {mode:4s} words={n:4d} mean_conf={mean:.2f} low_conf={low:.0%} {secs:.1f}s")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out}/")


if __name__ == "__main__":
    main()
