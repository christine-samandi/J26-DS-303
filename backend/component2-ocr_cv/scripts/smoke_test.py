"""Day-1 smoke test: run raw Tesseract (no preprocessing) on every image in a folder.

Why: (1) proves the install and Sinhala/Tamil language packs work,
     (2) gives the BASELINE - the "before" evidence for PP1 (raw OCR is messy),
     (3) shows which documents and capture conditions are hardest.

Usage:
    python scripts/smoke_test.py --images data/images --out outputs/smoke --lang sin+tam+eng
Outputs per image: <name>.txt (raw text), <name>_boxes.jpg (word boxes coloured by confidence),
plus summary.csv (word count, mean confidence, share of low-confidence words, seconds).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from pdf2image import convert_from_path  # noqa: F401  (only needed for PDFs)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_engines.tesseract_engine import run_tesseract, tokens_to_text  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(path: Path):
    if path.suffix.lower() == ".pdf":
        import numpy as np
        for k, page in enumerate(convert_from_path(str(path), dpi=300), 1):
            yield f"{path.stem}_p{k}", cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    elif path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is not None:
            yield path.stem, img


def colour(conf):
    if conf >= 0.95:
        return (0, 160, 0)      # green: trusted
    if conf >= 0.70:
        return (0, 165, 255)    # orange: uncertain
    return (0, 0, 255)          # red: likely wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--lang", default="sin+tam+eng")
    args = ap.parse_args()

    # data/ and outputs/ are git-ignored, so a fresh clone may not have them
    images = Path(args.images)
    images.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(p for p in images.iterdir() if p.is_file() and p.name != ".gitkeep")
    if not files:
        sys.exit(f"No files found in {images} - add images there and re-run")

    for f in files:
        for name, img in load_images(f):
            t0 = time.time()
            tokens = run_tesseract(img, lang=args.lang)
            secs = time.time() - t0

            (out / f"{name}.txt").write_text(tokens_to_text(tokens), encoding="utf-8")
            vis = img.copy()
            for t in tokens:
                x, y, w, h = t.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), colour(t.confidence), 2)
            cv2.imwrite(str(out / f"{name}_boxes.jpg"), vis)

            n = len(tokens)
            mean = sum(t.confidence for t in tokens) / n if n else 0
            low = sum(t.confidence < 0.70 for t in tokens) / n if n else 0
            rows.append({"image": name, "words": n, "mean_conf": round(mean, 3),
                         "low_conf_share": round(low, 3), "seconds": round(secs, 2)})
            print(f"{name:40s} words={n:4d} mean_conf={mean:.2f} low_conf={low:.0%} {secs:.1f}s")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out}/")


if __name__ == "__main__":
    main()
