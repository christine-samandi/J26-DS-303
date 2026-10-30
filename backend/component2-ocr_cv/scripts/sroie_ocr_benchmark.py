"""SROIE public-benchmark check for the OCR stage (raw vs preprocessed).

Why this exists: every other Component 2 result comes from our own small Sri Lankan set.
This script measures the preprocessing + Tesseract stage on a public, labelled benchmark
(ICDAR 2019 SROIE receipts), so the effect of preprocessing can be compared with published
work. It does NOT test field extraction or fiscal checks (SROIE has no Sri Lankan tax fields).

Data: the corrected SROIE train/val set from github.com/zzzDavid/ICDAR-2019-SROIE (MIT),
folder data/ with img/*.jpg, box/*.csv (x1..y4,text per line) and key/*.json.
Keep it under data/public/sroie/ (git-ignored). Never commit it.

Usage (from backend/component2-ocr_cv):
    python scripts/sroie_ocr_benchmark.py --sroie data/public/sroie/data --n 50
Outputs: outputs/sroie/per_image.csv and outputs/sroie/summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import jiwer
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_preprocess():
    """Import the component's own preprocessing; return a function image -> image."""
    from preprocessing import preprocess  # component2-ocr_cv/preprocessing

    def run(img):
        res = preprocess(img)
        for attr in ("image", "processed", "output"):
            if hasattr(res, attr):
                return getattr(res, attr)
        return res  # function returned the image itself

    return run


def reference_text(box_file: Path) -> str:
    """Ground-truth lines sorted top-to-bottom, then left-to-right (reading order)."""
    lines = []
    for row in box_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = row.split(",", 8)
        if len(parts) < 9 or not parts[8].strip():
            continue
        x, y = int(parts[0]), int(parts[1])
        lines.append((y, x, parts[8].strip()))
    lines.sort(key=lambda t: (round(t[0] / 10), t[1]))
    return "\n".join(t[2] for t in lines)


def norm(text: str) -> str:
    """Upper-case and collapse spaces: SROIE transcripts are upper-case."""
    return re.sub(r"[ \t]+", " ", text.upper()).strip()


def word_recall(ref: str, hyp: str) -> float:
    """Share of reference words found anywhere in the OCR output (order-free)."""
    ref_words = ref.split()
    hyp_words = {}
    for w in hyp.split():
        hyp_words[w] = hyp_words.get(w, 0) + 1
    hit = 0
    for w in ref_words:
        if hyp_words.get(w, 0) > 0:
            hit += 1
            hyp_words[w] -= 1
    return hit / len(ref_words) if ref_words else 0.0


def key_hits(key_file: Path, hyp: str) -> tuple[int, int]:
    """How many of the key values (date, total) appear verbatim in the OCR output."""
    if not key_file.exists():
        return 0, 0
    key = json.loads(key_file.read_text(encoding="utf-8"))
    flat = hyp.replace(" ", "")
    vals = [key.get(k, "") for k in ("date", "total") if key.get(k)]
    return sum(v.upper().replace(" ", "") in flat for v in vals), len(vals)


def ocr(img) -> str:
    return pytesseract.image_to_string(img, lang="eng")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sroie", required=True, help="folder containing img/, box/, key/")
    ap.add_argument("--n", type=int, default=50, help="number of receipts (fixed random sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sroie"))
    ap.add_argument("--tesseract-cmd", default=None,
                    help=r"path to tesseract.exe if it is not on PATH, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe")
    args = ap.parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    base = Path(args.sroie)
    images = sorted((base / "img").glob("*.jpg"))
    random.Random(args.seed).shuffle(images)
    images = images[: args.n]
    preprocess = load_preprocess()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        ref = norm(reference_text(base / "box" / f"{stem}.csv"))
        img = cv2.imread(str(img_path))
        for mode in ("raw", "pre"):
            t0 = time.perf_counter()
            try:
                src = img if mode == "raw" else preprocess(img)
                hyp = norm(ocr(src))
                err = ""
            except Exception as e:  # keep going; record the failure
                hyp, err = "", f"{type(e).__name__}: {e}"
            secs = time.perf_counter() - t0
            kh, kn = key_hits(base / "key" / f"{stem}.json", hyp)
            rows.append({
                "image": stem, "mode": mode,
                "cer": round(jiwer.cer(ref, hyp), 4) if hyp else 1.0,
                "wer": round(jiwer.wer(ref, hyp), 4) if hyp else 1.0,
                "word_recall": round(word_recall(ref, hyp), 4),
                "key_hits": kh, "key_total": kn,
                "seconds": round(secs, 2), "error": err,
            })
        print(f"[{i}/{len(images)}] {stem}")

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(mode, col, fn=statistics.mean):
        return fn([r[col] for r in rows if r["mode"] == mode])

    lines = [f"# SROIE OCR benchmark ({len(images)} receipts, seed {args.seed}, Tesseract eng)", "",
             "| Metric | Raw | Preprocessed |", "|---|---|---|"]
    for col, label in (("cer", "Mean CER (lower is better)"), ("wer", "Mean WER (lower is better)"),
                       ("word_recall", "Mean word recall (order-free)"), ("seconds", "Mean seconds per image")):
        lines.append(f"| {label} | {agg('raw', col):.3f} | {agg('pre', col):.3f} |")
    kr = (agg("raw", "key_hits", sum), agg("raw", "key_total", sum))
    kp = (agg("pre", "key_hits", sum), agg("pre", "key_total", sum))
    lines.append(f"| Date and total found in text | {kr[0]}/{kr[1]} | {kp[0]}/{kp[1]} |")
    better = sum(1 for s in {r['image'] for r in rows}
                 if next(r['cer'] for r in rows if r['image'] == s and r['mode'] == 'pre')
                 < next(r['cer'] for r in rows if r['image'] == s and r['mode'] == 'raw'))
    lines.append(f"| Receipts where preprocessing lowered CER | | {better}/{len(images)} |")
    fails = [r for r in rows if r["error"]]
    if fails:
        lines += ["", f"Errors: {len(fails)} (see per_image.csv)"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
