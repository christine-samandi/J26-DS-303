"""SROIE public benchmark for the reading stage: fixed page variants vs quality-gated reading.

Why this exists: every other Component 2 result comes from our own small Sri Lankan set. This
script measures the READING stage (preprocessing + Tesseract) and the document classifier on a
public, labelled benchmark (ICDAR 2019 SROIE receipts). It does not test field extraction or the
fiscal checks, because SROIE has no Sri Lankan tax fields.

v1 finding: always applying the full preprocessing (built for phone photos) rescues receipts that
raw OCR fails on, but damages clean scans. v2 therefore tests a reading-quality gate (own method):
read the page in a few variants and keep the reading with the most confident, word-like
characters, using nothing but the reading itself (no ground truth). The document type is decided
by a vote of the classifier over all readings.

Design vs test: the gate was designed on the 'dev' receipts (seed 42) and is reported on a separate
'test' set (seed 7, disjoint), so the result is not tuned to the receipts it is scored on.

Data: corrected SROIE train/val set from github.com/zzzDavid/ICDAR-2019-SROIE (MIT); folder data/
with img/*.jpg, box/*.csv (x1..y4,text per line) and key/*.json. Keep it in data/public/sroie/
(git-ignored). Never commit it.

Usage (from backend/component2-ocr_cv):
    python scripts/sroie_ocr_benchmark.py --sroie data/public/sroie/data --n 50
Outputs: outputs/sroie/per_reading.csv and outputs/sroie/summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import jiwer
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from document_classifier.classifier import classify_tokens  # noqa: E402
from ocr_engines.tesseract_engine import run_tesseract  # noqa: E402
from preprocessing import preprocess  # noqa: E402

GEOMETRIC = ("resize", "deskew", "perspective")      # steps that move pixels, as in api/pipeline.py
VARIANTS = ("raw", "full", "grey_psm4", "plain", "raw_psm4")
PURCHASE = ("receipt", "invoice")                     # SROIE receipts often print 'TAX INVOICE'
GATED = {                                             # candidate sets compared by the gate
    "gate: raw / full / grey (recommended)": ("raw", "full", "grey_psm4"),
    "gate: raw / grey (cheapest)": ("raw", "grey_psm4"),
    "gate: all 5 readings": VARIANTS,
}
WORDLIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.,:/&@#%()'-]*$")
CONFIDENT = 0.80


# ---------------------------------------------------------------- readings
def readings(img):
    """The page variants the pipeline can produce (same functions as api/pipeline.py)."""
    full = preprocess(img)
    steps = full.steps_applied
    grey_steps = [s for s in steps if s != "threshold"]
    plain_steps = [s for s in steps if s in GEOMETRIC]
    yield "raw", img, 6
    yield "full", full.image, 6
    yield "grey_psm4", preprocess(img, steps=grey_steps).image, 4
    yield "plain", preprocess(img, steps=plain_steps).image, 6
    yield "raw_psm4", img, 4


def confident_chars(tokens) -> int:
    """Reading-quality score (own method): characters in confident, word-like tokens.
    Uses only the reading itself, so it works on any document at run time."""
    return sum(len(t.text) for t in tokens if t.confidence >= CONFIDENT and WORDLIKE.match(t.text))


def vote(types) -> str:
    """Document type agreed by most readings; 'unknown' only if no reading found a type."""
    c = Counter(t for t in types if t != "unknown")
    return c.most_common(1)[0][0] if c else "unknown"


# ---------------------------------------------------------------- metrics
def norm(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.upper()).strip()


def reference_text(box_file: Path) -> str:
    lines = []
    for row in box_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = row.split(",", 8)
        if len(p) == 9 and p[8].strip():
            lines.append((int(p[1]), int(p[0]), p[8].strip()))
    lines.sort(key=lambda t: (round(t[0] / 10), t[1]))
    return norm("\n".join(t[2] for t in lines))


def tokens_text(tokens) -> str:
    lines = {}
    for t in tokens:
        lines.setdefault(t.line_id, []).append(t.text)
    return norm("\n".join(" ".join(w) for w in lines.values()))


def word_recall(ref: str, hyp: str) -> float:
    pool = Counter(hyp.split())
    hit = 0
    for w in ref.split():
        if pool[w] > 0:
            hit += 1
            pool[w] -= 1
    return hit / max(1, len(ref.split()))


def key_hits(key_file: Path, hyp: str) -> int:
    key = json.loads(key_file.read_text(encoding="utf-8")) if key_file.exists() else {}
    flat = hyp.replace(" ", "")
    return sum(key[k].upper().replace(" ", "") in flat for k in ("date", "total") if key.get(k))


# ---------------------------------------------------------------- main
def splits(base: Path, n: int):
    imgs = sorted(p.stem for p in (base / "img").glob("*.jpg"))
    a = imgs[:]
    random.Random(42).shuffle(a)
    dev = a[:n]
    rest = [s for s in imgs if s not in set(dev)]
    random.Random(7).shuffle(rest)
    return {"dev": dev, "test": rest[:n]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sroie", required=True, help="folder containing img/, box/, key/")
    ap.add_argument("--n", type=int, default=50, help="receipts per split (dev and test)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sroie"))
    ap.add_argument("--tesseract-cmd", default=None,
                    help=r"path to tesseract.exe if it is not on PATH, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe")
    args = ap.parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    base = Path(args.sroie)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per, recs = [], []
    todo = [(sp, s) for sp, stems in splits(base, args.n).items() for s in stems]
    for i, (split, stem) in enumerate(todo, 1):
        ref = reference_text(base / "box" / f"{stem}.csv")
        img = cv2.imread(str(base / "img" / f"{stem}.jpg"))
        rec = {"split": split, "image": stem, "r": {}}
        for name, page, psm in readings(img):
            t0 = time.perf_counter()
            toks = run_tesseract(page, lang="eng", psm=psm)
            secs = time.perf_counter() - t0
            hyp = tokens_text(toks)
            row = {"split": split, "image": stem, "reading": name,
                   "cer": round(jiwer.cer(ref, hyp), 4) if hyp else 1.0,
                   "word_recall": round(word_recall(ref, hyp), 4),
                   "key_hits": key_hits(base / "key" / f"{stem}.json", hyp),
                   "doc_type": classify_tokens(toks).document_type,
                   "confident_chars": confident_chars(toks), "seconds": round(secs, 2)}
            rec["r"][name] = row
            per.append(row)
        recs.append(rec)
        print(f"[{i}/{len(todo)}] {split} {stem}")

    with open(out / "per_reading.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)

    strategies = {f"fixed: {v}": (v,) for v in VARIANTS} | GATED
    lines = [f"# SROIE reading benchmark ({args.n} dev + {args.n} test receipts, Tesseract eng)", "",
             "Fixed = always the same page; gate = keep the reading with the most confident characters; "
             "document type for a gate = vote over its readings.", ""]
    for split in ("dev", "test"):
        rows = [r for r in recs if r["split"] == split]
        lines += [f"## {split}", "", "| Strategy | Mean CER | Median CER | Word recall | Date+total found | "
                  "Typed receipt/invoice | Seconds |", "|---|---|---|---|---|---|---|"]
        for label, cands in strategies.items():
            pick = [max(cands, key=lambda v: r["r"][v]["confident_chars"]) for r in rows]
            got = [r["r"][v] for r, v in zip(rows, pick)]
            typed = sum(vote([r["r"][v]["doc_type"] for v in cands]) in PURCHASE for r in rows)
            secs = st.mean(sum(r["r"][v]["seconds"] for v in cands) for r in rows)
            lines.append(f"| {label} | {st.mean(g['cer'] for g in got):.3f} | {st.median(g['cer'] for g in got):.3f} | "
                         f"{st.mean(g['word_recall'] for g in got):.3f} | {sum(g['key_hits'] for g in got)}/"
                         f"{2 * len(rows)} | {typed}/{len(rows)} | {secs:.1f} |")
        oracle = [min(VARIANTS, key=lambda v: r["r"][v]["cer"]) for r in rows]
        lines.append(f"| oracle (best CER, upper bound) | "
                     f"{st.mean(r['r'][v]['cer'] for r, v in zip(rows, oracle)):.3f} | | | | | |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
