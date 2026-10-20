"""Make redacted copies of real documents for the dataset (NFR4).

Why this script exists: real payslips and statements must be redacted (names, NIC, account and
phone numbers, addresses) before they are stored or shared for research. The PII guard finds them
with OCR and covers them with black boxes on a COPY; you still check every copy by eye, because
OCR can miss a badly printed identifier.

Usage (from backend/component2-ocr_cv):
    python scripts/redact.py path/to/photo.jpg [more files...] --out data/images
Writes <name>_redacted.png next to --out; prints only what KIND and how many items were covered.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from api.pipeline import blackout, choose_language, decode_upload  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from privacy import mask_tokens  # noqa: E402


def redact(path: Path, out_dir: Path) -> Path:
    img, _ = decode_upload(path.read_bytes(), path.name)
    res = preprocess(img)
    tokens, lang, _ = choose_language(res.image)
    _, report = mask_tokens(tokens)
    boxes = [res.to_original_bbox(b) for b in report.regions]
    out = out_dir / f"{path.stem}_redacted.png"
    cv2.imwrite(str(out), blackout(img, boxes))
    found = ", ".join(f"{k} {v}" for k, v in sorted(report.counts.items())) or "nothing found"
    print(f"{path.name}: covered {len(boxes)} areas ({found}) -> {out}   [OCR language: {lang}]")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        redact(Path(f), out_dir)
    print("\nCHECK EVERY COPY BY EYE before using or sharing it: automatic redaction can miss items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
