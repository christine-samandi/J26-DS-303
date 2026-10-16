"""Generate the synthetic "bad phone photo" payslip used for the first preprocessing check.

Why: it lets anyone reproduce the raw-vs-preprocessed comparison without any real (private)
document. It is deliberately hard: 7 degree tilt, dark gradient lighting, heavy noise and strong
JPEG compression, which are the artefacts of a cheap phone camera in a dim room.
All names and amounts are fictional.

Usage (from backend/component2-ocr_cv):
    python scripts/make_synthetic_sample.py
Writes sample_documents/SYN_payslip_clean.png and sample_documents/SYN_payslip_dark_skew.jpg
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "sample_documents"
LINES = ["LANKA TEXTILES (PVT) LTD", "PAY SLIP - SEPTEMBER 2026",
         "Basic Salary            120,000.00", "Allowances               30,000.00",
         "Gross Salary            150,000.00", "EPF Employee 8%          12,000.00",
         "APIT                      3,500.00", "Total Deductions         15,500.00",
         "Net Salary              134,500.00"]


def clean_page(w: int = 850, h: int = 1100) -> np.ndarray:
    page = np.full((h, w), 255, np.uint8)
    for i, line in enumerate(LINES):
        cv2.putText(page, line, (70, 100 + i * 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2)
    return page


def degrade(page: np.ndarray, angle: float = 7.0, noise: float = 22.0, seed: int = 1) -> np.ndarray:
    h, w = page.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(page, M, (w, h), borderValue=255).astype(np.float32)
    img *= np.linspace(0.45, 0.95, w)[None, :]                      # dark on the left: uneven light
    img += np.random.default_rng(seed).normal(0, noise, img.shape)   # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    page = clean_page()
    cv2.imwrite(str(OUT / "SYN_payslip_clean.png"), page)
    cv2.imwrite(str(OUT / "SYN_payslip_dark_skew.jpg"), cv2.cvtColor(degrade(page), cv2.COLOR_GRAY2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 60])
    print(f"Wrote 2 synthetic samples to {OUT}")


if __name__ == "__main__":
    main()
