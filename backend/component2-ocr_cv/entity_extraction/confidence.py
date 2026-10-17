"""Per-field confidence and review flags (Gap 7: explainable confidence).

Why this module exists:
OCR confidence alone measures how clearly the characters were seen, not whether the value is
right. Here it is combined with the fiscal checks:
    - field confirmed by a passed arithmetic rule  -> confidence raised towards 1
    - field corrected by the self-correction step  -> capped below the review threshold
    - field in a FAILED hard rule (net = gross - deductions) -> lowered
    - no rule could check the field                -> OCR confidence as it is
Anything below REVIEW_THRESHOLD (0.95, FR7) is flagged for the user to confirm.
"""
from __future__ import annotations

from typing import Optional

from api.schema import REVIEW_THRESHOLD

CORRECTED_CAP = 0.90      # a corrected value is a well-supported guess, so the user must confirm it
FAILED_HARD_FACTOR = 0.6
CLEANED_FACTOR = 0.95     # look-alike characters were replaced in the amount text


def field_confidence(ocr_conf: float, *, verified: bool, corrected: bool, contradicted: bool,
                     cleaned: bool = False, match_score: float = 1.0) -> float:
    c = max(0.0, min(1.0, ocr_conf)) * min(1.0, match_score)
    if cleaned:
        c *= CLEANED_FACTOR
    if corrected:
        c = min(c, CORRECTED_CAP) if c > 0 else CORRECTED_CAP
        c = max(c, 0.5)
    elif verified:
        c = 1 - (1 - c) * 0.3          # independent arithmetic evidence removes most of the doubt
    elif contradicted:
        c *= FAILED_HARD_FACTOR
    return round(min(c, 0.99), 3)


def needs_review(conf: float, threshold: Optional[float] = None) -> bool:
    return conf < (REVIEW_THRESHOLD if threshold is None else threshold)
