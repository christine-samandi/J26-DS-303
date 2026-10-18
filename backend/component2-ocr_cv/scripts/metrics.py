"""Scoring rules for the Component 2 benchmark (Objective 2.5, evaluation plan).

Why this module exists:
- "High accuracy" has to be a number the panel can check, computed the same way for every
  engine, preprocessing setting and capture condition. Keeping the rules here (pure functions,
  no OCR) makes them testable and identical across runs.
- Field accuracy matters more than character accuracy for a tax assistant: one wrong digit in
  gross_income is a wrong tax answer even if CER looks excellent. So both are reported.
- Confidence must be honest: a field marked trusted (confidence >= 0.95) should almost always be
  right. Expected Calibration Error (ECE) and the "wrong but not flagged" count measure that.

Field statuses (one per ground-truth field or predicted field):
    correct  predicted and equal to the ground truth
    wrong    predicted but different            (counts as a false positive AND a false negative)
    missed   in the ground truth, not predicted (false negative)
    extra    predicted, but the ground truth is null / not expected (false positive)
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

AMOUNT_TOLERANCE = 0.01        # rupees: amounts must match to the cent
ISSUER_SIMILARITY = 0.80       # issuer names: OCR may drop "(PVT)" punctuation etc.
TEXT_FIELDS = {"issuer"}
STRING_FIELDS = {"issuer", "pay_period", "document_date"}


@dataclass
class FieldResult:
    field: str
    status: str                 # correct | wrong | missed | extra
    confidence: Optional[float] = None
    needs_review: Optional[bool] = None
    predicted: object = None
    expected: object = None


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def values_match(field: str, predicted, expected) -> bool:
    if predicted is None or expected is None:
        return False
    if field in TEXT_FIELDS:
        a, b = normalise_text(predicted), normalise_text(expected)
        return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= ISSUER_SIMILARITY
    if field in STRING_FIELDS:
        return str(predicted).strip() == str(expected).strip()
    try:
        return abs(float(predicted) - float(expected)) <= AMOUNT_TOLERANCE
    except (TypeError, ValueError):
        return False


def score_fields(predicted: Dict[str, Tuple[object, float, bool]], expected: Dict[str, object]) -> List[FieldResult]:
    """predicted: field -> (value, confidence, needs_review); expected: ground-truth 'fields' dict."""
    out: List[FieldResult] = []
    for f, gt in expected.items():
        if gt is None:
            if f in predicted:
                v, c, r = predicted[f]
                out.append(FieldResult(f, "extra", c, r, v, None))
            continue
        if f not in predicted:
            out.append(FieldResult(f, "missed", expected=gt))
            continue
        v, c, r = predicted[f]
        out.append(FieldResult(f, "correct" if values_match(f, v, gt) else "wrong", c, r, v, gt))
    for f, (v, c, r) in predicted.items():
        if f not in expected:
            out.append(FieldResult(f, "extra", c, r, v, None))
    return out


def precision_recall_f1(results: Iterable[FieldResult]) -> Dict[str, float]:
    results = list(results)
    tp = sum(r.status == "correct" for r in results)
    wrong = sum(r.status == "wrong" for r in results)
    fp = wrong + sum(r.status == "extra" for r in results)
    fn = wrong + sum(r.status == "missed" for r in results)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4)}


def expected_calibration_error(pairs: Sequence[Tuple[float, bool]], bins: int = 10) -> Optional[float]:
    """pairs: (confidence, was_correct) for every predicted field. 0 = perfectly honest confidence."""
    if not pairs:
        return None
    total, ece = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if bucket:
            conf = sum(c for c, _ in bucket) / len(bucket)
            acc = sum(ok for _, ok in bucket) / len(bucket)
            ece += len(bucket) / total * abs(conf - acc)
    return round(ece, 4)


def review_quality(results: Iterable[FieldResult]) -> Dict[str, int]:
    """How useful the needs_review flag is: wrong values should be flagged, right ones trusted."""
    pred = [r for r in results if r.status in ("correct", "wrong", "extra")]
    return {
        "trusted_correct": sum(r.status == "correct" and not r.needs_review for r in pred),
        "trusted_wrong": sum(r.status != "correct" and not r.needs_review for r in pred),   # the dangerous case
        "flagged_correct": sum(r.status == "correct" and bool(r.needs_review) for r in pred),
        "flagged_wrong": sum(r.status != "correct" and bool(r.needs_review) for r in pred),
    }


def cer_wer(reference: str, hypothesis: str) -> Tuple[Optional[float], Optional[float]]:
    """Character/word error rate (jiwer); None when no reference text was typed."""
    ref = re.sub(r"\s+", " ", reference or "").strip()
    if not ref:
        return None, None
    import jiwer
    hyp = re.sub(r"\s+", " ", hypothesis or "").strip()
    return round(jiwer.cer(ref, hyp), 4), round(jiwer.wer(ref, hyp), 4)
