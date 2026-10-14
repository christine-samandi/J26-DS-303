"""C1 confidence, v1 formula (CLAUDE.md section 8). Kept simple so it can be audited:

score = min(ocr, retrieval, interpretation, explanation) x conflict_penalty

ocr             lowest confidence among the input fields actually used (typed input = 1)
retrieval       lowest C3 confidence among the rules actually used
interpretation  lowest LLM interpretation confidence (1 when no LLM interpretation ran)
explanation     1 if the explanation passed its post-check, 0.5 if it fell back to the skeleton
conflict_penalty 0.6 if the seed rules and C3 disagree, else 1
"""

from __future__ import annotations

from decimal import Decimal

from orchestrator.models import ConfidenceBreakdown

ONE = Decimal(1)
FALLBACK_EXPLANATION = Decimal("0.5")
CONFLICT_PENALTY = Decimal("0.6")
REVIEW_THRESHOLD = Decimal("0.7")  # starting value; tune on the golden set


def c1_confidence(
    ocr: list[Decimal],
    retrieval: list[Decimal],
    interpretation: list[Decimal],
    explanation_fell_back: bool,
    has_conflict: bool,
) -> ConfidenceBreakdown:
    terms = {
        "ocr": min(ocr, default=ONE),
        "retrieval": min(retrieval, default=ONE),
        "interpretation": min(interpretation, default=ONE),
        "explanation": FALLBACK_EXPLANATION if explanation_fell_back else ONE,
    }
    penalty = CONFLICT_PENALTY if has_conflict else ONE
    return ConfidenceBreakdown(
        **terms, conflict_penalty=penalty, score=min(terms.values()) * penalty
    )
