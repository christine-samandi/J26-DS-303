"""Trilingual label dictionary for payslip fields (Gap 2: mixed-script field mapping).

Why this module exists:
- The same field is printed differently on every payslip ("Gross Salary", "Gross Pay",
  "Total Earnings") and in three languages. This dictionary maps all of them to ONE field
  name from the output schema, so later stages never care about wording or language.
- OCR often damages a letter or two ("Net Safary"), so matching is fuzzy, word by word.

Sinhala and Tamil phrases are standard terms and must be VERIFIED against real Sri Lankan
payslips (see TODO markers); add the exact printed wording when it differs.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# Fields with an output-schema name, plus two helper fields used only for arithmetic checks.
HELPER_FIELDS = {"basic_salary", "allowances"}

FIELD_LABELS: Dict[str, List[str]] = {
    "gross_income": [
        "gross salary", "gross pay", "gross earnings", "gross income", "total earnings", "total gross",
        "දළ වැටුප", "මුළු ඉපැයීම්",                       # TODO verify on real payslips
        "மொத்த சம்பளம்", "மொத்த வருமானம்",                # TODO verify on real payslips
    ],
    "basic_salary": [
        "basic salary", "basic pay", "basic",
        "මූලික වැටුප", "அடிப்படைச் சம்பளம்", "அடிப்படை சம்பளம்",
    ],
    "allowances": [
        "allowances", "allowance", "total allowances",
        "දීමනා", "கொடுப்பனவுகள்",
    ],
    "epf_employee": [
        "epf employee", "employee epf", "epf 8%", "epf (8%)", "epf employee 8%", "epf deduction", "epf",
        "සේවක අර්ථසාධක අරමුදල", "ஊழியர் சேமலாப நிதியம்",
    ],
    "epf_employer": [
        "epf employer", "employer epf", "epf 12%", "epf (12%)", "epf employer 12%", "company epf",
        "epf corporation contribution", "epf company contribution", "epf employer contribution",
        "epf corpn contribution",
    ],
    "etf": [
        "etf", "etf 3%", "etf (3%)", "employer etf",
        "සේවා නියුක්තයන්ගේ භාරකාර අරමුදල", "ஊழியர் நம்பிக்கை நிதியம்",
    ],
    "apit": [
        "apit", "paye", "paye tax", "advance personal income tax", "pay as you earn", "income tax",
        "අග්‍රිම පුද්ගල ආදායම් බද්ද", "முற்கூட்டிய தனிநபர் வருமான வரி",
    ],
    "total_deductions": [
        "total deductions", "total deduction", "deductions total",
        "මුළු අඩු කිරීම්", "மொத்த கழிவுகள்",
    ],
    "net_amount": [
        "net salary", "net pay", "take home pay", "take home", "net amount", "amount payable", "net",
        "ශුද්ධ වැටුප", "நிகர சம்பளம்",
    ],
}

WORD_MATCH = 0.80     # per-word similarity needed (handles 1 wrong letter in ~5)
FIELD_MATCH = 0.80    # overall score needed to accept a label


def normalise(text: str) -> str:
    """Lower-case, Unicode NFC, keep letters/digits/% and spaces only."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w%඀-෿஀-௿ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _phrase_score(label_words: List[str], phrase: str) -> float:
    """Every phrase word must be found (fuzzily) among the label words; score = mean similarity.
    Extra words on the label are allowed ('EPF Employee 8% Deduction')."""
    words = phrase.split()
    if len(words) > len(label_words):
        return 0.0
    best = []
    for w in words:
        # Short words (<= 4 letters, e.g. 'pay', 'net', 'epf') must match exactly: one wrong
        # letter changes their meaning ('pay' is not 'paye').
        s = max((1.0 if w == lw else 0.0) if len(w) <= 4 else _sim(w, lw) for lw in label_words)
        if s < WORD_MATCH:
            return 0.0
        best.append(s)
    return sum(best) / len(best)


def match_field(label_text: str) -> Optional[Tuple[str, float, str]]:
    """Return (field, score, matched_phrase) for the best-matching field, or None.
    Ties are broken towards the LONGER phrase, so 'EPF Employer 12%' beats plain 'EPF'."""
    label = normalise(label_text)
    if not label:
        return None
    words = label.split()
    best = None
    for field, phrases in FIELD_LABELS.items():
        for p in phrases:
            s = _phrase_score(words, normalise(p))
            if s >= FIELD_MATCH:
                key = (s + 0.02 * len(p.split()), len(p))
                if best is None or key > best[0]:
                    best = (key, field, s, p)
    if best is None:
        return None
    _, field, s, p = best
    # 'EPF Gross' / 'ETF Gross' are the BASE amounts the contributions are calculated on, not the
    # contributions themselves; matching them as epf/etf put the wrong amount in those fields.
    if field in ("epf_employee", "epf_employer", "etf") and "gross" in words:
        return None
    # Percentage cue resolves EPF employee (8%) vs employer (12%) when the words alone are ambiguous.
    if field.startswith("epf_"):
        if "12%" in label:
            field = "epf_employer"
        elif "8%" in label:
            field = "epf_employee"
    return field, round(s, 3), p
