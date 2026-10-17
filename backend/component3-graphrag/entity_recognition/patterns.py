"""The extraction rules, kept as data: regex patterns for formatted values and a term lexicon.

Every pattern was written against wording found in the seven Inland Revenue Acts (examples in
the comments). Labels are the contract's EntityLabel values (contracts/j26_contracts).

Confidence is a fixed prior per rule: how often that kind of match is a true entity. Exact
formats (Rs. 1,800,000; 24%; April 1, 2025) are near-certain; single generic words ("relief",
"individual") are less certain because they also appear in non-rule text.
"""

from __future__ import annotations

import re

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# --- number words ("one hundred and eighty thousand rupees", "six per centum") -------------

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_NUM_WORD = "|".join(sorted([*UNITS, *TENS, *SCALES], key=len, reverse=True))
NUMBER_WORDS = rf"(?i:(?:{_NUM_WORD})(?:(?:\s+and\s+|[\s-]+)(?:{_NUM_WORD}))*)"  # "Zero percent"

# --- regex rules: (rule name, label, pattern, confidence) ---------------------------------

FIGURE = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
RS = rf"Rs\.?\s*(?:{FIGURE})(?:\s+(?:million|billion))?"  # "Rs. 1,800,000", "Rs. 3 million"

REGEX_RULES: list[tuple[str, str, str, float]] = [
    # Rate-table rows start with a capital: "Not Exceeding Rs. 600,000", "Exceeding Rs. 1,000,000
    # but not exceeding Rs. 1,500,000". Lower-case "not exceeding Rs. 50,000" in running text is
    # a cap, not a band, so it is left to the AMOUNT rule.
    (
        "tax_slab",
        "TAX_SLAB",
        rf"\b(?:Not\s+[Ee]xceeding|Exceeding)\s+{RS}(?:\s+but\s+not\s+exceeding\s+{RS})?",
        0.95,
    ),
    # "Rs. 1,800,000", "Rs.0", "Rs. 1.5 million"
    ("amount_figures", "AMOUNT", rf"\b{RS}", 0.95),
    # "one hundred and eighty thousand rupees"
    ("amount_words", "AMOUNT", rf"\b{NUMBER_WORDS}\s+rupees\b", 0.85),
    # "24%", "25 percent", "15 per centum", "10 per cent"
    ("rate_figures", "RATE", rf"\b(?:{FIGURE})\s*(?:%|per\s*cent(?:um)?\b|percent\b)", 0.95),
    # "six per centum"
    ("rate_words", "RATE", rf"\b{NUMBER_WORDS}\s+(?:per\s*cent(?:um)?|percent)\b", 0.85),
    # "April 1, 2025", "April 1 2025"
    ("date_mdy", "DATE", rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", 0.95),
    # "1st April, 2025", "24th of October, 2017", "1 April 2018"
    (
        "date_dmy",
        "DATE",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS}),?\s+\d{{4}}\b",
        0.95,
    ),
    # "01.04.2025" (tables of operation dates)
    (
        "date_dotted",
        "DATE",
        r"\b(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:19|20)\d{2}\b",
        0.9,
    ),
]

# legal references, joined into one span: "item (iii) of subparagraph (a) of paragraph 2 of the
# Fifth Schedule to the principal enactment", "section 52 of the Inland Revenue Act, No. 24 of 2017"
_ID = r"(?:\d+[A-Z]{0,2}|\([0-9]+[A-Z]{0,2}\)|\([a-z]{1,4}\)|\([ivx]{1,6}\))"
_IDS = rf"{_ID}(?:\s*{_ID})*"
_UNIT = rf"(?i:(?:sub-?)?(?:sections?|paragraphs?|sub-?paragraphs?|items?|chapter|part))\s+{_IDS}"
_SCHEDULE = r"(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)\s+Schedule"
_ACT = r"(?:[A-Z][A-Za-z]+\s+){0,4}(?:\([A-Za-z]+\)\s+)?Act,?\s+No\.?\s*\d+\s+of\s+\d{4}"
_THAT = (
    r"(?i:(?:that|this|the\s+said)\s+(?:section|subsection|paragraph|subparagraph|schedule|item))"
)
_TARGET = rf"(?:{_UNIT}|{_SCHEDULE}|{_ACT}|{_THAT}|(?i:principal\s+enactment|this\s+Act))"
SECTION_REF = rf"\b{_TARGET}(?:\s+(?:of|to)\s+(?:the\s+)?{_TARGET})*"
REGEX_RULES.append(("legal_reference", "SECTION_REF", SECTION_REF, 0.9))

# --- lexicon rules: (label, normalised value, phrases, confidence) -------------------------
# Matched on lower-cased tokens with spaCy's PhraseMatcher; the longest match wins.

LEXICON: list[tuple[str, str, list[str], float]] = [
    ("RELIEF", "personal_relief", ["personal relief", "personal reliefs"], 0.95),
    ("RELIEF", "relief", ["relief", "reliefs", "tax relief"], 0.75),
    ("DEDUCTION", "qualifying_payment", ["qualifying payment", "qualifying payments"], 0.95),
    (
        "DEDUCTION",
        "deduction",
        ["deduction", "deductions", "allowable deduction", "allowable deductions"],
        0.8,
    ),
    ("EXEMPTION", "exempt_amount", ["exempt amount", "exempt amounts"], 0.95),
    (
        "EXEMPTION",
        "exemption",
        ["exemption", "exemptions", "exempt from tax", "exempt from income tax"],
        0.85,
    ),
    (
        "INCOME_TYPE",
        "employment",
        ["employment income", "income from employment", "income from an employment"],
        0.95,
    ),
    (
        "INCOME_TYPE",
        "business",
        ["business income", "income from business", "income from a business"],
        0.95,
    ),
    (
        "INCOME_TYPE",
        "investment",
        [
            "investment income",
            "income from investment",
            "income from an investment",
            "rental income",
            "dividend income",
            "interest income",
        ],
        0.9,
    ),  # fmt: skip
    ("INCOME_TYPE", "other", ["other income", "income from other sources"], 0.85),
    (
        "TAXPAYER_CATEGORY",
        "resident_or_non_resident_individual",
        ["resident or non-resident individual", "resident or non - resident individual"],
        0.95,
    ),
    (
        "TAXPAYER_CATEGORY",
        "resident_individual",
        [
            "resident individual",
            "individual who is resident in sri lanka",
            "individual resident in sri lanka",
        ],
        0.95,
    ),  # fmt: skip
    (
        "TAXPAYER_CATEGORY",
        "non_resident_individual",
        [
            "non-resident individual",
            "non - resident individual",
            "individual who is not resident in sri lanka",
        ],
        0.95,
    ),  # fmt: skip
    ("TAXPAYER_CATEGORY", "senior_citizen", ["senior citizen", "senior citizens"], 0.95),
    ("TAXPAYER_CATEGORY", "individual", ["individual", "individuals"], 0.7),
    ("TAXPAYER_CATEGORY", "company", ["company", "companies"], 0.7),
    ("TAXPAYER_CATEGORY", "partnership", ["partnership", "partnerships"], 0.7),
    ("TAXPAYER_CATEGORY", "trust", ["trust", "trusts"], 0.6),
    (
        "TAXPAYER_CATEGORY",
        "charitable_institution",
        ["charitable institution", "charitable institutions"],
        0.9,
    ),
]

COMPILED = [(name, label, re.compile(p), conf) for name, label, p, conf in REGEX_RULES]
