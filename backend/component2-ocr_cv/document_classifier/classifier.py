"""Document type classification: payslip | bank_statement | invoice | receipt | unknown (FR8).

Why this module exists:
- Each document type needs a different extractor (a payslip has EPF/ETF, a utility bill has
  VAT/SSCL), so the type must be known BEFORE field extraction.
- Just as important: a document that is NOT a financial document (a course outline, a letter)
  must be recognised as 'unknown' instead of being forced into payslip fields. Saying "I don't
  know" is safer for the tax assistant than inventing EPF values.

Method: count the distinct type-specific key phrases found in the OCR text (English, Sinhala,
Tamil). The type with the most evidence wins only if it has enough evidence AND clearly beats the
runner-up; otherwise the answer is 'unknown'. Simple, explainable, and needs no training data.
Sinhala/Tamil phrases must be verified against real documents (TODO).

Layout evidence (own method, added after a real scanned statement came back 'unknown'): real
statements print short codes (B/F, ATM, POS) instead of textbook headers, but every transaction
row ends with an amount and the running balance. If consecutive balances add up
(balance = previous balance +/- amount) on enough rows, that arithmetic is strong evidence of a
bank statement, even when OCR garbles the words. It mirrors the fiscal self-correction idea:
use the document's own arithmetic as evidence.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

KEYWORDS: Dict[str, List[str]] = {
    "payslip": [
        "pay slip", "payslip", "salary slip", "pay statement", "basic salary", "basic pay", "gross salary",
        "gross pay", "net salary", "net pay", "epf", "etf", "apit", "paye", "allowance", "allowances",
        "total deductions", "earnings", "employee no", "employee id", "designation",
        "මුළු ආදායම", "ශුද්ධ වැටුප", "වැටුප", "බෝනස්", "දීමනාව", "මුළු අඩුකිරීම්",   # real Sinhala payslip
        "වැටුප් පත්‍රිකාව", "මූලික වැටුප", "සම්බළ", "சம்பளச் சீட்டு", "அடிப்படைச் சம்பளம்",   # TODO verify
    ],
    "bank_statement": [
        "statement of account", "account statement", "bank statement", "opening balance", "closing balance",
        "balance brought forward", "brought forward", "withdrawal", "withdrawals", "deposit", "deposits",
        "account no", "account number", "transaction", "value date", "cheque", "debit", "credit",
        "statement", "balance", "b/f", "c/f", "balance b/f", "savings", "savings account", "current account",
        "atm", "sms charges", "standing order",
        "ගිණුම් ප්‍රකාශය", "வங்கி அறிக்கை",                                                   # TODO verify
    ],
    "invoice": [
        "invoice", "tax invoice", "bill date", "billing period", "due date", "amount due", "total due",
        "payment due", "previous balance", "account summary", "vat", "sscl", "subscriber", "customer id",
        "broadband", "usage", "monthly rental", "telephone", "electricity", "water bill",
        "බිල්පත", "பட்டியல்",                                                                  # TODO verify
    ],
    "receipt": [
        "receipt", "cash", "change", "subtotal", "sub total", "qty", "item", "cashier", "tendered",
        "thank you", "pos", "card no", "bill no", "price",
        "රිසිට්පත", "ரசீது",                                                                    # TODO verify
    ],
}

MIN_EVIDENCE = 2      # at least 2 different key phrases of the winning type
MIN_MARGIN = 1        # and at least 1 more than the runner-up

# Running-balance check (layout evidence for bank statements)
MIN_TABLE_ROWS = 5            # rows ending in two amounts (amount, balance)
MIN_CHAIN_SHARE = 0.6         # share of consecutive rows whose balances add up
CHAIN_WEIGHT = 3              # counts like 3 key phrases: arithmetic is harder to fake than a word
CHAIN_TOLERANCE = 0.011       # rupees; allows rounding to cents
# Card/cash words that appear on statements as transaction codes, not as receipt evidence
STATEMENT_CODES_NOT_RECEIPT = ("pos", "cash")

_AMOUNT = re.compile(r"^\(?\d{1,3}(?:,\d{3})+\.\d{2}\)?$|^\(?\d+\.\d{2}\)?(?:cr|dr)?$", re.IGNORECASE)


@dataclass
class Classification:
    document_type: str
    confidence: float
    scores: Dict[str, int] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w඀-෿஀-௿ ]+", " ", text)
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def _amount(word: str) -> float | None:
    if not _AMOUNT.match(word):
        return None
    return float(re.sub(r"[^\d.]", "", word))


def balance_chain(lines: Sequence[str]) -> tuple[int, int]:
    """Return (transaction_rows, consecutive_pairs_that_add_up).

    A transaction row is a line whose last two money values are (amount, running balance).
    A pair adds up when balance = previous balance + amount or previous balance - amount.
    """
    rows = []
    for line in lines:
        amounts = [a for a in (_amount(w) for w in line.split()) if a is not None]
        if len(amounts) >= 2:
            rows.append((amounts[-2], amounts[-1]))
    ok = sum(
        1 for (_, prev_bal), (amt, bal) in zip(rows, rows[1:])
        if min(abs(prev_bal + amt - bal), abs(prev_bal - amt - bal)) <= CHAIN_TOLERANCE
    )
    return len(rows), ok


def classify_text(text: str) -> Classification:
    """Classify OCR text. Lines (separated by newlines) are used for the running-balance check."""
    norm = _normalise(text)
    evidence = {t: [k for k in kws if f" {_normalise(k).strip()} " in norm] for t, kws in KEYWORDS.items()}

    rows, ok = balance_chain(text.splitlines())
    if rows >= MIN_TABLE_ROWS and ok >= MIN_CHAIN_SHARE * (rows - 1):
        evidence["bank_statement"] += [f"[layout] {ok}/{rows - 1} running balances add up"] * CHAIN_WEIGHT
        evidence["receipt"] = [k for k in evidence["receipt"] if k not in STATEMENT_CODES_NOT_RECEIPT]

    scores = {t: len(v) for t, v in evidence.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < MIN_EVIDENCE:
        return Classification("unknown", 0.0, scores, evidence,
                              f"not enough financial-document evidence (best: {best} with {s1} key phrase(s))")
    if s1 - s2 < MIN_MARGIN:
        return Classification("unknown", 0.0, scores, evidence,
                              f"ambiguous: {ranked[0][0]} and {ranked[1][0]} both have {s1} key phrases")
    conf = min(0.99, 0.5 + 0.08 * s1 + 0.05 * (s1 - s2))
    shown = list(dict.fromkeys(evidence[best]))          # layout evidence listed once
    return Classification(best, round(conf, 3), scores, {best: shown},
                          f"{s1} {best} evidence points vs {s2} for the next type")


def classify_tokens(tokens: Sequence, min_confidence: float = 0.4) -> Classification:
    """Classify from OCR tokens, ignoring very low-confidence fragments (noise, logos).

    Tokens are joined line by line (line_id) so the running-balance check sees table rows.
    """
    lines: Dict[str, List[str]] = {}
    for t in tokens:
        if t.confidence >= min_confidence:
            lines.setdefault(getattr(t, "line_id", "0"), []).append(t.text)
    return classify_text("\n".join(" ".join(ws) for ws in lines.values()))
