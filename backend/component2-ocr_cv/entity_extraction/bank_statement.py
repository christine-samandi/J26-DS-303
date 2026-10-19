"""Bank statement extraction: bank, statement date, statement month and closing balance.

Why this module exists:
- For tax, a bank statement is evidence of income flowing through an account (interest is
  taxable and banks withhold tax on it; deposits can show business income). The first step is
  the statement's identity and its closing balance, all fields already in the output contract.
- Statements carry their own arithmetic, so the closing balance can be VERIFIED, the same idea as
  the payslip fiscal rules (own method, consistent across document types):
      opening balance + total deposits - total withdrawals = closing balance   (hard rule)
      closing balance = last running balance in the transaction table           (hard rule)
  If the closing balance fails and a single OCR misreading of it (one digit dropped or swapped)
  satisfies the rules uniquely, it is corrected and flagged for review.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from api.schema import ExtractedField, OCROutput
from entity_extraction.amounts import parse_amount
from entity_extraction.confidence import field_confidence, needs_review
from entity_extraction.extractor import Line, group_lines, split_amount, union_bbox
from entity_extraction.fiscal import misreadings

ONTOLOGY = json.loads((Path(__file__).parent / "ontology" / "tax_ontology.json").read_text(encoding="utf-8"))["fields"]
TOL = 0.01
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], start=1)}
# Licensed banks in Sri Lanka and the distinctive words that identify each on a statement header.
# Checked in order (Commercial Bank of Ceylon before Bank of Ceylon). Fuzzy per word: OCR on logos
# and small print damages letters ("Centon" for "Ceylon").
BANKS = [
    ("Commercial Bank of Ceylon PLC", ["commercial", "bank"]), ("Hatton National Bank PLC", ["hatton"]),
    ("Sampath Bank PLC", ["sampath"]), ("Seylan Bank PLC", ["seylan"]), ("Nations Trust Bank PLC", ["nations", "trust"]),
    ("National Development Bank PLC", ["national", "development"]), ("DFCC Bank PLC", ["dfcc"]),
    ("Pan Asia Banking Corporation PLC", ["pan", "asia"]), ("Union Bank of Colombo PLC", ["union", "bank"]),
    ("Amana Bank PLC", ["amana"]), ("Cargills Bank PLC", ["cargills"]), ("National Savings Bank", ["savings", "bank"]),
    ("People's Bank", ["people", "bank"]), ("Sanasa Development Bank PLC", ["sanasa"]),
    ("Regional Development Bank", ["regional", "development"]), ("Standard Chartered Bank", ["standard", "chartered"]),
    ("HSBC", ["hsbc"]), ("Bank of Ceylon", ["bank", "ceylon"]),
]
BANK_NAME = re.compile(r"\b(bank|plc|ceylon|sampath|hnb|hatton|seylan|ndb|dfcc|nsb|nations trust|people'?s)\b", re.I)
LABELS = {
    "opening": re.compile(r"\b(opening balance|balance brought forward|brought forward|b\s*/\s*f|balance as (of|at))\b", re.I),
    "closing": re.compile(r"\b(closing( balance)?|balance carried forward|carried forward|c\s*/\s*f|ending balance)\b", re.I),
    "deposits": re.compile(r"\btotal (deposits|credits|receipts)\b", re.I),
    "withdrawals": re.compile(r"\btotal (withdrawals|debits|payments)\b", re.I),
}
DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b"), "dmy"),
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*(\d{4}|\d{2})\b", re.I), "dMy"),
]


def parse_date(text: str) -> Optional[date]:
    """'29/07/24', '29JUL24', '17 June 2025', '2024-07-29' -> date (day first, as printed in Sri Lanka)."""
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(text):
            try:
                if kind == "ymd":
                    y, mo, d = int(m[1]), int(m[2]), int(m[3])
                elif kind == "dmy":
                    d, mo, y = int(m[1]), int(m[2]), int(m[3])
                    if mo > 12 >= d:            # '10/21/25': printed month first (US style)
                        d, mo = mo, d
                else:
                    d, mo, y = int(m[1]), MONTHS[m[2][:3].lower()], int(m[3])
                y = y + 2000 if y < 100 else y
                return date(y, mo, d)
            except (ValueError, KeyError):
                continue
    return None


CENTS = re.compile(r"\d[.,\s]\d{2}$")


def _amount(line: Line):
    """Statement amounts are always printed with cents; anything else ('4', '50,000') is OCR noise
    from the table and must not be used as a balance."""
    _, amount = split_amount(line)
    if not amount:
        return None, None
    raw = " ".join(t.text for t in amount)
    p = parse_amount(raw)
    return (p, amount) if p and CENTS.search(raw.strip()) else (None, None)


def _conf(tokens) -> float:
    return round(sum(t.confidence for t in tokens) / len(tokens), 4)


def _word_hit(word: str, tokens) -> Optional[object]:
    from difflib import SequenceMatcher
    for t in tokens:
        w = re.sub(r"[^a-z]", "", t.text.lower())
        if not w:
            continue
        if (w == word) if len(word) <= 4 else (SequenceMatcher(None, w, word).ratio() >= 0.8 or w.startswith(word)):
            return t
    return None


def _find_issuer(lines: List[Line]):
    """Match the statement header against the list of Sri Lankan banks; returns the official name."""
    # Transaction rows (they start with a date such as 120825) hold shop and payee names, never the bank.
    lines = [ln for ln in lines if not TXN_DATE.match(ln.text)]
    header = [t for ln in lines[: max(8, len(lines) // 3)] for t in ln.tokens]
    for name, words in BANKS:
        hits = [_word_hit(w, header) for w in words]
        if all(hits):
            return name, Line(hits), "header"
    # Some banks print their name only in a round stamp. On the whole page a shop name can look like a
    # bank ('CARGILLS - KOTHALAWA' is a supermarket purchase), so the word 'bank' must appear within
    # two lines of the bank's own words.
    for name, words in BANKS:
        need = words if "bank" in words else words + ["bank"]
        for i in range(len(lines)):
            near = [t for ln in lines[max(0, i - 2): i + 3] for t in ln.tokens]
            hits = [_word_hit(w, near) for w in need]
            if all(hits):
                return name, Line(hits), "stamp"
    name, ln = _find_issuer_line(lines)
    return name, ln, "header"


def _find_issuer_line(lines: List[Line]):
    best = None
    for ln in lines[: max(6, len(lines) // 3)]:
        text = ln.text.strip(" -|")
        if BANK_NAME.search(text) and not split_amount(ln)[1] and not re.search(r"statement", text, re.I):
            key = (bool(re.search(r"\b(plc|ltd|limited)\b", text, re.I)), len(text))
            if best is None or key > best[0]:
                best = (key, text, ln)
    return (best[1], best[2]) if best else (None, None)


def _find_date(lines: List[Line]):
    """Statement date: a line starting with 'DATE' / 'STATEMENT DATE', else the first date in the header."""
    for want_label in (True, False):
        for ln in lines[: max(8, len(lines) // 2)]:
            if want_label and not re.match(r"^\W*(statement\s+)?date\b", ln.text, re.I):
                continue
            d = parse_date(ln.text)
            if d:
                return d, ln
    return None, None


TXN_DATE = re.compile(r"^\W*(\d{2})(\d{2})(\d{2})\b|^\W*(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{2})\b",
                      re.I)
END = re.compile(r"end of statement|closing", re.I)
MONEY = re.compile(r"\d{1,3}(?:,\d{3})+[.\s]\d{2}(?!\d)|\d+[.\s]\d{2}(?!\d)")


def row_amounts(tokens) -> List[float]:
    """Money values in a table row, rebuilt from the whole row because OCR often splits an amount at
    its decimal point ('4,118 17') and misreads 0 as @ inside numbers ('5 0@')."""
    text = " ".join(t.text for t in tokens)
    text = re.sub(r"(?<=\d)[@Oo](?=\d|\b)|(?<=\s)[@Oo](?=\d)", "0", text)
    return [float(re.sub(r"[,\s]", "", m.replace(" ", ".", 1) if "." not in m else m).replace(",", ""))
            for m in MONEY.findall(text)]


def _transactions(lines: List[Line]):
    """Table rows that start with a transaction date (310725 or 29JUL24) and end with an amount;
    returns [(date, [amounts...], line)] in page order, up to 'END OF STATEMENT' / closing line."""
    rows = []
    for ln in lines:
        if END.search(ln.text):
            break
        m = TXN_DATE.match(ln.text)
        if not m:
            continue
        try:
            if m[1]:
                d = date(2000 + int(m[3]), int(m[2]), int(m[1]))
            else:
                d = date(2000 + int(m[6]), MONTHS[m[5][:3].lower()], int(m[4]))
        except ValueError:
            continue
        _, amount = split_amount(ln)
        values = row_amounts(amount)
        if values:
            rows.append((d, values, ln))
    return rows


def _running_balances(lines: List[Line]) -> List[float]:
    """Last amount of every table row ABOVE the closing-balance line (the running balance column).
    Lines with totals or the closing balance itself are not running balances."""
    out = []
    for ln in lines:
        if LABELS["closing"].search(ln.text):
            break
        if LABELS["deposits"].search(ln.text) or LABELS["withdrawals"].search(ln.text):
            continue
        _, amount = split_amount(ln)
        values = [p.value for t in amount if (p := parse_amount(t.text))]
        if values:
            out.append(values[-1])
    return out


def extract_bank_statement(tokens: Sequence, doc_id: str, engine: str = "tesseract-5",
                           processing_ms: Optional[int] = None,
                           to_original: Optional[Callable[[List[int]], List[int]]] = None,
                           languages: Optional[List[str]] = None) -> Tuple[OCROutput, dict]:
    lines = group_lines(tokens)
    found: Dict[str, Tuple[float, object, Line]] = {}
    for ln in lines:
        for key, pat in LABELS.items():
            if key not in found and pat.search(ln.text):
                p, toks = _amount(ln)
                if p is not None:
                    found[key] = (p.value, p, toks)

    checks, corrections, notes = [], {}, {}
    txns = _transactions(lines)
    closing_from_row = False
    if "closing" not in found and txns:
        # No 'closing balance' line (e.g. Bank of Ceylon prints '*** END OF STATEMENT ***'):
        # the balance column of the last transaction row is the closing balance.
        last = txns[-1]
        _, amount = split_amount(last[2])
        value = last[1][-1]
        found["closing"] = (value, parse_amount(f"{value:.2f}"), amount[-2:] if len(amount) >= 2 else amount)
        closing_from_row = True
    closing = found["closing"][0] if "closing" in found else None
    balances = _running_balances(lines)

    def rule_flow(c):
        o, dep, wd = (found.get(k, (None,))[0] for k in ("opening", "deposits", "withdrawals"))
        if None in (o, dep, wd, c):
            return None
        return abs(o + dep - wd - c) <= TOL

    def rule_last(c):
        if c is None or not balances:
            return None
        return abs(balances[-1] - c) <= TOL

    def rule_last_row(c):
        """Last row: previous balance +/- this row's amount = closing balance."""
        if not closing_from_row or len(txns) < 2 or len(txns[-1][1]) < 2:
            return None
        prev, amt = txns[-2][1][-1], txns[-1][1][-2]
        return min(abs(prev - amt - c), abs(prev + amt - c)) <= TOL

    def run_checks(c):  # noqa: C901
        out = []
        # The flow rule is hard (all four printed totals). The running-balance rule is soft: table rules
        # often garble the balance column, so a failure there is weak evidence.
        for name, fn, hard, text in [("opening + deposits - withdrawals = closing", rule_flow, True,
                                      "the closing balance equals the opening balance plus deposits minus withdrawals"),
                                     ("closing = last running balance", rule_last, False,
                                      "the closing balance appears as a running balance in the transaction table")]:
            r = fn(c)
            if r is not None:
                out.append({"rule": name, "hard": hard, "passed": r, "explanation": text})
        r = rule_last_row(c)
        if r is not None:
            out.append({"rule": "previous balance -/+ last amount = closing", "hard": False, "passed": r,
                        "explanation": "the last row's running balance follows from the row above it"})
        return out

    checks = run_checks(closing)
    hard_fail = [c for c in checks if c["hard"] and not c["passed"]]
    if closing is not None and hard_fail:
        fixes = [v for v, _ in misreadings(closing)
                 if all(c["passed"] for c in run_checks(v) if c["hard"])]
        if len(fixes) == 1:
            corrections["total_amount"] = {"ocr_value": closing, "corrected_value": fixes[0]}
            closing = fixes[0]
            checks = run_checks(closing)

    fields: List[ExtractedField] = []

    def add(name, value, raw, conf, toks, verified=False, corrected=False, contradicted=False, cleaned=False):
        c = field_confidence(conf, verified=verified, corrected=corrected, contradicted=contradicted, cleaned=cleaned)
        box = union_bbox(toks)
        fields.append(ExtractedField(field_name=name, value=value, raw_text=raw, confidence=c,
                                     bbox=[int(x) for x in (to_original(box) if to_original else box)],
                                     tax_entity=ONTOLOGY.get(name, {}).get("tax_entity"), needs_review=needs_review(c)))

    issuer, iln, where = _find_issuer(lines)
    if issuer:
        add("issuer", issuer, iln.text, _conf(iln.tokens) * (1.0 if where == "header" else 0.85), iln.tokens)
        notes["issuer"] = [f"bank identified from its name words in the {where} (list of Sri Lankan banks)"]
    d, dln = _find_date(lines)
    if d:
        add("document_date", d.isoformat(), dln.text, _conf(dln.tokens), dln.tokens)
    if txns:
        # The statement covers the transactions, which can be months before the print date.
        last_d, _, tln = txns[-1]
        add("pay_period", f"{last_d.year}-{last_d.month:02d}", tln.text, _conf(tln.tokens[:1]) * 0.9, tln.tokens[:1])
        notes["pay_period"] = ["statement month taken from the last transaction date"]
    elif d:
        add("pay_period", f"{d.year}-{d.month:02d}", dln.text, _conf(dln.tokens) * 0.9, dln.tokens)
        notes["pay_period"] = ["statement month taken from the statement date"]
    if "closing" in found:
        _, p, toks = found["closing"]
        hard = [c["passed"] for c in checks if c["hard"]]
        add("total_amount", closing, " ".join(t.text for t in toks), min(t.confidence for t in toks), toks,
            verified=bool(hard) and all(hard) and "total_amount" not in corrections,
            corrected="total_amount" in corrections, contradicted=bool(hard) and not all(hard),
            cleaned=p.corrected)
        notes["total_amount"] = ["closing balance of the statement" + (
            " (balance of the last transaction row; no closing-balance line printed)" if closing_from_row else "")]

    overall = round(sum(f.confidence for f in fields) / len(fields), 3) if fields else 0.0
    out = OCROutput(doc_id=doc_id, document_type="bank_statement", languages=languages or ["en"],
                    extracted_fields=fields, overall_confidence=overall,
                    needs_review=(not fields) or any(f.needs_review for f in fields),
                    engine=engine, processing_ms=processing_ms)
    report = {"checks": checks, "corrections": corrections, "notes": notes,
              "helper_values": {k: v[0] for k, v in found.items() if k != "closing"}}
    return out, report
