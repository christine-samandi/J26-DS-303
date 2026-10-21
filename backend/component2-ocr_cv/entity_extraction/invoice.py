"""Invoice, receipt and quotation extraction (FR4 for purchase documents: solar installations,
donation receipts, utility bills, shop bills) in English and Sinhala.

What it returns (contract v0.2):
- fields: issuer, document_number, document_date, due_date, seller_reg_no, subtotal, discount, vat,
  sscl, total_amount, amount_paid, balance_due, previous_balance
- line_items: every row of the items table, description exactly as printed
- tax_components: VAT / SSCL / GST parts with rate and amount
- attributes: printed key-value facts such as 'System Size 5 kW' (generic on purpose)

What it does NOT do: decide whether a purchase qualifies for tax relief (e.g. solar panels).
Components 1 and 3 do that. This module only reads the document faithfully, so the same code
serves every relief document type without per-type code.

Layout handling (own method):
- Two-column pages put a label far from its value, or two label/value pairs on one line
  ('Date of Invoice : ...    Invoice No. : ...'). Lines are split into SEGMENTS at large gaps; a
  label's value is the rest of its segment, else the numeric segment(s) to its right.
- The items table is found from its header row; each number is assigned to the column whose header
  it sits under, so 'Rate | Quantity' and 'Qty | Rate' layouts both work.
- A 'Total' row that adds up table columns (quantities in mixed units, rounded amounts) is not an
  item; only its last column is read, and the arithmetic checks decide whether it can be trusted.
- Dates: the page's own format hint ('(MM/DD/YYYY)') wins; otherwise day > 12 decides; otherwise
  the date is marked ambiguous (read as DD/MM, the Sri Lankan default) unless due date = issue date
  + payment terms settles it.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from api.schema import Attribute, ExtractedField, LineItem, OCROutput, TaxComponent
from entity_extraction.amounts import looks_numeric, parse_amount
from entity_extraction.confidence import field_confidence, needs_review
from entity_extraction.extractor import ISSUER_HINTS, Line, find_issuer, group_lines, split_amount, union_bbox
from entity_extraction.invoice_checks import run_checks
from entity_extraction.keywords import _phrase_score, normalise

SCHEMA_VERSION = "0.2"
AMOUNT_FIELDS = ("subtotal", "discount", "vat", "sscl", "total_amount", "amount_paid", "balance_due",
                 "previous_balance")
DATE_FIELDS = ("document_date", "due_date")
TEXT_FIELDS = ("document_number", "seller_reg_no")
TAX_HELPERS = {"tax:SGST": "SGST", "tax:CGST": "CGST", "tax:IGST": "IGST", "tax:GST": "GST"}

# Sinhala phrases are standard terms: TODO verify against real Sri Lankan bills and receipts.
INVOICE_LABELS: Dict[str, List[str]] = {
    "document_number": ["invoice no", "invoice number", "invoice #", "bill no", "bill number", "receipt no",
                        "receipt number", "quotation no", "quote no", "estimate no", "reference no", "ref no",
                        "ඉන්වොයිස් අංකය", "බිල්පත් අංකය", "ලදුපත් අංකය", "රිසිට් අංකය"],
    "document_date": ["date of invoice", "invoice date", "bill date", "receipt date", "tax date",
                      "quotation date", "date", "දිනය"],
    "due_date": ["due date", "payment due by", "pay on or before", "due by", "ගෙවිය යුතු දිනය"],
    "seller_reg_no": ["supplier s tin", "supplier tin", "tin", "tin no", "vat no", "vat reg no",
                      "vat registration no", "company registration no", "registration no", "reg no", "gstin",
                      "br no", "charity reg no", "ලියාපදිංචි අංකය"],
    "payment_terms": ["terms", "payment terms"],
    "subtotal": ["sub total", "subtotal", "total value of supply", "total before tax", "taxable value",
                 "charge for electricity consumed", "උප එකතුව"],
    "discount": ["discount", "වට්ටම"],
    "vat": ["vat", "vat amount", "total vat", "value added tax", "වැට්", "එකතු කළ අගය මත බද්ද"],
    "sscl": ["sscl", "s s c levy", "social security contribution levy"],
    "tax:SGST": ["sgst"], "tax:CGST": ["cgst"], "tax:IGST": ["igst"], "tax:GST": ["gst"],
    "total_amount": ["total", "grand total", "gross total", "net total", "total amount", "total payable",
                     "total charges for bill", "total charges for bill period", "amount", "the sum of",
                     "amount received", "donation amount", "මුදල", "පරිත්‍යාග මුදල",
                     "total amount consideration including vat", "total with tax", "total estimate",
                     "මුළු මුදල", "එකතුව", "මුළු එකතුව"],
    "amount_paid": ["advance", "amount paid", "paid", "payments", "total payments", "cash", "received",
                    "ගෙවූ මුදල", "අත්තිකාරම්"],
    "balance_due": ["balance", "balance due", "total due", "total due amount", "amount due",
                    "total amount due", "ශේෂය", "ගෙවිය යුතු මුදල"],
    "previous_balance": ["previous due amount", "previous due", "previous balance", "balance brought forward",
                         "පෙර ශේෂය"],
}
# Table header words -> column role. Two-word headers are matched before single words.
ROLE_WORDS: Dict[str, List[str]] = {
    "desc": ["description", "item", "items", "particulars", "package name", "details", "විස්තරය", "අයිතමය"],
    "type": ["type"],
    "qty": ["qty", "qnty", "quantity", "ප්‍රමාණය"],
    "unit": ["unit"],
    "rate": ["unit price", "unit cost", "rate", "price", "ඒකක මිල", "මිල"],
    "dis": ["dis", "disc", "discount"],
    "tax": ["gst", "vat", "tax"],
    "amount": ["total cost", "line total", "amount", "gross", "total", "මුදල", "එකතුව"],
}
NUMERIC_ROLES = ("qty", "rate", "dis", "tax", "amount")
PII_KEY = re.compile(r"name|address|phone|mobile|e-?mail|contact|location|nic\b|account|client|customer|"
                     r"subscription|contract|premises|walk order|නම|ලිපිනය|දුරකථන", re.I)
CURRENCY = re.compile(r"^(?:rs\.?|lkr\.?|£|\$|€|₹|රු\.?|ரூ\.?)", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
GAP_FACTOR, GAP_MIN = 2.0, 24       # a gap wider than 2 letter-heights starts a new segment
MAX_LABEL_WORDS = 9                 # longer runs of words are sentences, not labels
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], start=1)}
AMBIGUOUS_DATE_CAP = 0.8
SELLER_HINTS = re.compile(r"\b(foundation|trust|society|sabha|enterprises?|traders|solutions|installations?|"
                          r"engineering|electricals?|hardware|stores)\b", re.I)
TITLE_WORDS = {"invoice", "tax invoice", "receipt", "quotation", "estimate", "bill", "statement", "donation receipt"}
MIN_ATTR_CONF = 0.6
LETTERHEAD_LINES = 2                # seller name/address lines are not 'key | value' facts
BUYER = re.compile(r"\b(client|customer|bill to|ship to|deliver to|to:|sold to|ලබන්නා|පාරිභෝගික)", re.I)


@dataclass
class Cand:
    field: str
    value: object
    raw_text: str
    conf: float
    bbox: List[int]
    key: tuple
    line: int
    cleaned: bool = False
    notes: List[str] = field(default_factory=list)
    rate: Optional[float] = None
    alt: Optional[date] = None        # the other reading of an ambiguous date


# ---------------------------------------------------------------- small helpers
def _strip_currency(tokens: Sequence) -> list:
    out = []
    for t in tokens:
        m = CURRENCY.match(t.text)
        if m:
            rest = t.text[m.end():].strip()
            if rest and not looks_numeric(rest):
                out.append(t)
                continue
            if not rest:
                continue
            t = copy.copy(t)
            t.text = rest
        out.append(t)
    return out


def segments(line: Line) -> List[list]:
    toks = line.tokens
    heights = sorted(t.bbox[3] for t in toks)
    gap = max(GAP_MIN, GAP_FACTOR * heights[len(heights) // 2])
    segs = [[toks[0]]]
    for a, b in zip(toks, toks[1:]):
        (segs.append([b]) if b.bbox[0] - (a.bbox[0] + a.bbox[2]) > gap else segs[-1].append(b))
    return segs


def _text(toks) -> str:
    return " ".join(t.text for t in toks)


def _conf(toks) -> float:
    return round(min(t.confidence for t in toks), 4) if toks else 0.0


def _is_num(t) -> bool:
    return looks_numeric(t.text) or bool(PERCENT.fullmatch(t.text.strip()))


def _numeric_seg(seg) -> bool:
    return bool(seg) and all(_is_num(t) for t in seg)


def head(seg) -> Tuple[list, list]:
    """(label tokens, value tokens): the label ends at a ':' or before the first token with a digit."""
    for i, t in enumerate(seg):
        if any(c.isdigit() for c in t.text):
            return seg[:i], seg[i:]
        if t.text.endswith(":") or t.text in (":", ":-", "-"):
            return seg[:i + 1], seg[i + 1:]
    return seg, []


def match_label(text: str) -> Optional[Tuple[str, tuple, str]]:
    """Best (field, key, phrase). key = (anchored, score, length): a label that STARTS with the phrase
    beats one that only contains it ('VAT Amount (Total Value of Supply @ VAT Rate)' is VAT)."""
    words = normalise(text).split()
    if not words or len(words) > MAX_LABEL_WORDS:
        return None
    best = None
    for f, phrases in INVOICE_LABELS.items():
        for p in phrases:
            pw = normalise(p).split()
            s = _phrase_score(words, " ".join(pw))
            if s < 0.8:
                continue
            anchored = _phrase_score(words[:len(pw)], " ".join(pw)) >= 0.8
            if not anchored and len(pw) == 1 and len(words) > 2:
                continue        # 'Expected Start Date' is not the document date
            key = (anchored, round(s + 0.02 * len(pw), 4), len(p))
            if best is None or key > best[1]:
                best = (f, key, p)
    return best


def parse_date(text: str, order: Optional[str]) -> Tuple[Optional[date], bool, Optional[date]]:
    """Returns (date, ambiguous, other reading). order: 'mdy', 'dmy' or None."""
    t = text.lower()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", t)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])), False, None
        m = re.search(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})", t)
        if m:
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            y += 2000 if y < 100 else 0
            if order == "mdy" or (order is None and b > 12):
                return date(y, a, b), False, None
            if order == "dmy" or a > 12 or a == b:
                return date(y, b, a), False, None
            other = None
            try:
                other = date(y, a, b)
            except ValueError:
                pass
            return date(y, b, a), other is not None, other
        m = re.search(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", t)        # January 15, 2025
        if m and m[1] in MONTHS:
            return date(int(m[3]), MONTHS[m[1]], int(m[2])), False, None
        m = re.search(r"(\d{1,2})\s+([a-z]{3})[a-z]*\.?,?\s+(\d{4})", t)        # 15 Jan 2025
        if m and m[2] in MONTHS:
            return date(int(m[3]), MONTHS[m[2]], int(m[1])), False, None
    except ValueError:
        return None, False, None
    return None, False, None


# ---------------------------------------------------------------- items table
def _roles_in(line: Line) -> Dict[str, Tuple[float, float]]:
    """Column role -> (x centre, left edge) for a header line, or {} if it is not a header."""
    toks = line.tokens
    words = [normalise(t.text) for t in toks]
    used, roles = set(), {}
    for size in (2, 1):
        for i in range(len(toks) - size + 1):
            if any(j in used for j in range(i, i + size)):
                continue
            phrase = " ".join(words[i:i + size])
            for role, names in ROLE_WORDS.items():
                if role in roles or phrase not in names:
                    continue
                span = toks[i:i + size]
                x0, x1 = span[0].bbox[0], span[-1].bbox[0] + span[-1].bbox[2]
                roles[role] = ((x0 + x1) / 2, x0)
                used.update(range(i, i + size))
                break
    if "amount" not in roles or len({"desc", "qty", "rate"} & set(roles)) < 1 or len(roles) < 2:
        return {}
    if any(looks_numeric(t.text) for t in toks):
        return {}
    return roles


def _stop_row(line: Line) -> bool:
    label, _ = head(segments(line)[0])
    m = match_label(_text(label)) if label else None
    lowered = normalise(line.text)
    return bool(m and m[1][0] and m[0] not in ("document_number", "document_date", "seller_reg_no")) \
        or lowered.startswith(("amount in words", "terms", "payment schedule", "ගෙවීම්"))


def parse_table(lines: List[Line]) -> Tuple[List[dict], set, dict]:
    """Returns (items, consumed line indexes, report)."""
    for hi, ln in enumerate(lines):
        cols = _roles_in(ln)
        if cols:
            break
    else:
        return [], set(), {"found": False}
    consumed, items = {hi}, []
    num_cols = {r: x for r, (x, _) in cols.items() if r in NUMERIC_ROLES}
    text_cols = {r: x for r, (x, _) in cols.items() if r not in NUMERIC_ROLES}
    desc_left = cols.get("desc", (0, 0))[1]
    # numbers left of the first numeric column belong to the description ('Solar PV Panel 550W', 'Item 1')
    first_num_left = min((x0 for r, (_, x0) in cols.items() if r in NUMERIC_ROLES), default=0)
    report = {"found": True, "header": ln.text, "columns": sorted(cols, key=lambda r: cols[r][0])}
    for i in range(hi + 1, len(lines)):
        row = lines[i]
        if _stop_row(row):
            report["ended_at"] = row.text
            if len([s for s in segments(row) if _numeric_seg(s)]) >= 2:
                report["column_total_row"] = ("a 'Total' row adds up table columns (mixed units, rounding); "
                                              "only its last column is read as a total")
            break
        toks = list(row.tokens)
        if toks and toks[0].text.strip(".").isdigit() and len(toks[0].text) <= 3 and "desc" in cols \
                and toks[0].bbox[0] + toks[0].bbox[2] <= desc_left + 2:
            toks = toks[1:]                                  # row number '1', '2' ...
        cells: Dict[str, list] = {}
        desc, unit = [], []
        for t in toks:
            cx = t.bbox[0] + t.bbox[2] / 2
            if _is_num(t) and num_cols and cx >= first_num_left:
                role = min(num_cols, key=lambda r: abs(num_cols[r] - cx))
                if PERCENT.fullmatch(t.text.strip()) and "tax" in num_cols:
                    role = "tax"
                cells.setdefault(role, []).append(t)
            else:
                every = {**num_cols, **text_cols}
                nearest = min(every, key=lambda r: abs(every[r] - cx)) if every else "desc"
                (unit if nearest == "unit" else desc).append(t)
        if not cells:
            if desc and items:                              # wrapped description
                items[-1]["description"] += " " + _text(desc)
                items[-1]["tokens"] += desc
                consumed.add(i)
            continue
        item = {"description": _text(desc), "unit": _text(unit) or None, "tokens": toks}
        for role, ts in cells.items():
            raw = " ".join(t.text for t in ts)
            pm = PERCENT.search(raw)
            if role == "tax":
                item["tax_rate"] = float(pm[1]) / 100 if pm else None
                continue
            p = parse_amount(raw)
            key = {"qty": "quantity", "rate": "unit_price", "dis": "discount", "amount": "amount"}[role]
            item[key] = p.value if p else None
            item["cleaned"] = item.get("cleaned", False) or bool(p and p.corrected)
        items.append(item)
        consumed.add(i)
    report["rows"] = len(items)
    return items, consumed, report


# ---------------------------------------------------------------- fields and attributes
def find_fields(lines: List[Line], skip: set, order: Optional[str],
                table_start: Optional[int] = None) -> Tuple[Dict[str, Cand], List[Attribute], List[int]]:
    best: Dict[str, Cand] = {}
    attributes: List[Attribute] = []
    field_lines: List[int] = []

    def offer(c: Cand):
        old = best.get(c.field)
        if old is None or c.key > old.key:
            best[c.field] = c

    for li, ln in enumerate(lines):
        if li in skip:
            continue
        segs = segments(ln)
        used = set()
        for si, seg in enumerate(segs):
            if si in used:
                continue
            label, rest = head(seg)
            m = match_label(_text(label)) if label else None
            if not m:
                continue
            f, key, phrase = m
            nxt = []
            if rest and all(PERCENT.fullmatch(t.text.strip()) for t in rest):
                rest = []         # 'VAT 18%': the rate is part of the label; the amount is further right
            if not rest:          # value in the next segment(s), unless that is another label
                j = si + 1
                while j < len(segs) and not (head(segs[j])[0] and match_label(_text(head(segs[j])[0]))):
                    nxt.append(j)
                    if not _numeric_seg(segs[j]):
                        break
                    j += 1
            if f in AMOUNT_FIELDS or f in TAX_HELPERS:
                if rest:
                    _, amt = split_amount(Line([t for t in seg if not PERCENT.fullmatch(t.text.strip())]))
                else:
                    nums = [j for j in nxt if _numeric_seg(segs[j])]
                    amt = segs[nums[-1]] if nums else []
                    amt = [t for t in amt if not PERCENT.fullmatch(t.text.strip())]
                if not amt:
                    continue
                p = parse_amount(" ".join(t.text for t in amt))
                if p is None:
                    continue
                pm = PERCENT.search(ln.text)
                single = len([j for j in nxt if _numeric_seg(segs[j])]) < 2
                c = Cand(f, p.value, _text(amt), _conf(amt), union_bbox(amt), key[:1] + (single,) + key[1:], li,
                         cleaned=p.corrected,
                         rate=float(pm[1]) / 100 if pm and f not in ("total_amount", "subtotal") else None)
                if p.corrected:
                    c.notes.append(f"amount text cleaned: '{c.raw_text}' -> {p.value:,.2f}")
                if not single:
                    c.notes.append("several numbers on the row (column totals); the last one was read")
                offer(c)
            else:
                val = rest or (segs[nxt[0]] if nxt else [])
                if not val:
                    continue
                if f in DATE_FIELDS:
                    d, amb, other = parse_date(_text(val), order)
                    if d is None:
                        continue
                    c = Cand(f, d, _text(val), _conf(val), union_bbox(val), key[:1] + (True,) + key[1:], li,
                             alt=other if amb else None)
                    if amb:
                        c.notes.append(f"day and month both <= 12; read as DD/MM ({d.isoformat()}), "
                                       f"could be {other.isoformat()}")
                    offer(c)
                elif f == "payment_terms":
                    dm = re.search(r"(\d{1,3})\s*days?", _text(val), re.I)
                    if dm:
                        offer(Cand(f, int(dm[1]), _text(val), _conf(val), union_bbox(val), key[:1] + (True,) + key[1:], li))
                else:
                    tok = val[0]
                    if not any(ch.isdigit() for ch in tok.text) or "\u25cf" in tok.text:
                        continue
                    offer(Cand(f, tok.text.strip(":;,"), tok.text, _conf([tok]), list(tok.bbox), key[:1] + (True,) + key[1:], li))
            used.update([si] + nxt)
            field_lines.append(li)
        # printed key-value facts that are not standard fields ('System Size | 5 kW')
        free = [s for k, s in enumerate(segs) if k not in used]
        in_header_block = LETTERHEAD_LINES <= li and (table_start is None or li < table_start)
        attributes += _attributes(free, pairs_allowed=in_header_block)
    return best, attributes, field_lines


def _attributes(segs: List[list], pairs_allowed: bool = True) -> List[Attribute]:
    out: List[Attribute] = []
    pairs = []
    for s in segs:
        txt = _text(s)
        if ":" in txt and not txt.strip().endswith(":") and not re.match(r"^\d", txt):
            k, v = txt.split(":", 1)
            pairs.append((k, v, s))
    if not pairs and pairs_allowed and len(segs) >= 2 and len(segs) % 2 == 0:
        for k_seg, v_seg in zip(segs[::2], segs[1::2]):
            kt = _text(k_seg)
            if any(c.isdigit() for c in kt) or len(kt.split()) > 5:
                return out
            pairs.append((kt, _text(v_seg), v_seg))
    for k, v, s in pairs:
        k, v = k.strip(" :-|"), v.strip(" :-|")
        letters = sum(ch.isalpha() for ch in k)
        if not k or not v or PII_KEY.search(k) or "\u25cf" in v or match_label(k) or match_label(v) \
                or letters < 3 or letters < 0.5 * len(k.replace(" ", "")) or sum(ch.isalnum() for ch in v) < 2 \
                or _conf(s) < MIN_ATTR_CONF:
            continue
        out.append(Attribute(key=k, value=v, confidence=_conf(s), bbox=union_bbox(s)))
    return out


def seller(lines: List[Line]):
    """Seller = a company-like name near the top (first column only: letterheads share lines with
    titles); never the buyer's block and never the document title ('INVOICE')."""
    firsts = [Line(segments(ln)[0]) for ln in lines if not BUYER.search(ln.text)]
    firsts = [ln for ln in firsts if normalise(ln.text) not in TITLE_WORDS]
    # A company line keeps its masked part ('●●●●●● Broadband Networks (Pvt) Ltd'): the PII guard
    # may hide a brand word that looks like a name; the visible part still identifies the seller type.
    for ln in firsts[: max(6, len(firsts) // 3)]:
        name = re.sub(r"\(\s*pv\s*\d+\s*\)|\bpv\s*\d+\b", "", ln.text, flags=re.I).strip(" -|,")
        if (ISSUER_HINTS.search(name) or SELLER_HINTS.search(name)) and not match_label(name):
            from entity_extraction.extractor import Candidate
            return Candidate("issuer", name, ln.text, ocr_confidence=_conf(ln.tokens), bbox=union_bbox(ln.tokens))
    return find_issuer([ln for ln in firsts if "\u25cf" not in ln.text])


def date_order(tokens: Sequence) -> Optional[str]:
    text = re.sub(r"[il|\\]", "/", " ".join(t.text for t in tokens).lower().replace(" ", ""))
    if re.search(r"mm[/.-]dd[/.-]yy", text):
        return "mdy"
    if re.search(r"dd[/.-]mm[/.-]yy", text):
        return "dmy"
    return None


# ---------------------------------------------------------------- amount in words
WORD_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
WORD_NUM.update({w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})
SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "million": 1_000_000}


def words_to_number(text: str) -> Optional[float]:
    """'Twenty Five Thousand Rupees and Fifty Cents Only' -> 25000.50 (English only; Sinhala TODO)."""
    words = re.sub(r"[^a-z ]", " ", text.lower().replace("-", " ")).split()
    rupees, cents, part, total, seen = 0, 0, 0, 0, False
    for w in words:
        if w in WORD_NUM:
            part += WORD_NUM[w]
            seen = True
        elif w == "hundred":
            part *= 100
        elif w in SCALES:
            total += part * SCALES[w]
            part = 0
        elif w in ("cents", "cent") and seen:
            cents, part = part, 0
        elif w in ("rupees", "rupee", "and"):
            if w.startswith("rupee"):
                rupees, total, part = total + part, 0, 0
    if not seen:
        return None
    if not rupees:
        rupees, part = total + part, 0
    return round(rupees + cents / 100, 2)


def amount_in_words(lines: List[Line]) -> Optional[Tuple[float, str]]:
    for ln in lines:
        n = normalise(ln.text)
        if n.startswith(("amount in words", "rupees", "the sum of rupees", "sum of rupees")):
            words = ln.text.split(":", 1)[1] if ":" in ln.text else ln.text
            value = words_to_number(re.sub(r"(?i)amount in words", "", words))
            if value:
                return value, ln.text
    return None


# ---------------------------------------------------------------- main entry
def extract_invoice(tokens: Sequence, doc_id: str, engine: str = "tesseract-5",
                    processing_ms: Optional[int] = None,
                    to_original: Optional[Callable[[List[int]], List[int]]] = None,
                    document_type: str = "invoice", languages: Optional[List[str]] = None) -> Tuple[OCROutput, dict]:
    from entity_extraction.payslip import ONTOLOGY, detect_languages

    back = (lambda b: [int(x) for x in (to_original(b) if to_original else b)])
    toks = _strip_currency(tokens)
    lines = group_lines(toks)
    order = date_order(toks)
    items, consumed, table = parse_table(lines)
    cands, attributes, _ = find_fields(lines, consumed, order, table_start=min(consumed) if consumed else None)
    notes: Dict[str, List[str]] = {}

    issuer = seller(lines)
    if issuer:
        cands["issuer"] = Cand("issuer", issuer.value, issuer.raw_text, issuer.ocr_confidence, issuer.bbox,
                               (True, True, 1.0, 0), -1, notes=list(issuer.notes))

    # taxes: VAT and SSCL are fields AND components; GST parts are components only
    taxes: Dict[str, Cand] = {}
    for f, name in [("vat", "VAT"), ("sscl", "SSCL")] + list(TAX_HELPERS.items()):
        if f in cands:
            taxes[name] = cands[f]
    if "GST" in taxes and ("SGST" in taxes or "CGST" in taxes):
        taxes.pop("GST")            # 'GST' total next to its SGST/CGST parts would be counted twice

    values = {f: float(c.value) for f, c in cands.items() if f in AMOUNT_FIELDS}
    result = run_checks(values, items, {n: float(c.value) for n, c in taxes.items()})
    for f, (old, new) in result.corrections.items():
        name = f.split(":", 1)[1] if f.startswith("tax:") else None
        target = taxes[name] if name else cands[f]
        target.notes.append(f"self-correction: OCR read {old:,.2f}; the document's arithmetic gives {new:,.2f}")
        target.value = new
        if name and name in ("VAT", "SSCL"):
            cands[name.lower()].value = new
    verified = set(result.verified) | {n.lower() for n in taxes if f"tax:{n}" in result.verified}
    contradicted = set(result.contradicted) | {n.lower() for n in taxes if f"tax:{n}" in result.contradicted}
    corrected = set(result.corrections) | {n.lower() for n in taxes if f"tax:{n}" in result.corrections}

    # due date = issue date + terms resolves an ambiguous DD/MM vs MM/DD reading
    checks = [c.to_dict() for c in result.checks]
    if "document_date" in cands and "due_date" in cands:
        issue, due = cands["document_date"], cands["due_date"]
        if due.alt and due.value < issue.value <= due.alt:
            due.value, due.alt = due.alt, None
            due.notes.append("read as MM/DD: a due date cannot be before the issue date")
    terms = cands.pop("payment_terms", None)
    if terms and "document_date" in cands and "due_date" in cands:
        issue, due = cands["document_date"], cands["due_date"]
        options = [(i, d) for i in [issue.value, issue.alt] if i for d in [due.value, due.alt] if d]
        ok = [(i, d) for i, d in options if (d - i).days == terms.value]
        if ok:
            issue.value, due.value = ok[0]
            for c in (issue, due):
                if c.alt:
                    c.notes.append(f"date order settled by the payment terms ({terms.value} days)")
                c.alt = None
        checks.append({"rule": "due date = issue date + terms", "hard": False, "passed": bool(ok),
                       "explanation": f"terms {terms.value} days: " + ("consistent" if ok else "inconsistent")})

    in_words = amount_in_words(lines)
    if in_words and "total_amount" in cands:
        ok = abs(in_words[0] - float(cands["total_amount"].value)) < 0.011
        checks.append({"rule": "amount in words = amount in figures", "hard": True, "passed": ok,
                       "explanation": f"words say {in_words[0]:,.2f}, figures say {float(cands['total_amount'].value):,.2f}"})
        if ok:
            verified.add("total_amount")
            contradicted.discard("total_amount")
        elif "total_amount" not in verified:
            contradicted.add("total_amount")

    out_fields: List[ExtractedField] = []
    for f, c in sorted(cands.items(), key=lambda kv: kv[1].line):
        if f in TAX_HELPERS:
            continue
        conf = field_confidence(c.conf, verified=f in verified and f not in corrected, corrected=f in corrected,
                                contradicted=f in contradicted, cleaned=c.cleaned)
        if c.alt:
            conf = min(conf, AMBIGUOUS_DATE_CAP)
        value = c.value.isoformat() if isinstance(c.value, date) else c.value
        out_fields.append(ExtractedField(field_name=f, value=value, raw_text=c.raw_text, confidence=conf,
                                         bbox=back(c.bbox), tax_entity=ONTOLOGY.get(f, {}).get("tax_entity"),
                                         needs_review=needs_review(conf)))
        if c.notes:
            notes[f] = c.notes

    line_items: List[LineItem] = []
    for n, (it, r) in enumerate(zip(items, result.items), 1):
        nums = [t for t in it["tokens"] if _is_num(t)] or it["tokens"]
        conf = field_confidence(_conf(nums), verified=bool(r.passed), corrected=False,
                                contradicted=r.passed is False, cleaned=it.get("cleaned", False))
        line_items.append(LineItem(description=it["description"], quantity=it.get("quantity"), unit=it.get("unit"),
                                   unit_price=it.get("unit_price"), discount=it.get("discount"),
                                   tax_rate=it.get("tax_rate"), amount=it.get("amount"),
                                   amount_includes_tax=r.includes_tax, confidence=conf,
                                   bbox=back(union_bbox(it["tokens"])), needs_review=needs_review(conf)))
    tax_components = []
    for name, c in taxes.items():
        k = f"tax:{name}"
        conf = field_confidence(c.conf, verified=k in result.verified,
                                corrected=k in result.corrections, contradicted=k in result.contradicted,
                                cleaned=c.cleaned)
        tax_components.append(TaxComponent(name=name, rate=c.rate, amount=float(c.value), confidence=conf,
                                           bbox=back(c.bbox), needs_review=needs_review(conf)))
    attributes = [a.model_copy(update={"bbox": back(a.bbox)}) for a in attributes]

    confs = [x.confidence for x in out_fields] + [x.confidence for x in line_items]
    overall = round(sum(confs) / len(confs), 3) if confs else 0.0
    out = OCROutput(schema_version=SCHEMA_VERSION, doc_id=doc_id, document_type=document_type,
                    languages=languages if languages is not None else detect_languages(tokens),
                    extracted_fields=out_fields, overall_confidence=overall,
                    needs_review=(not out_fields) or any(x.needs_review for x in out_fields + line_items),
                    engine=engine, processing_ms=processing_ms,
                    line_items=line_items or None, tax_components=tax_components or None,
                    attributes=attributes or None)
    report = {
        "checks": checks,
        "corrections": {f.replace("tax:", ""): {"ocr_value": a, "corrected_value": b}
                        for f, (a, b) in result.corrections.items()},
        "notes": notes,
        "helper_values": {"payment_terms_days": terms.value} if terms else {},
        "table": table,
        "date_order": order or "not printed; DD/MM assumed when ambiguous",
    }
    return out, report
