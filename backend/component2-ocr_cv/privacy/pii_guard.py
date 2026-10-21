"""PII guard: detect and mask personal identifiers immediately after OCR (own method, NFR4).

Why this module exists:
- Payslips and bank statements carry the person's name, NIC, address, account and phone numbers.
  None of these are needed for tax extraction, so they should never travel further than the OCR
  step: not into the JSON for Components 1/3, not into the explanation, logs or the drawn image.
- Masking at the source ("privacy by design", data minimisation) is safer than cleaning outputs
  later: no later step can leak what it never receives.

Method (explainable rules tuned to Sri Lankan documents, no extra model):
  1. Fixed patterns anywhere: old NIC (9 digits + V/X), new NIC (12 digits starting 19/20),
     mobile/landline numbers (07x / +94 / 0xx), card numbers (16 digits, Luhn check), and long
     identifier runs (9+ plain digits: account, cheque, reference numbers; amounts keep their
     separators so they are never matched).
  2. Labelled values: everything after a personal label on the same line
     (Name, NIC, TIN, Account No, Address, Mr/Mrs/Miss, Sinhala නම / ජා.හැ. අංකය / ගිණුම් අංකය ...).
  3. Sri Lankan name pattern: two or more initials followed by a name ('J T MITHUM', 'W.A.D. Premalal').
     Works even when OCR breaks the title (MR read as BR).
  4. Address blocks near the top of the page: lines starting 'No. 127/...' or with road words,
     the name line just above them and the place lines that follow (organisation lines such as
     the bank's own name are kept).
  5. Every other occurrence on the page of a word already masked as a name or address.
Masked tokens keep their position (bbox) but their text becomes '●'; the report lists only the
KIND and COUNT of what was masked plus the masked regions, never the values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

MASK = "●"

NIC_OLD = re.compile(r"^\d{9}[VvXx]$")
NIC_NEW = re.compile(r"^(19|20)\d{10}$")
PHONE = re.compile(r"(?:\+94|0094|\b0)\s?(?:7\d|[1-9]\d)[\s-]?\d{3}[\s-]?\d{4}\b")
LONG_ID = re.compile(r"^#?\d{9,}$")
HEADER_ID = re.compile(r"^#?\d{8,}$")          # 8+ digits in the page header: account / customer numbers
INITIAL = re.compile(r"^(?:[A-Z][.,]?|[A-Z](?:\.[A-Z])+\.?|\d)$")   # 'J', 'T.', 'W.A.D.' (OCR may read J as 9)
NAME_WORD = re.compile(r"^[A-Z][A-Za-z]{2,}$")
CARD = re.compile(r"(?:\d[ -]?){15}\d")

# Personal labels: the rest of the line after them is personal data.
LABELS = [
    ("name", r"(employee\s+)?name|නම|பெயர்|mr\.?|mrs\.?|miss|ms\.?"),
    ("nic", r"nic(\s*no\.?)?|n\.?i\.?c\.?|national\s+identity(\s+card)?(\s+no\.?)?|ජා\.?\s*හැ\.?(\s*අංකය)?|ජාතික\s+හැඳුනුම්(පත්)?(\s+අංකය)?"),
    ("tin", r"tin(\s*no\.?)?|taxpayer\s+identification\s+number"),
    ("account_no", r"(account|acc|a/c)\.?\s*(no\.?|number)|ගිණුම්\s+අංකය"),
    ("address", r"address|ලිපිනය|முகவரி"),
    ("person_no", r"number|අංකය|employee\s+(no\.?|number)|emp\s*no\.?"),
]
LABEL_RE = [(kind, re.compile(rf"^\W*({pat})\s*[:\-]?\s*$", re.I)) for kind, pat in LABELS]
ADDRESS_LINE = re.compile(r"^\W*no\.?\s*\d+(/\d+)*|\b(road|rd|mawatha|lane|place|street|avenue|para|watta|junction|"
                          r"පාර|මාවත|வீதி)\b", re.I)
KEEP_WORDS = re.compile(r"\b(bank|plc|ltd|pvt|limited|branch|statement|account|date|page|currency|lkr|"
                        r"pay\s*slip|payslip|salary|department|balance|total|"
                        # document titles on purchase documents: 'TAX INVOICE' under a seller's address
                        # is not a place name (it was masked, then every 'Invoice' label was masked too)
                        r"invoice|receipt|quotation|estimate|tax|vat|donation)\b", re.I)


@dataclass
class PIIReport:
    counts: Dict[str, int] = field(default_factory=dict)
    regions: List[List[int]] = field(default_factory=list)      # [x, y, w, h] of masked tokens

    def add(self, kind: str, tokens) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.regions += [list(t.bbox) for t in tokens]

    def to_dict(self) -> dict:
        return {"masked": dict(sorted(self.counts.items())), "regions": len(self.regions),
                "method": "rule-based Sri Lankan PII guard applied right after OCR; values are never reported"}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d) * (2 if alt else 1)
        total += n - 9 if n > 9 else n
        alt = not alt
    return total % 10 == 0


def _lines(tokens) -> List[List[int]]:
    """Token indices per line (OCR line id when present, else by vertical position)."""
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        key = getattr(t, "line_id", "") or f"y{round((t.bbox[1] + t.bbox[3] / 2) / max(t.bbox[3], 1))}"
        groups.setdefault(key, []).append(i)
    rows = [sorted(ix, key=lambda i: tokens[i].bbox[0]) for ix in groups.values()]
    return sorted(rows, key=lambda ix: min(tokens[i].bbox[1] for i in ix))


def _is_amount_or_date(text: str) -> bool:
    return bool(re.fullmatch(r"[\d,.\-/]+", text)) and bool(re.search(r"[.,/]", text)) and not LONG_ID.match(text)


def mask_tokens(tokens: Sequence) -> Tuple[list, PIIReport]:
    """Return (tokens with personal data masked, report). Input tokens are not modified."""
    toks = list(tokens)
    report = PIIReport()
    hide: Dict[int, str] = {}

    def mark(indices, kind):
        new = [i for i in indices if i not in hide and toks[i].text.strip()]
        if new:
            for i in new:
                hide[i] = kind
            report.add(kind, [toks[i] for i in new])

    lines = _lines(toks)
    header_rows = max(6, int(len(lines) * 0.4))
    # 1. Fixed patterns, token by token (and pairs/triples for numbers OCR split with spaces)
    for n, ix in enumerate(lines):
        texts = [toks[i].text for i in ix]
        for j, i in enumerate(ix):
            # Trim punctuation at the EDGES only: an amount keeps its inner ',' and '.', so
            # '460,786.24' can never look like the identifier '46078624'.
            t = texts[j].strip(".,:;()[]'\"-|")
            if NIC_OLD.match(t) or NIC_NEW.match(t):
                mark([i], "nic")
            elif LONG_ID.match(t) or (n < header_rows and HEADER_ID.match(t)):
                mark([i], "long_id")
        joined = " ".join(texts)
        for pat, kind in ((PHONE, "phone"), (CARD, "card")):
            for m in pat.finditer(joined):
                if kind == "card" and not _luhn(re.sub(r"\D", "", m.group())):
                    continue
                pos, covered = 0, []
                for j, w in enumerate(texts):
                    start = joined.find(w, pos)
                    end = start + len(w)
                    pos = end
                    if start < m.end() and end > m.start():
                        covered.append(ix[j])
                mark(covered, kind)

    # 2. Labelled values: 'Name : X Y', 'NIC No : ...', 'ගිණුම් අංකය : ...'
    for ix in lines:
        for j in range(len(ix)):
            for k in range(1, 4):                                   # labels of up to 3 words
                if j + k > len(ix):
                    break
                label = " ".join(toks[i].text for i in ix[j:j + k])
                kind = next((kd for kd, rx in LABEL_RE if rx.match(label)), None)
                if kind:
                    rest = [i for i in ix[j + k:] if toks[i].text.strip(" :-")]
                    if kind == "person_no" and not any(re.search(r"[^\W\d_]", toks[i].text) for i in rest):
                        rest = []                                    # 'Number : 12345' is an employee id, kept
                    mark([i for i in rest if not _is_amount_or_date(toks[i].text)], kind)
                    break

    # 3. Sri Lankan name pattern in the page header: initials followed by a capitalised name
    for ix in lines[:header_rows]:
        j = 0
        while j < len(ix):
            k = j
            while k < len(ix) and INITIAL.match(toks[ix[k]].text.strip("-,")):
                k += 1
            if k - j >= 2 or (k - j == 1 and "." in toks[ix[j]].text):
                names = []
                while k < len(ix) and NAME_WORD.match(toks[ix[k]].text.strip(".,")) and not KEEP_WORDS.search(toks[ix[k]].text):
                    names.append(ix[k])
                    k += 1
                if names:
                    mark(ix[j:k], "name")
            j = max(k, j + 1)

    # 4. Address block in the top part of the page
    top = lines[:header_rows]
    for n, ix in enumerate(top):
        text = " ".join(toks[i].text for i in ix)
        if not ADDRESS_LINE.search(text):
            continue
        block = [n]
        if n > 0:                                                   # the name line just above
            prev = " ".join(toks[i].text for i in lines[n - 1])
            if not KEEP_WORDS.search(prev) and not re.search(r"\d", prev) and len(prev.split()) <= 5:
                block.insert(0, n - 1)
        for m in range(n + 1, min(n + 5, len(lines))):             # place lines that follow (may pass the header)
            nxt = " ".join(toks[i].text for i in lines[m])
            if KEEP_WORDS.search(nxt) or re.search(r"\d[\d,]*\.\d{2}", nxt) or len(nxt.split()) > 5:
                break
            block.append(m)
        for b in block:
            words = [i for i in lines[b] if not KEEP_WORDS.search(toks[i].text) and not _is_amount_or_date(toks[i].text)]
            is_addr = b != block[0] or ADDRESS_LINE.search(" ".join(toks[i].text for i in lines[b]))
            mark(words, "address" if is_addr else "name")

    # 5. Other occurrences of words already masked as a name or address (e.g. the town repeated lower down)
    learned = {toks[i].text.strip(".,").lower() for i, kd in hide.items()
               if kd in ("name", "address") and len(toks[i].text.strip(".,")) >= 4 and toks[i].text.strip(".,").isalpha()}
    repeats = [i for i, t in enumerate(toks) if i not in hide and t.text.strip(".,").lower() in learned]
    if repeats:
        mark(repeats, "repeat")

    masked = [replace(t, text=MASK * max(1, len(t.text))) if i in hide else t for i, t in enumerate(toks)]
    return masked, report
