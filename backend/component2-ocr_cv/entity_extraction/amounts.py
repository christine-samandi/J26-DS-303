"""Turn OCR'd money text into numbers (post-OCR correction for financial text).

Why this module exists:
- OCR confuses look-alike characters inside numbers (O/0, l/1, S/5) and mixes up the
  decimal point and thousands separators ("30,000,00"). Fixing these with rules that only
  apply INSIDE numbers removes a large share of amount errors without touching words.
- Every correction is recorded, so the final JSON can say the value was cleaned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Look-alike letters that OCR returns instead of digits.
CONFUSIONS = {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1",
              "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9"}
CURRENCY = re.compile(r"(rs\.?|lkr|රු\.?|ரூ\.?|/=|=)", re.IGNORECASE)


@dataclass
class ParsedAmount:
    value: float
    raw: str            # text as OCR read it
    cleaned: str        # text after character fixes, before separator handling
    corrected: bool     # True if a look-alike character or a misplaced decimal comma was fixed


def looks_numeric(text: str) -> bool:
    """A token is treated as part of an amount if, after look-alike fixes, it is mostly digits
    and has at least one real digit (so words like 'Basic' are never converted)."""
    t = CURRENCY.sub("", text).strip()
    if not t or t.endswith("%") or not any(c.isdigit() for c in t):
        return False
    mapped = "".join(CONFUSIONS.get(c, c) for c in t)
    core = [c for c in mapped if c not in ".,()-+ "]
    return bool(core) and sum(c.isdigit() for c in core) / len(core) >= 0.75


def parse_amount(text: str) -> Optional[ParsedAmount]:
    """'Rs. 150,000.00' -> 150000.0 ; '30,000,00' -> 30000.0 ; 'l2,OOO.OO' -> 12000.0 .
    Rule for separators: the LAST '.' or ',' is a decimal point only when exactly 1-2 digits
    follow it; every other separator is a thousands separator.
    Real photos often lose the decimal point: '28 775 29' or '47,958 82'. Sri Lankan payslip and
    bank amounts always print two decimals, so a final group of exactly 2 digits after a SPACE
    is read as the cents ('28 775 29' -> 28775.29) and the value is marked as corrected."""
    raw = text
    t0 = CURRENCY.sub("", text).strip().strip("()-+").strip()
    space_cents = False
    m = re.fullmatch(r"(.*[0-9OoIlSsBDQZzg])\s+([0-9OoIlSsBDQZzg]{2})", t0)
    if m and not re.search(r"[.,]\d{1,2}$", m.group(1)):
        t0 = f"{m.group(1)}.{m.group(2)}"
        space_cents = True
    t = t0.replace(" ", "")
    if not t or t.endswith("%"):
        return None
    fixed = "".join(CONFUSIONS.get(c, c) for c in t)
    corrected = fixed != t or space_cents
    if not re.fullmatch(r"[0-9.,]+", fixed) or not any(c.isdigit() for c in fixed):
        return None
    last = max(fixed.rfind("."), fixed.rfind(","))
    if last != -1 and 1 <= len(fixed) - last - 1 <= 2:
        whole, frac = fixed[:last], fixed[last + 1:]
        if fixed[last] == "," and ("." in whole or "," in whole):
            corrected = True        # '30,000,00': comma used as the decimal point
    else:
        whole, frac = fixed, ""
    whole = whole.replace(".", "").replace(",", "")
    if not whole and not frac:
        return None
    value = float(f"{whole or '0'}.{frac or '0'}")
    return ParsedAmount(value=round(value, 2), raw=raw, cleaned=fixed, corrected=corrected)
