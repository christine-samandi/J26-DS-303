"""Replace personal identifiers with placeholders such as [NIC_1], and restore them.

Redacted, most specific first:
  - EMAIL    e-mail addresses
  - EMP_NO   employee numbers like "EMP-0042"
  - EPF_NO   EPF numbers, only after a label ("EPF No: 12345", "EPF Number A/12345"), so
             "EPF employee contribution: Rs. 120,000.00" is left alone
  - NIC      Sri Lankan NIC: new (12 digits) or old (9 digits + V/X)
  - PHONE    Sri Lankan numbers: +94 / 0094 / 0 followed by 9 digits, spaces or "-" allowed
  - NAME     the names in `known_names`, case-insensitive

Never touched: text inside square brackets (citation tags and placeholders already made),
and digit runs that are part of a larger number ("1,800,000.00", "123456789012.50"), so the
amounts, step figures, citation tags and year of assessment the post-check relies on reach
the LLM unchanged. Known gap: a bare 12-digit figure with no commas or decimals reads as a
new NIC; the engine always writes figures with commas and two decimals.

The same value always gets the same placeholder within one `redact` call. The mapping is
returned to the caller only, to keep in memory for the request; never log it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_BRACKETED = re.compile(r"(\[[^\[\]]*\])")  # split() keeps the tags as their own parts
_ALONE = r"(?<![\w.,]){}(?![\w]|[.,]\d)"  # not part of a longer word or number
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("EMP_NO", re.compile(r"\bEMP-?\d+\b", re.I)),
    # Only the value (group "v") is replaced; the label stays so the text still reads.
    ("EPF_NO", re.compile(r"\bEPF\s*(?:No\.?|Number|#)\s*:?\s*(?P<v>[A-Z]{0,3}/?\d[\d/-]*)", re.I)),
    ("NIC", re.compile(_ALONE.format(r"(?:\d{12}|\d{9}[VX])"), re.I)),
    ("PHONE", re.compile(_ALONE.format(r"(?:\+94|0094|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}"))),
)


@dataclass(frozen=True)
class Redaction:
    text: str
    mapping: dict[str, str] = field(repr=False)  # placeholder -> original; never log it

    @property
    def count(self) -> int:
        """Distinct values redacted."""
        return len(self.mapping)


def redact(text: str, known_names: Iterable[str] = ()) -> Redaction:
    """`text` with personal identifiers replaced by placeholders, and the map to undo it."""
    originals: dict[tuple[str, str], str] = {}  # (kind, value) -> placeholder
    counts: dict[str, int] = {}

    def placeholder(kind: str, value: str) -> str:
        if (kind, value) not in originals:
            counts[kind] = counts.get(kind, 0) + 1
            originals[kind, value] = f"[{kind}_{counts[kind]}]"
        return originals[kind, value]

    for kind, pattern in (*_PATTERNS, *_name_patterns(known_names)):
        text = _sub_outside_brackets(text, pattern, kind, placeholder)
    return Redaction(text, {p: value for (_, value), p in originals.items()})


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put the originals back. Placeholders not in `mapping` are left as they are."""
    if not mapping:
        return text
    found = re.compile("|".join(re.escape(p) for p in mapping))
    return found.sub(lambda m: mapping[m.group()], text)


def _name_patterns(names: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Longest first, so "Nimal Perera" is one name rather than "Nimal" plus a surname."""
    cleaned = sorted({" ".join(n.split()) for n in names if n and n.strip()}, key=len, reverse=True)
    return [
        ("NAME", re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, n.split())) + r"(?!\w)", re.I))
        for n in cleaned
    ]


def _sub_outside_brackets(text: str, pattern: re.Pattern[str], kind: str, placeholder) -> str:
    def swap(m: re.Match[str]) -> str:
        if "v" not in pattern.groupindex:
            return placeholder(kind, m.group())
        start, end = (i - m.start() for i in m.span("v"))
        whole = m.group()
        return whole[:start] + placeholder(kind, m.group("v")) + whole[end:]

    parts = _BRACKETED.split(text)
    return "".join(p if i % 2 else pattern.sub(swap, p) for i, p in enumerate(parts))
