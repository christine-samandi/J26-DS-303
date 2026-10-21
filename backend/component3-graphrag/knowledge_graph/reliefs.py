"""General relief engine: every relief in Fifth Schedule paragraph 2, rebuilt from the Acts.

    python -m knowledge_graph.versions     # prints these timelines with the others

Personal relief (2(a)) has its own reader in knowledge_graph.versions. The other reliefs are
handled here by ONE general method instead of code per relief:

1. Consolidate the text. Start from the 2017 wording of paragraph 2, split it into its
   subparagraphs (b), (c), ... and apply every amendment in the order the Acts were
   certified. Three kinds of amendment occur in the corpus:
     - "by the repeal of subparagraph (x) ... and the substitution therefor, of ... “...”"
     - "in subparagraph (x) ..., by the substitution for the words “old”, of the words “new”"
     - "by the addition immediately after subparagraph (x) ..., of the following new
       subparagraphs: “(f) ...; (g) ...”"
   The result is every wording each subparagraph has had, with the Act that gave it.
2. Read the rule from the latest wording: its value (a fixed amount, an amount "up to" a
   limit = a cap, or "N percent of" something = a rate) and the period each value covers
   ("on or after", "prior to", "up to December 31, 2019", "for the first nine months of the
   year of assessment commencing on ..." which is annualised like personal relief).
3. A wording with no date of its own starts when the amending section came into operation:
   the Act's Table 'A' / Table 'B' (section 1), else the date the Act was certified.
4. Build the timeline with the same code as the other rules. A version that ends without a
   successor is `outdated` (the relief was withdrawn), not `replaced`.

What the engine does NOT know is written by hand in RELIEFS below: the rule's name, what the
relief is limited by, who may claim it and the words people use for it. The values and dates
all come from the Acts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from entity_recognition.chunking import Chunk
from entity_recognition.extract import normalise_date, words_to_number

PRINCIPAL = "DOC-IRA-2017"
SCHEDULE = "FIFTH SCHEDULE"
PARAGRAPH = "2"


@dataclass(frozen=True)
class ReliefSpec:
    """Hand-written facts about one relief; values and dates come from the Acts."""

    letter: str  # subparagraph of Fifth Schedule para 2
    rule_key: str
    title: str
    id_prefix: str
    limited_to: str  # what the relief cannot exceed / is a share of
    conditions: str  # who may claim it, in plain words
    taxpayer_categories: tuple[str, ...]
    income_types: tuple[str, ...]
    keywords: tuple[str, ...]  # words that name this relief in a question


ALL_INCOME = ("employment", "business", "investment", "other")

RELIEFS = {
    s.rule_key: s
    for s in (
        ReliefSpec(
            "b",
            "employment_relief",
            "Relief on employment income",
            "RULE-EMPLOYMENT_RELIEF",
            "the individual's income from employment for the year",
            "an individual with income from employment",
            ("resident_individual", "non_resident_individual"),
            ("employment",),
            ("employment relief",),
        ),
        ReliefSpec(
            "c",
            "rental_relief",
            "Rent relief (repair, maintenance and depreciation)",
            "RULE-RENTAL_RELIEF",
            "the total rental income from an investment asset for the year",
            "an individual with rental income from an investment asset, who does not also claim "
            "the actual repair, maintenance and depreciation costs",
            ("resident_individual", "non_resident_individual"),
            ("investment",),
            (
                "rent relief",
                "rental relief",
                "rental income",
                "rent income",
                "rent",
                "rental",
                "landlord",
                "house rent",
            ),
        ),
        ReliefSpec(
            "d",
            "senior_citizen_interest_relief",
            "Senior citizen relief on interest income",
            "RULE-SENIOR_CITIZEN_RELIEF",
            "the individual's interest income from a financial institution for the year",
            "an individual who is a senior citizen in the year, with interest income from a "
            "financial institution",
            ("senior_citizen",),
            ("investment",),
            (
                "senior citizen",
                "senior citizens",
                "senior",
                "elderly",
                "fixed deposit",
                "fd interest",
                "deposit interest",
            ),
        ),
        ReliefSpec(
            "e",
            "foreign_currency_service_relief",
            "Relief on foreign-currency income from services",
            "RULE-FOREIGN_SERVICE_RELIEF",
            "such foreign-currency income for the year",
            "a resident individual (or partner of a partnership) earning foreign currency in Sri "
            "Lanka from services used outside Sri Lanka",
            ("resident_individual",),
            ("employment", "business", "other"),
            (
                "foreign currency",
                "foreign exchange",
                "export of services",
                "exported services",
                "freelance",
                "freelancer",
                "foreign clients",
                "overseas clients",
                "paid in dollars",
                "paid in us dollars",
                "paid in pounds",
                "paid in euros",
                "receive dollars",
                "receive pounds",
                "receive euros",
            ),
        ),
        ReliefSpec(
            "f",
            "qualifying_expenditure_relief",
            "Expenditure relief (health, education, housing loan interest, pension, shares)",
            "RULE-EXPENDITURE_RELIEF",
            "the total of the qualifying expenditure: health expenditure and medical insurance; "
            "local education for the individual or their children; housing loan interest; "
            "contributions to a local pension scheme; listed shares, treasury bonds or bills",
            "a resident individual",
            ("resident_individual",),
            ALL_INCOME,
            (
                "expenditure relief",
                "qualifying expenditure",
                "medical",
                "health expenditure",
                "health insurance",
                "medical insurance",
                "medical expenses",
                "education expenses",
                "educational expenditure",
                "school fees",
                "tuition",
                "housing loan",
                "home loan",
                "pension scheme",
                "treasury bond",
                "treasury bill",
                "shares",
            ),
        ),
        ReliefSpec(
            "g",
            "solar_panel_relief",
            "Solar panel relief",
            "RULE-SOLAR_RELIEF",
            "the total cost of the solar panels, or the amounts paid to a bank on a loan taken to "
            "buy them",
            "a resident individual who has fixed solar panels on their premises, connected to "
            "the national grid",
            ("resident_individual",),
            ALL_INCOME,
            (
                "solar",
                "solar panel",
                "solar panels",
                "solar power",
                "solar energy",
                "rooftop solar",
            ),
        ),
    )
}
BY_LETTER = {s.letter: s for s in RELIEFS.values()}


# --- 1. consolidating the text ------------------------------------------------------------


@dataclass
class Wording:
    """One wording a subparagraph has had, and where it came from."""

    letter: str
    text: str
    document_id: str
    chunk_id: str
    how: str  # "enacted" | "substituted" | "words substituted" | "added"
    verbatim: str  # text exactly as printed in the chunk (the quoted words for an amendment)


def split_subparagraphs(text: str, first: str = "a") -> dict[str, str]:
    """'(a) ...; (b) ...' -> {'a': '...', 'b': '...'}, looking only for the next letter in
    order, so roman numerals like '(i)' or '(v)' inside a subparagraph are not mistaken."""
    out: dict[str, str] = {}
    letter = first
    m = re.search(rf"\({letter}\)\s", text)
    if not m:
        return out
    pos = m.end()
    while True:
        nxt_letter = chr(ord(letter) + 1)
        nxt = re.search(rf"\s\({nxt_letter}\)\s", text[pos:])
        if not nxt:
            out[letter] = text[pos:].strip()
            return out
        out[letter] = text[pos : pos + nxt.start()].strip()  # keeps its own ";" or "."
        pos, letter = pos + nxt.end(), nxt_letter


_QUOTE = re.compile(r"“(.*?)”", re.S)
_ITEM = re.compile(r"-S(\d+)-(\d+)")


def _groups(chunks: list[Chunk], order: dict[str, int]) -> list[tuple[str, list[Chunk]]]:
    """Amendment text about the Fifth Schedule, one group per numbered item of a section
    (continuation chunks -P1, -P2 joined), in the order the Acts were certified."""
    groups: dict[tuple[str, str, int], list[Chunk]] = {}
    for c in chunks:
        if c.document_id == PRINCIPAL or "fifth schedule" not in (c.marginal_note or "").lower():
            continue
        m = _ITEM.search(c.chunk_id)
        item = int(m[2]) if m else 0
        groups.setdefault((c.document_id, c.section or "", item), []).append(c)
    keys = sorted(groups, key=lambda k: (order.get(k[0], 99), int(k[1] or 0), k[2]))
    return [(k[0], groups[k]) for k in keys]


def _op_spans(text: str) -> list[tuple[int, int]]:
    """Where each lettered operation '(a) by the ...', '(b) in subparagraph ...' starts and
    ends, ignoring anything inside quotation marks (the new text itself has letters too)."""
    masked = _QUOTE.sub(lambda m: "“" + "x" * len(m[1]) + "”", text)
    starts = [m.start() for m in re.finditer(r"\([a-z]\)\s+(?=by\s+the|in\s+subparagraph)", masked)]
    return [(s, starts[i + 1] if i + 1 < len(starts) else len(text)) for i, s in enumerate(starts)]


def _chunk_at(parts: list[tuple[int, Chunk]], pos: int) -> Chunk:
    return [c for start, c in parts if start <= pos][-1]


def consolidate(chunks: list[Chunk], order: dict[str, int]) -> dict[str, list[Wording]]:
    """letter -> every wording of Fifth Schedule para 2(letter), oldest first."""
    history: dict[str, list[Wording]] = {}
    current: dict[str, str] = {}
    base = [c for c in chunks if c.document_id == PRINCIPAL and c.schedule == SCHEDULE
            and c.paragraph == PARAGRAPH]  # fmt: skip
    for c in base:
        for letter, text in split_subparagraphs(c.text).items():
            current[letter] = text
            history.setdefault(letter, []).append(
                Wording(letter, text, c.document_id, c.chunk_id, "enacted", text)
            )

    for doc, group in _groups(chunks, order):
        parts, text = [], ""
        for c in group:
            parts.append((len(text), c))
            text += c.text + " "
        if not re.search(r"in\s+paragraph\s+2\s+of\s+that\s+Schedule", text):
            continue  # about paragraph 1, or item by item inside subparagraph (a)
        for start, end in _op_spans(text):
            op = text[start:end]
            chunk = _chunk_at(parts, start)
            quotes = _QUOTE.findall(op)
            if m := re.search(r"repeal\s+of\s+subparagraph\s+\((\w)\)", op):
                if quotes:
                    new = re.sub(rf"^\s*\({m[1]}\)\s*", "", quotes[0]).strip()
                    _record(history, current, m[1], new, doc, chunk, "substituted", quotes[0])
            elif m := re.search(r"in\s+subparagraph\s+\((\w)\)\s+of\s+that\s+paragraph,\s+by\s+"
                                r"the\s+substitution", op):  # fmt: skip
                if len(quotes) >= 2 and quotes[0] in current.get(m[1], ""):
                    new = current[m[1]].replace(quotes[0], quotes[1], 1)
                    _record(history, current, m[1], new, doc, chunk, "words substituted",
                            quotes[1])  # fmt: skip
            elif m := re.search(r"addition\s+immediately\s+after\s+subparagraph\s+\((\w)\)", op):
                if quotes:
                    first = chr(ord(m[1]) + 1)
                    for letter, new in split_subparagraphs(quotes[0], first).items():
                        added_in = _chunk_at(parts, start + op.find(f"({letter})"))
                        _record(history, current, letter, new, doc, added_in, "added",
                                f"({letter}) {new}")  # fmt: skip
    return history


def _record(history, current, letter, text, doc, chunk, how, verbatim) -> None:
    current[letter] = text
    history.setdefault(letter, []).append(Wording(letter, text, doc, chunk.chunk_id, how,
                                                  verbatim.strip()))  # fmt: skip


# --- 2./3. reading the rule ---------------------------------------------------------------


@dataclass
class Reading:
    kind: str  # amount | cap | rate
    value: int | float
    start: date | None
    end: date | None  # first day NOT covered
    matched: str


_RS = re.compile(r"Rs\.?\s*(\d[\d,]*)")
_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*per\s?cent", re.I)
_DATE = r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
_ON_OR_AFTER = re.compile(rf"on\s+or\s+after\s+{_DATE}")
_PRIOR_TO = re.compile(rf"prior\s+to\s+{_DATE}")
_UP_TO_DATE = re.compile(rf"up\s?to\s+{_DATE}")
_PART_YEAR = re.compile(
    rf"for\s+the\s+first\s+(\w+)\s+months\s+of\s+the\s+year\s+of\s+assessment\s+commencing\s+"
    rf"on\s+{_DATE}"
)


def _day(text: str) -> date:
    return date.fromisoformat(normalise_date(text))


def _add_months(day: date, months: int) -> date:
    y, m = divmod(day.month - 1 + months, 12)
    return day.replace(year=day.year + y, month=m + 1)


def _period(text: str) -> tuple[date | None, date | None]:
    start = end = None
    if m := _ON_OR_AFTER.search(text):
        start = _day(m[1])
    if m := _PRIOR_TO.search(text):
        end = _day(m[1])
    elif m := _UP_TO_DATE.search(text):
        end = date.fromordinal(_day(m[1]).toordinal() + 1)
    return start, end


def _clause(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def read_wording(text: str) -> list[Reading]:
    """The values a wording states and the period of each."""
    if m := _RATE.search(text):
        start, end = _period(text)
        return [Reading("rate", float(m[1]), start, end, _clause(text))]
    kind = "cap" if re.search(r"up\s?to\s+(?:the|a|such)\b", text) else "amount"
    amounts = list(_RS.finditer(text))
    out = []
    for i, m in enumerate(amounts):
        segment = text[m.start() : amounts[i + 1].start() if i + 1 < len(amounts) else len(text)]
        value = int(m[1].replace(",", ""))
        if part := _PART_YEAR.search(segment):  # "Rs. 900,000 ... for the first nine months"
            months = int(part[1]) if part[1].isdigit() else words_to_number(part[1])
            begins = _day(part[2])
            out.append(Reading(kind, round(value * 12 / months), begins,
                               _add_months(begins, months), _clause(text)))  # fmt: skip
            continue
        start, end = _period(segment)
        out.append(Reading(kind, value, start, end, _clause(text)))
    return out


def operation_date(chunks: list[Chunk], document_id: str, letter: str) -> date | None:
    """When an amending Act's change to Fifth Schedule para 2(letter) came into operation:
    Table 'A' of the Act (deemed date in section 1), else None (the certified date applies)."""
    texts = [c.text for c in chunks if c.document_id == document_id]
    whole = " ".join(texts)
    target = rf"subparagraphs?\s+(?:\([a-z]\)(?:,\s*|\s+and\s+))*\({letter}\)[^|]*?of\s+paragraph\s+2\s+of\s+the\s+Fifth\s+Schedule"  # noqa: E501
    a = whole.find("Table ‘A’ (section 1)")
    b = whole.find("Table ‘B’", a + 1) if a >= 0 else -1
    if a >= 0 and re.search(target, whole[a : b if b > 0 else len(whole)]):
        deemed = (
            r"Table\s+‘A’\s+set\s+out\s+in\s+this\s+Amendment\s+Act\s+shall\s+be\s+deemed"
            rf"\s+to\s+have\s+come\s+into\s+operation\s+on\s+{_DATE}"
        )
        m = re.search(deemed, whole)
        if m:
            return _day(m[1])
    return None


@dataclass
class ReliefStatements:
    """What the engine read for one relief, ready for the shared timeline code."""

    spec: ReliefSpec
    wordings: list[Wording] = field(default_factory=list)
    readings: list[tuple[Reading, Wording]] = field(default_factory=list)  # from every wording


def relief_readings(chunks: list[Chunk], order: dict[str, int], certified: dict[str, date]
                    ) -> dict[str, ReliefStatements]:  # fmt: skip
    history = consolidate(chunks, order)
    out = {}
    for key, spec in RELIEFS.items():
        rs = ReliefStatements(spec, history.get(spec.letter, []))
        for w in rs.wordings:
            for r in read_wording(w.text):
                # a NEW subparagraph without its own date starts when its section came into
                # operation; a rewritten one without a date still speaks from the start
                if r.start is None and w.how == "added":
                    r.start = operation_date(chunks, w.document_id, spec.letter) or certified.get(
                        w.document_id
                    )
                rs.readings.append((r, w))
        out[key] = rs
    return out
