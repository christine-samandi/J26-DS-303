"""Rule versioning: rebuild the history of a tax rule from the Acts that state and amend it.

    python -m knowledge_graph.versions     # print the timelines and score them (no Neo4j needed)

How it works, for each tracked rule (RULES):
1. Find the text that states the rule: the provision in the 2017 Act, and the words the
   amendment Acts insert into that provision (the quoted text in "... the following
   subparagraph: - “(a) (i) Rs. 500,000, ...”").
2. Read *statements* from that text with the Phase 2 extractor: a value (an AMOUNT, or a table
   of TAX_SLAB rows with their RATEs) and the period it applies to (DATE entities with their
   qualifier: "on or after" starts the period, "prior to" ends it).
   Mid-year changes are annualised: "Rs. 2,250,000 for first nine months and Rs. 300,000 for
   second three months of the year of assessment commencing on April 1, 2022" is
   Rs. 3,000,000 a year from 1 April 2022 and Rs. 1,200,000 a year from 1 January 2023.
3. Build the timeline: for every day, the statement made by the LATEST Act that covers that
   day wins (a later Act overrides an earlier one for the period it speaks about). Days with
   the same value are merged into one version; each version ends the day before the next
   one starts.

The result is compared with the hand-verified versions in ground_truth/ (objective:
versioning accuracy).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from entity_recognition.chunking import Chunk, load_chunks
from entity_recognition.extract import extract_entities, normalise_date, words_to_number
from knowledge_graph.reliefs import RELIEFS, ReliefStatements, relief_readings


@dataclass(frozen=True)
class RuleSpec:
    rule_key: str
    rule_type: str  # contract RuleType
    title: str
    id_prefix: str  # rule_id = f"{id_prefix}-V{n}"
    provision: str  # where the rule lives in the 2017 Act
    taxpayer_categories: tuple[str, ...]
    income_types: tuple[str, ...] = ("employment", "business", "investment", "other")
    limited_to: str = ""  # reliefs: what the relief cannot exceed / is a share of
    conditions: str = ""  # reliefs: who may claim it


RULES = {
    "personal_relief": RuleSpec(
        "personal_relief",
        "relief",
        "Personal relief for resident individuals",
        "RULE-PERSONAL_RELIEF",
        "Fifth Schedule, paragraph 2(a)",
        ("resident_individual",),  # s.52(2): "individual who is resident in Sri Lanka"
    ),
    "individual_income_tax_slabs": RuleSpec(
        "individual_income_tax_slabs",
        "tax_slab",
        "Income tax rates for individuals (progressive bands)",
        "RULE-INDIVIDUAL_SLABS",
        "First Schedule, paragraph 1",
        ("resident_individual", "non_resident_individual"),  # "resident or non-resident"
    ),
}
# every other relief in Fifth Schedule para 2, read by the general relief engine
RULES |= {
    key: RuleSpec(key, "relief", r.title, r.id_prefix, f"Fifth Schedule, paragraph 2({r.letter})",
                  r.taxpayer_categories, r.income_types, r.limited_to, r.conditions)
    for key, r in RELIEFS.items()
}  # fmt: skip

Band = tuple[int, int | None, float]  # (lower, upper or None, rate %)


@dataclass
class Statement:
    """One thing an Act says about a rule: this value, for this period."""

    rule_key: str
    value: int | tuple[Band, ...]
    start: date | None  # None: from the start of the Act
    end: date | None  # first day NOT covered; None: open-ended
    document_id: str
    chunk_id: str
    matched_text: str  # verbatim from the chunk


@dataclass
class Version:
    rule_id: str
    rule_key: str
    version: int
    value: int | list[list]
    valid_from: date
    valid_to: date | None  # last day in force (inclusive)
    status: str  # active | replaced
    source_document_id: str  # the Act that first set this value
    source_chunk_id: str
    matched_text: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    supersedes: str | None = None
    superseded_by: str | None = None
    value_kind: str = "amount"  # amount | cap | rate | bands

    def to_dict(self) -> dict:
        d = asdict(self)
        d["valid_from"] = self.valid_from.isoformat()
        d["valid_to"] = self.valid_to.isoformat() if self.valid_to else None
        return d


# --- finding the text that states each rule ----------------------------------------------

_QUOTED = re.compile(r"(?:“|‘‘|\")(.+?)(?:”|’’|\")", re.S)


def _quoted(text: str) -> list[str]:
    return [m[1] for m in _QUOTED.finditer(text)]


def _leads(chunks: list[Chunk]) -> dict[tuple[str, str], str]:
    """Lead-in text of each amendment section ("... is hereby amended in subparagraph (a) of
    paragraph 2 as follows"), which says what the numbered items below it change."""
    return {
        (c.document_id, c.section): c.text
        for c in chunks
        if c.section and c.chunk_id.endswith(f"S{int(c.section):03d}-0")
        if c.section.isdigit()
    }


def relief_texts(chunks: list[Chunk]) -> list[tuple[Chunk, str]]:
    """Text of Fifth Schedule para 2(a): the 2017 wording and every amendment's new wording."""
    out = []
    leads = _leads(chunks)
    for c in chunks:
        if c.document_id == "DOC-IRA-2017":
            if c.schedule == "FIFTH SCHEDULE" and c.paragraph == "2":
                m = re.search(r"\(a\)(.*?);\s*\(b\)", c.text, re.S)
                if m:
                    out.append((c, m[1]))
            continue
        if "fifth schedule" not in (c.marginal_note or "").lower():
            continue
        lead = leads.get((c.document_id, c.section), "")
        if re.search(r"paragraph\s+2\s+of\s+that\s+Schedule", c.text):
            out += [(c, q) for q in _quoted(c.text) if q.lstrip().startswith("(a)")]
        elif re.search(r"subparagraph\s+\(a\)\s+of\s+paragraph\s+2\b", lead):
            out += [(c, q) for q in _quoted(c.text)]
    return out


_INDIVIDUAL_TABLE = re.compile(
    r"taxable\s+income\s+of\s+a\s+resident\s+or\s+non-?\s?resident\s+individual"
    r"(?P<cond>.{0,160}?)shall\s+be\s+taxed\s+at\s+the\s+following\s+rates(?P<body>.*?)(?:”|’’|$)",
    re.S | re.I,
)


def slab_texts(chunks: list[Chunk]) -> list[tuple[Chunk, str, str]]:
    """(chunk, condition, table text) for every rate table of First Schedule para 1."""
    out = []
    for c in chunks:
        in_2017 = c.document_id == "DOC-IRA-2017" and c.schedule == "FIRST SCHEDULE"
        amends = "first schedule" in (c.marginal_note or "").lower()
        if in_2017 or amends:
            out += [(c, m["cond"], m["body"]) for m in _INDIVIDUAL_TABLE.finditer(c.text)]
    return out


# --- reading statements -------------------------------------------------------------------


def _add_months(day: date, months: int) -> date:
    y, m = divmod(day.month - 1 + months, 12)
    return day.replace(year=day.year + y, month=m + 1)


def _period(text: str) -> tuple[date | None, date | None]:
    """Start and end from the DATE entities in a condition ("on or after X, but prior to Y")."""
    start = end = None
    for e in extract_entities(text):
        if e.label != "DATE" or not e.value:
            continue
        if e.qualifier == "on_or_after":
            start = date.fromisoformat(e.value)
        elif e.qualifier == "prior_to":
            end = date.fromisoformat(e.value)
    return start, end


_RS = r"Rs\.?\s*(\d[\d,]*)"
_SPLIT_YEAR = re.compile(
    rf"{_RS},?\s+for\s+(?:the\s+)?first\s+(\w+)\s+months\s+and\s+{_RS},?\s+for\s+(?:the\s+)?"
    r"second\s+(\w+)\s+months\s+of\s+the\s+year\s+of\s+assessment\s+commencing\s+on\s+"
    r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
)
_EACH_YEAR = re.compile(
    rf"{_RS},?\s+for\s+each\s+year\s+of\s+assessment(?P<cond>(?:(?!,?\s*except)[^;”(])*)"
)


def _months(word: str) -> int:
    return int(word) if word.isdigit() else words_to_number(word)


def relief_statements(chunks: list[Chunk]) -> list[Statement]:
    out = []
    for c, text in relief_texts(chunks):
        for m in _SPLIT_YEAR.finditer(text):
            first, second = _months(m[2]), _months(m[4])
            begins = date.fromisoformat(normalise_date(m[5]))
            middle = _add_months(begins, first)
            for amount, months, start, end in (
                (m[1], first, begins, middle),
                (m[3], second, middle, _add_months(middle, second)),
            ):
                annual = round(int(amount.replace(",", "")) * 12 / months)
                out.append(
                    Statement("personal_relief", annual, start, end, c.document_id, c.chunk_id,
                              m[0])
                )  # fmt: skip
        for m in _EACH_YEAR.finditer(text):
            start, end = _period(m["cond"])
            amount = int(m[1].replace(",", ""))
            out.append(
                Statement("personal_relief", amount, start, end, c.document_id, c.chunk_id,
                          m[0].rstrip(" ,"))
            )  # fmt: skip
    return out


_SUB_PERIOD = re.compile(
    r"for\s+the\s+(first|second)\s+(\w+)\s+months?\s+period\s+of\s+the\s+year\s+of\s+"
    r"assessment\s+commencing\s+from\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
)


def _bands(table: str, scale: float = 1.0) -> tuple[Band, ...]:
    """Bands from a rate table: each TAX_SLAB row with the first RATE after it."""
    entities = extract_entities(table)
    slabs = [e for e in entities if e.label == "TAX_SLAB"]
    rates = [e for e in entities if e.label == "RATE"]
    bands = []
    for i, slab in enumerate(slabs):
        stop = slabs[i + 1].start if i + 1 < len(slabs) else len(table)
        rate = next((r.value for r in rates if slab.end <= r.start < stop), None)
        lower, upper = slab.value.split("-")
        bands.append(
            (round(int(lower) * scale), round(int(upper) * scale) if upper else None, rate)
        )
    return tuple(bands)


def slab_statements(chunks: list[Chunk]) -> list[Statement]:
    out = []
    for c, cond, body in slab_texts(chunks):
        parts = list(_SUB_PERIOD.finditer(body))
        if not parts:  # one table for the whole period in the condition
            start, end = _period(cond)
            bands = _bands(body)
            if bands:
                out.append(
                    Statement("individual_income_tax_slabs", bands, start, end, c.document_id,
                              c.chunk_id, _verbatim(c.text, cond, body))
                )  # fmt: skip
            continue
        for i, m in enumerate(parts):  # a year split into periods: annualise each table
            which, months = m[1], _months(m[2])
            begins = date.fromisoformat(normalise_date(m[3]))
            start = begins if which == "first" else _add_months(begins, 12 - months)
            table = body[m.end() : parts[i + 1].start() if i + 1 < len(parts) else len(body)]
            bands = _bands(table, scale=12 / months)
            if bands:
                out.append(
                    Statement("individual_income_tax_slabs", bands, start,
                              _add_months(start, months), c.document_id, c.chunk_id,
                              " ".join(m[0].split()))
                )  # fmt: skip
    return out


def _verbatim(text: str, cond: str, body: str) -> str:
    """The sentence that introduces the table, as printed in the chunk."""
    m = re.search(
        r"[^.;:“‘\-]*?taxable\s+income\s+of\s+a\s+resident.*?following\s+rates", text, re.S
    )
    return " ".join(m[0].split()).strip() if m else " ".join((cond + body).split())[:300]


# --- timeline -----------------------------------------------------------------------------


def document_order() -> dict[str, int]:
    """Later Acts override earlier ones: rank documents by certified date."""
    from data_collection.sources import load_registry

    docs = sorted(load_registry().documents, key=lambda d: d.certified_date)
    return {d.document_id: i for i, d in enumerate(docs)}


def corpus_start() -> date:
    from data_collection.sources import load_registry

    return load_registry().get("DOC-IRA-2017").operative_from


def build_timeline(
    statements: list[Statement], spec: RuleSpec, order: dict[str, int], start_of_law: date
) -> list[Version]:
    def span(s: Statement) -> tuple[date, date | None]:
        return max(s.start or start_of_law, start_of_law), s.end

    cuts = sorted({span(s)[0] for s in statements} | {s.end for s in statements if s.end})
    pieces: list[tuple[date, date | None, Statement]] = []
    for i, day in enumerate(cuts):
        covering = [
            (order[s.document_id], n, s)
            for n, s in enumerate(statements)
            if span(s)[0] <= day and (s.end is None or day < s.end)
        ]
        if not covering:
            continue
        winner = max(covering, key=lambda t: (t[0], t[1]))[2]
        pieces.append((day, cuts[i + 1] if i + 1 < len(cuts) else None, winner))

    merged: list[list] = []  # [start, end, value, [statements]]
    for start, end, st in pieces:
        if merged and merged[-1][2] == st.value and merged[-1][1] == start:
            merged[-1][1] = end
            merged[-1][3].append(st)
        else:
            merged.append([start, end, st.value, [st]])

    versions = []
    for n, (start, end, value, _) in enumerate(merged, start=1):
        same = [
            s
            for s in statements
            if s.value == value and span(s)[0] <= start and (s.end is None or start < s.end)
        ]
        first = min(same, key=lambda s: order[s.document_id])  # the Act that set this value
        evidence = list(
            dict.fromkeys([first.chunk_id] + [s.chunk_id for s in statements if s.value == value])
        )
        versions.append(
            Version(
                rule_id=f"{spec.id_prefix}-V{n}",
                rule_key=spec.rule_key,
                version=n,
                value=value if isinstance(value, (int, float)) else [list(b) for b in value],
                valid_from=start,
                valid_to=end - timedelta(days=1) if end else None,
                status="active" if end is None else "replaced",
                source_document_id=first.document_id,
                source_chunk_id=first.chunk_id,
                matched_text=first.matched_text,
                evidence_chunk_ids=evidence,
            )
        )
    for prev, nxt in zip(versions, versions[1:], strict=False):
        prev.superseded_by, nxt.supersedes = nxt.rule_id, prev.rule_id
    return versions


def relief_versions(rs: ReliefStatements, order: dict[str, int], start_of_law: date
                    ) -> list[Version]:  # fmt: skip
    """Versions of one relief from the general engine. The timeline is read from the LATEST
    wording (it is the consolidated law: an amendment that ends a relief only says so in the
    new wording); each version is then credited to the earliest Act that stated its value."""
    if not rs.wordings:
        return []
    latest = rs.wordings[-1]
    readings = [(r, w) for r, w in rs.readings if w is latest]
    statements = [Statement(rs.spec.rule_key, r.value, r.start, r.end, w.document_id, w.chunk_id,
                            r.matched) for r, w in readings]  # fmt: skip
    versions = build_timeline(statements, RULES[rs.spec.rule_key], order, start_of_law)
    kind = readings[0][0].kind if readings else "amount"
    wording_chunks = list(dict.fromkeys(w.chunk_id for w in rs.wordings))
    for v in versions:
        v.value_kind = kind
        first = next(
            (w for r, w in rs.readings
             if r.value == v.value and max(r.start or start_of_law, start_of_law) <= v.valid_from
             and (r.end is None or v.valid_from < r.end)),
            latest,
        )  # fmt: skip
        v.source_document_id, v.source_chunk_id = first.document_id, first.chunk_id
        v.matched_text = " ".join(first.verbatim.split())
        v.evidence_chunk_ids = list(dict.fromkeys([first.chunk_id, *wording_chunks]))
    if versions and versions[-1].valid_to is not None:
        versions[-1].status = "outdated"  # ended with no successor: the relief was withdrawn
    return versions


def build_versions(chunks: list[Chunk]) -> dict[str, list[Version]]:
    order, start = document_order(), corpus_start()
    readers = {"personal_relief": relief_statements, "individual_income_tax_slabs": slab_statements}
    out = {key: build_timeline(readers[key](chunks), RULES[key], order, start) for key in readers}
    for v in out["individual_income_tax_slabs"]:
        v.value_kind = "bands"
    certified = _certified_dates()
    for key, rs in relief_readings(chunks, order, certified).items():
        out[key] = relief_versions(rs, order, start)
    return out


def _certified_dates() -> dict[str, date]:
    from data_collection.sources import load_registry

    return {d.document_id: d.certified_date for d in load_registry().documents}


# --- scoring against the hand-verified versions -------------------------------------------


def _gold(ground_truth_dir: Path) -> dict[str, list[dict]]:
    def rows(name):
        with (ground_truth_dir / name).open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    relief = [
        {"value": int(r["amount_lkr"]), "valid_from": r["valid_from"],
         "valid_to": r["valid_to"] or None, "source": r["source_document_id"]}
        for r in rows("personal_relief_versions.csv")
    ]  # fmt: skip
    slabs: dict[str, dict] = {}
    for r in rows("individual_tax_bands.csv"):
        v = slabs.setdefault(
            r["version"],
            {"value": [], "valid_from": r["valid_from"], "valid_to": r["valid_to"] or None,
             "source": r["source_document_id"]},
        )  # fmt: skip
        upper = int(r["upper_lkr"]) if r["upper_lkr"] else None
        v["value"].append([int(r["lower_lkr"]), upper, float(r["rate_percent"])])
    gold = {"personal_relief": relief, "individual_income_tax_slabs": list(slabs.values())}
    extra = (
        rows("relief_versions.csv") if (ground_truth_dir / "relief_versions.csv").exists() else []
    )
    for r in extra:  # the other Fifth Schedule para 2 reliefs
        value = float(r["value"]) if r["value_kind"] == "rate" else int(r["value"])
        gold.setdefault(r["rule_key"], []).append(
            {"value": value, "valid_from": r["valid_from"], "valid_to": r["valid_to"] or None,
             "source": r["source_document_id"], "status": r["status"]}
        )  # fmt: skip
    return gold


def score(versions: dict[str, list[Version]], ground_truth_dir: Path) -> dict:
    """Per rule: how many hand-verified versions were rebuilt exactly (value, both dates and
    source Act all correct), and which fields were wrong."""
    report = {}
    for key, gold in _gold(ground_truth_dir).items():
        got = [v.to_dict() for v in versions.get(key, [])]
        checks = []
        for i, g in enumerate(gold):
            v = got[i] if i < len(got) else None
            wrong = (
                ["missing"]
                if v is None
                else [
                    f
                    for f, ok in (
                        ("value", v["value"] == g["value"]),
                        ("valid_from", v["valid_from"] == g["valid_from"]),
                        ("valid_to", v["valid_to"] == g["valid_to"]),
                        ("source", v["source_document_id"] == g["source"]),
                        ("status", v["status"] == g.get("status", v["status"])),
                    )
                    if not ok
                ]
            )
            checks.append({"version": i + 1, "exact": not wrong, "wrong": wrong})
        report[key] = {
            "gold_versions": len(gold),
            "built_versions": len(got),
            "exact": sum(c["exact"] for c in checks),
            "checks": checks,
        }
    total = sum(r["gold_versions"] for r in report.values())
    exact = sum(r["exact"] for r in report.values())
    report["ALL"] = {"gold_versions": total, "exact": exact, "accuracy": round(exact / total, 4)}
    return report


# --- CLI ----------------------------------------------------------------------------------


def _show(v: Version) -> str:
    to = v.valid_to.isoformat() if v.valid_to else "now       "
    if v.value_kind == "rate":
        what = f"{v.value:g}% (of the base)"
    elif v.value_kind == "cap":
        what = f"up to Rs. {v.value:,}"
    elif isinstance(v.value, int):
        what = f"Rs. {v.value:,}"
    else:
        what = ", ".join(f"{r:g}%" for _, _, r in v.value) + f"  ({len(v.value)} bands)"
    return (f"  {v.rule_id:30} {v.valid_from} .. {to}  {what:40} {v.source_document_id:22} "
            f"{v.status}")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import COMPONENT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Rebuild rule versions and score them.")
    parser.add_argument("--out", type=Path, default=COMPONENT_DIR / "data" / "graph")
    args = parser.parse_args(argv)

    chunks = load_chunks(get_settings().processed_dir)
    if not chunks:
        print("No parsed Acts. Run: python -m data_collection.parse")
        return 1
    versions = build_versions(chunks)
    for key, vs in versions.items():
        print(f"{RULES[key].title}  [{RULES[key].provision}]")
        for v in vs:
            print(_show(v))
        print()
    result = score(versions, COMPONENT_DIR / "ground_truth")
    for key in RULES:
        r = result[key]
        bad = [f"V{c['version']}: {', '.join(c['wrong'])}" for c in r["checks"] if not c["exact"]]
        print(f"{key:32} {r['exact']}/{r['gold_versions']} versions exact"
              + (f"   wrong: {'; '.join(bad)}" if bad else ""))  # fmt: skip
    a = result["ALL"]
    print(f"{'versioning accuracy':32} {a['exact']}/{a['gold_versions']} = {a['accuracy']:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {k: [v.to_dict() for v in vs] for k, vs in versions.items()}
    (args.out / "rule_versions.json").write_text(json.dumps(payload, indent=2) + "\n")
    eval_dir = COMPONENT_DIR / "data" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "versioning_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVersions written to {args.out / 'rule_versions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
