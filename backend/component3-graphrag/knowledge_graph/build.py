"""Build the C3 knowledge graph in Neo4j.

    docker compose up -d                      # start Neo4j (browser: http://localhost:7474)
    python -m knowledge_graph.build           # rebuild the whole graph (about 10 s)
    python -m knowledge_graph.build --dry-run # only count what would be written

Inputs: data/extracted (chunks + entities, Phase 2) and the rule versions
(knowledge_graph.versions). Everything is rebuilt from those files, so the graph can be
dropped and recreated at any time.

Graph model (relation names follow the retrieval contract's `Relation` list):

    (:Document)                         one Act
    (:Provision)-[:PART_OF]->(:Document)            a section or a Schedule paragraph
    (:Chunk)-[:PART_OF]->(:Provision)               retrievable text with its page numbers
    (:Chunk)-[:AMENDS]->(:Provision)                an amendment Act changing the 2017 Act
    (:Chunk)-[:REFERS_TO]->(:Provision)             "section 52", "paragraph 2 of the Fifth
                                                     Schedule to the principal enactment"
    (:Chunk)-[:MENTIONS]->(:Term)                   relief / deduction / income type / ...
    (:RuleVersion)-[:VERSION_OF]->(:Rule)           e.g. personal relief V1..V4
    (:RuleVersion)-[:DEFINED_IN]->(:Chunk)          the evidence for that version
    (:RuleVersion)-[:SUPERSEDES]->(:RuleVersion)    V2 replaced V1, ...
    (:RuleVersion)-[:HAS_BAND]->(:Band)             tax-table rows
    (:RuleVersion)-[:APPLIES_TO]->(:TaxpayerCategory)
    (:Document)-[:AMENDS]->(:Rule)                  which Acts changed the rule
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from knowledge_graph.versions import RULES, Version, build_versions

LEXICON_LABELS = {"RELIEF", "DEDUCTION", "EXEMPTION", "INCOME_TYPE", "TAXPAYER_CATEGORY"}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
PRINCIPAL = "DOC-IRA-2017"


def load_extracted(extracted_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(extracted_dir.glob("DOC-*.json")):
        chunks += json.loads(path.read_text(encoding="utf-8"))["chunks"]
    return chunks


def _seg(document_id: str, section: str | None = None, schedule: int | None = None,
         paragraph: str | None = None) -> str:  # fmt: skip
    stem = document_id.removeprefix("DOC-")
    if schedule:
        return f"SEG-{stem}-SCH{schedule}-P{int(paragraph):03d}"
    number = re.match(r"(\d+)([A-Z]*)", section.upper())
    return f"SEG-{stem}-S{int(number[1]):03d}{number[2]}"


_SECTION = re.compile(r"\bsection\s+(\d+[A-Z]?)\b", re.I)
_SCHEDULE_PARA = re.compile(
    r"\bparagraph\s+(\d+)(?:\s*\([^)]*\))*\s+of\s+the\s+(" + "|".join(ORDINALS) + r")\s+Schedule",
    re.I,
)


def resolve_reference(ref: str, chunk: dict) -> str | None:
    """Segment ID a SECTION_REF points to, or None ("subsection (2)", "that paragraph").

    "of the principal enactment" (or "Inland Revenue Act, No. 24 of 2017") points into the
    2017 Act; otherwise a reference stays inside the chunk's own Act.
    """
    to_principal = re.search(r"principal\s+enactment|No\.\s*24\s+of\s+2017", ref, re.I)
    doc = PRINCIPAL if to_principal else chunk["document_id"]
    m = _SCHEDULE_PARA.search(ref)
    if m:
        return _seg(doc, schedule=ORDINALS.index(m[2].lower()) + 1, paragraph=m[1])
    m = _SECTION.search(ref)
    if m and not re.match(r"\s*(?:\(\w+\)\s*)*of\s+section", ref[m.end() :]):
        return _seg(doc, section=m[1])
    return None


_AMENDS_SECTION = re.compile(r"Amendment\s+of\s+section\s+(\d+[A-Z]?)", re.I)
_AMENDS_SCHEDULE = re.compile(r"Amendment\s+of\s+the\s+(" + "|".join(ORDINALS) + r")\s+Schedule",
                              re.I)  # fmt: skip


def amended_provision(chunk: dict, leads: dict[tuple[str, str], str]) -> str | None:
    """The 2017 provision an amendment chunk changes, from its marginal note
    ("Amendment of section 52 ...") and, for a Schedule, the paragraph its text names."""
    note = chunk.get("marginal_note") or ""
    m = _AMENDS_SECTION.search(note)
    if m:
        return _seg(PRINCIPAL, section=m[1])
    m = _AMENDS_SCHEDULE.search(note)
    if not m:
        return None
    lead = leads.get((chunk["document_id"], chunk["section"]), "")
    para = re.search(r"\bparagraph\s+(\d+)", chunk["text"]) or re.search(
        r"\bparagraph\s+(\d+)", lead
    )
    if not para:
        return None
    return _seg(PRINCIPAL, schedule=ORDINALS.index(m[1].lower()) + 1, paragraph=para[1])


def known_documents(documents: list, until: date | None = None) -> list:
    """The registry documents certified on or before `until` (all of them if None)."""
    return [d for d in documents if until is None or d.certified_date <= until]


def graph_rows(chunks: list[dict], versions: dict[str, list[Version]], documents: list) -> dict:
    """Everything the graph holds, as plain rows (no database needed: easy to test)."""
    known_segments = {c["segment_id"] for c in chunks}
    leads = {
        (c["document_id"], c["section"]): c["text"]
        for c in chunks
        if c.get("section") and c["chunk_id"].endswith("-0")
    }
    rows: dict[str, list[dict]] = {k: [] for k in (
        "documents", "provisions", "chunks", "amends", "refers_to", "mentions",
        "rules", "versions", "defined_in", "supersedes", "bands", "applies_to", "rule_amended_by",
    )}  # fmt: skip

    for d in documents:
        rows["documents"].append({
            "document_id": d.document_id, "title": d.title, "document_type": d.document_type,
            "act_number": d.act_number, "certified_date": d.certified_date.isoformat(),
            "url": d.url, "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })  # fmt: skip

    seen_provisions = set()
    for c in chunks:
        if c["segment_id"] not in seen_provisions:
            seen_provisions.add(c["segment_id"])
            rows["provisions"].append({
                "segment_id": c["segment_id"], "document_id": c["document_id"],
                "kind": c["kind"], "section": c["section"], "schedule": c["schedule"],
                "paragraph": c["paragraph"], "marginal_note": c["marginal_note"],
            })  # fmt: skip
        rows["chunks"].append({
            k: c[k] for k in ("chunk_id", "segment_id", "document_id", "section", "subsection",
                              "schedule", "paragraph", "marginal_note", "page_start", "page_end",
                              "text")
        })  # fmt: skip

        target = amended_provision(c, leads) if c["document_id"] != PRINCIPAL else None
        if target in known_segments:
            rows["amends"].append({"chunk_id": c["chunk_id"], "segment_id": target})

        refs = set()
        for e in c["entities"]:
            if e["label"] == "SECTION_REF":
                seg = resolve_reference(e["text"], c)
                if seg in known_segments and seg != c["segment_id"]:
                    refs.add((seg, e["text"]))
            elif e["label"] in LEXICON_LABELS:
                rows["mentions"].append(
                    {"chunk_id": c["chunk_id"], "label": e["label"], "value": e["value"]}
                )
        rows["refers_to"] += [
            {"chunk_id": c["chunk_id"], "segment_id": s, "text": t} for s, t in sorted(refs)
        ]

    for key, vs in versions.items():
        spec = RULES[key]
        rows["rules"].append({"rule_key": key, "rule_type": spec.rule_type, "title": spec.title,
                              "provision": spec.provision})  # fmt: skip
        for v in vs:
            d = v.to_dict()
            rows["versions"].append({
                "rule_id": v.rule_id, "rule_key": key, "version": v.version,
                "valid_from": d["valid_from"], "valid_to": d["valid_to"], "status": v.status,
                "amount": v.value if isinstance(v.value, int) else None,
                "source_document_id": v.source_document_id,
                "source_chunk_id": v.source_chunk_id, "matched_text": v.matched_text,
                "income_types": list(spec.income_types),
            })  # fmt: skip
            rows["defined_in"] += [
                {"rule_id": v.rule_id, "chunk_id": cid, "primary": cid == v.source_chunk_id}
                for cid in v.evidence_chunk_ids
            ]
            if v.supersedes:
                rows["supersedes"].append({"rule_id": v.rule_id, "previous": v.supersedes})
            if not isinstance(v.value, int):
                rows["bands"] += [
                    {"rule_id": v.rule_id, "band_id": f"{v.rule_id}-B{i}", "order": i,
                     "lower": lo, "upper": up, "rate_percent": rate}
                    for i, (lo, up, rate) in enumerate(v.value, start=1)
                ]  # fmt: skip
            rows["applies_to"] += [
                {"rule_id": v.rule_id, "category": cat} for cat in spec.taxpayer_categories
            ]
            if v.source_document_id != PRINCIPAL:
                rows["rule_amended_by"].append(
                    {"rule_key": key, "document_id": v.source_document_id, "rule_id": v.rule_id}
                )
    return rows


# --- Neo4j --------------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (n:Document) REQUIRE n.document_id IS UNIQUE",
    "CREATE CONSTRAINT seg_id IF NOT EXISTS FOR (n:Provision) REQUIRE n.segment_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT rule_key IF NOT EXISTS FOR (n:Rule) REQUIRE n.rule_key IS UNIQUE",
    "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (n:RuleVersion) REQUIRE n.rule_id IS UNIQUE",
    "CREATE CONSTRAINT band_id IF NOT EXISTS FOR (n:Band) REQUIRE n.band_id IS UNIQUE",
]

WRITES = [
    ("documents", "MERGE (d:Document {document_id: r.document_id}) SET d += r"),
    ("provisions", """MATCH (d:Document {document_id: r.document_id})
        MERGE (p:Provision {segment_id: r.segment_id}) SET p += r
        MERGE (p)-[:PART_OF]->(d)"""),
    ("chunks", """MATCH (p:Provision {segment_id: r.segment_id})
        MERGE (c:Chunk {chunk_id: r.chunk_id}) SET c += r
        MERGE (c)-[:PART_OF]->(p)"""),
    ("amends", """MATCH (c:Chunk {chunk_id: r.chunk_id})
        MATCH (p:Provision {segment_id: r.segment_id})
        MERGE (c)-[:AMENDS]->(p)"""),
    ("refers_to", """MATCH (c:Chunk {chunk_id: r.chunk_id})
        MATCH (p:Provision {segment_id: r.segment_id})
        MERGE (c)-[x:REFERS_TO]->(p) SET x.text = r.text"""),
    ("mentions", """MATCH (c:Chunk {chunk_id: r.chunk_id})
        MERGE (t:Term {label: r.label, value: r.value})
        MERGE (c)-[x:MENTIONS]->(t) ON CREATE SET x.count = 1
        ON MATCH SET x.count = x.count + 1"""),
    ("rules", "MERGE (u:Rule {rule_key: r.rule_key}) SET u += r"),
    ("versions", """MATCH (u:Rule {rule_key: r.rule_key})
        MERGE (v:RuleVersion {rule_id: r.rule_id}) SET v += r,
            v.valid_from = date(r.valid_from),
            v.valid_to = CASE WHEN r.valid_to IS NULL THEN NULL ELSE date(r.valid_to) END
        MERGE (v)-[:VERSION_OF]->(u)"""),
    ("defined_in", """MATCH (v:RuleVersion {rule_id: r.rule_id})
        MATCH (c:Chunk {chunk_id: r.chunk_id})
        MERGE (v)-[x:DEFINED_IN]->(c) SET x.primary = r.primary"""),
    ("supersedes", """MATCH (v:RuleVersion {rule_id: r.rule_id})
        MATCH (p:RuleVersion {rule_id: r.previous})
        MERGE (v)-[:SUPERSEDES]->(p)"""),
    ("bands", """MATCH (v:RuleVersion {rule_id: r.rule_id})
        MERGE (b:Band {band_id: r.band_id}) SET b += r
        MERGE (v)-[:HAS_BAND]->(b)"""),
    ("applies_to", """MATCH (v:RuleVersion {rule_id: r.rule_id})
        MERGE (t:TaxpayerCategory {name: r.category})
        MERGE (v)-[:APPLIES_TO]->(t)"""),
    ("rule_amended_by", """MATCH (d:Document {document_id: r.document_id})
        MATCH (u:Rule {rule_key: r.rule_key})
        MERGE (d)-[x:AMENDS]->(u) SET x.rule_id = r.rule_id"""),
]  # fmt: skip


def load(driver, rows: dict, reset: bool = True, batch: int = 500) -> None:
    if reset:
        driver.execute_query("MATCH (n) DETACH DELETE n")
    for statement in CONSTRAINTS:
        driver.execute_query(statement)
    for key, cypher in WRITES:
        data = rows[key]
        for i in range(0, len(data), batch):
            driver.execute_query(f"UNWIND $rows AS r {cypher}", rows=data[i : i + batch])


def summary(rows: dict) -> str:
    counts = Counter({k: len(v) for k, v in rows.items()})
    return "  ".join(f"{k}={n}" for k, n in counts.items())


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import get_settings
    from data_collection.sources import load_registry
    from entity_recognition.chunking import load_chunks

    parser = argparse.ArgumentParser(description="Build the C3 knowledge graph in Neo4j.")
    parser.add_argument("--dry-run", action="store_true", help="count rows, do not connect")
    parser.add_argument("--keep", action="store_true", help="do not delete the existing graph")
    parser.add_argument(
        "--until",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="only the Acts certified on or before this date (the corpus as it was known then)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    documents = known_documents(load_registry().documents, args.until)
    doc_ids = {d.document_id for d in documents}
    chunks = [c for c in load_extracted(settings.extracted_dir) if c["document_id"] in doc_ids]
    if not chunks:
        print("No extracted chunks. Run: python -m entity_recognition.extract")
        return 1
    started = time.perf_counter()
    parsed = [c for c in load_chunks(settings.processed_dir) if c.document_id in doc_ids]
    versions = build_versions(parsed)
    rows = graph_rows(chunks, versions, documents)
    if args.until:
        print(f"Acts certified on or before {args.until}: " + ", ".join(sorted(doc_ids)))
    print(summary(rows))
    if args.dry_run:
        return 0

    from knowledge_graph.connection import get_driver

    with get_driver() as driver:
        driver.verify_connectivity()
        load(driver, rows, reset=not args.keep)
        records, _, _ = driver.execute_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"
        )
    print("Neo4j now holds: " + ", ".join(f"{r['label']} {r['n']}" for r in records))
    print(f"Built in {time.perf_counter() - started:.1f} s. Open http://localhost:7474 and try:")
    print("  MATCH (v:RuleVersion)-[:VERSION_OF]->(:Rule {rule_key:'personal_relief'}) RETURN v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
