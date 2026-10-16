"""Add a newly published Act to the knowledge graph without rebuilding it.

    python -m knowledge_graph.update DOC-IRA-AMEND-02-2025

What happens (the same pipeline as a full build, for one document):
1. parse the Act's PDF (it must be registered and downloaded, see data_collection);
2. extract its entities (Phase 2);
3. rebuild every rule's timeline from ALL Acts now known: the graph's Acts + the new one
   (cheap: a few statements per rule), so a version the new Act ends gets its end date;
4. write only what changed: the new Act's provisions and chunks with their AMENDS / REFERS_TO
   links, and the rule versions (new ones added, changed ones updated, none deleted unless the
   new timeline no longer has them);
5. print a change report: which versions were added or replaced, and their dates.

Typical use: data_collection.discover spots a new amendment Act on the IRD site -> add it to
corpus_sources.json -> download -> this command. The demo: build the graph as it was known
before 2025 (python -m knowledge_graph.build --until 2025-01-01), then add Act No. 2 of 2025.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from knowledge_graph.build import (
    PRINCIPAL,
    graph_rows,
    load,
    load_extracted,
)
from knowledge_graph.versions import RULES, Version, build_versions

RULE_KEYS = ("rules", "versions", "defined_in", "supersedes", "bands", "applies_to",
             "rule_amended_by")  # fmt: skip

CURRENT_VERSIONS = """
MATCH (v:RuleVersion)
OPTIONAL MATCH (v)-[:HAS_BAND]->(b:Band)
WITH v, b ORDER BY b.`order`
RETURN v.rule_id AS rule_id, v.rule_key AS rule_key, toString(v.valid_from) AS valid_from,
       toString(v.valid_to) AS valid_to, v.status AS status, v.amount AS amount,
       v.source_document_id AS source,
       [x IN collect(b) | [x.lower, x.upper, x.rate_percent]] AS bands
"""
KNOWN_DOCUMENTS = "MATCH (d:Document) RETURN d.document_id AS document_id"
DROP_VERSIONS = """
UNWIND $ids AS id
MATCH (v:RuleVersion {rule_id: id})
OPTIONAL MATCH (v)-[:HAS_BAND]->(b:Band)
DETACH DELETE v, b
"""


@dataclass
class Change:
    kind: str  # "+" added, "~" changed, "-" removed
    rule_id: str
    detail: str
    evidence: str = ""  # why: the statement in the Act, or the version that replaced it


def _value(v: dict) -> str:
    if v.get("amount") is not None:
        return f"Rs. {int(v['amount']):,}"
    rates = [f"{float(b[2]):g}%" for b in v.get("bands") or []]
    return "/".join(rates) + f" ({len(rates)} bands)" if rates else "?"


def _period(v: dict) -> str:
    return f"{v['valid_from']} .. {v['valid_to'] or 'now'}"


def version_changes(before: list[dict], after: dict[str, list[Version]]) -> list[Change]:
    """Compare the versions in the graph with the newly built timelines."""
    old = {v["rule_id"]: v for v in before}
    new, by_id = {}, {}
    for versions in after.values():
        for v in versions:
            by_id[v.rule_id] = v
            d = v.to_dict()
            new[v.rule_id] = {
                "rule_id": v.rule_id, "rule_key": v.rule_key, "valid_from": d["valid_from"],
                "valid_to": d["valid_to"], "status": v.status, "source": v.source_document_id,
                "amount": v.value if isinstance(v.value, int) else None,
                "bands": None if isinstance(v.value, int) else v.value,
            }  # fmt: skip
    changes = []
    for rid, v in new.items():
        if rid not in old:
            src = by_id[rid]
            changes.append(
                Change("+", rid, f"{_value(v)}  {_period(v)}  ({v['source']})",
                       f'{src.source_chunk_id}: "{src.matched_text}"')
            )  # fmt: skip
            continue
        o = old[rid]
        diffs = []
        if (o["status"], o["valid_to"]) != (v["status"], v["valid_to"]):
            diffs.append(f"{o['status']} -> {v['status']}"
                         + (f" (valid to {v['valid_to']})" if v["valid_to"] else ""))  # fmt: skip
        if o["valid_from"] != v["valid_from"]:
            diffs.append(f"valid from {o['valid_from']} -> {v['valid_from']}")
        if _value(o) != _value(v):
            diffs.append(f"value {_value(o)} -> {_value(v)}")
        if diffs:
            by = by_id[rid].superseded_by
            changes.append(Change("~", rid, "; ".join(diffs), f"replaced by {by}" if by else ""))
    changes += [Change("-", rid, "no longer in the timeline") for rid in old if rid not in new]
    return sorted(changes, key=lambda c: (c.rule_id.rsplit("-V", 1)[0], c.rule_id))


def delta_rows(rows: dict, new_documents: set[str]) -> dict:
    """Only the new Acts' document-level rows, plus every rule row (they are few)."""
    chunk_ids = {c["chunk_id"] for c in rows["chunks"] if c["document_id"] in new_documents}
    out = {
        "documents": [d for d in rows["documents"] if d["document_id"] in new_documents],
        "provisions": [p for p in rows["provisions"] if p["document_id"] in new_documents],
        "chunks": [c for c in rows["chunks"] if c["chunk_id"] in chunk_ids],
    }
    for key in ("amends", "refers_to", "mentions"):
        out[key] = [r for r in rows[key] if r["chunk_id"] in chunk_ids]
    for key in RULE_KEYS:
        out[key] = rows[key]
    return out


def report(new_documents: list, delta: dict, changes: list[Change]) -> str:
    lines = []
    for d in new_documents:
        n_chunks = sum(1 for c in delta["chunks"] if c["document_id"] == d.document_id)
        prefix = d.document_id.replace("DOC-", "CHUNK-") + "-"
        n_amends = len({a["segment_id"] for a in delta["amends"]
                        if a["chunk_id"].startswith(prefix)})  # fmt: skip
        lines.append(
            f"+ document  {d.title} (certified {d.certified_date}): "
            f"{n_chunks} chunks, {n_amends} provisions of the 2017 Act amended"
        )
    if not changes:
        lines.append("  no change to the tracked rules ("
                     + ", ".join(r.title.lower() for r in RULES.values()) + ")")  # fmt: skip
    for c in changes:
        lines.append(f"{c.kind} {c.rule_id:26} {c.detail}")
        if c.evidence:
            lines.append(f"  {'':26} {c.evidence}")
    return "\n".join(lines)


STAGES = 5


def _stage(n: int, title: str, step: bool) -> None:
    if step and n > 1:
        input("\n  (press Enter for the next step) ")
    print(f"\n[{n}/{STAGES}] {title}")


def main(argv: list[str] | None = None) -> int:
    from collections import Counter

    from c3_common.config import get_settings
    from data_collection.parse import parse_corpus
    from data_collection.sources import load_registry
    from entity_recognition.chunking import load_chunks
    from entity_recognition.extract import extract_corpus
    from knowledge_graph.connection import get_driver

    parser = argparse.ArgumentParser(description="Add newly published Acts to the graph.")
    parser.add_argument("documents", nargs="+", metavar="DOC_ID")
    parser.add_argument("--skip-parse", action="store_true", help="the Act is already parsed")
    parser.add_argument("--step", action="store_true", help="pause after each stage (for a demo)")
    args = parser.parse_args(argv)

    settings = get_settings()
    registry = load_registry()
    new_docs = [registry.get(doc_id) for doc_id in args.documents]
    if any(d.document_id == PRINCIPAL for d in new_docs):
        print("The principal Act is the base of the graph: rebuild with knowledge_graph.build")
        return 1
    started = time.perf_counter()
    print("New Act: " + "; ".join(f"{d.title} (certified {d.certified_date})" for d in new_docs))

    _stage(1, "Parse the PDF into its legal structure", args.step)
    for d in new_docs:
        if args.skip_parse:
            print(f"  {d.document_id}: already parsed")
            continue
        parsed = parse_corpus(settings.raw_docs_dir, settings.processed_dir, d.document_id)
        if not parsed:
            print(f"  {d.document_id} is not downloaded. Run: python -m data_collection.download")
            return 1
        kinds = Counter(seg.kind for seg in parsed[0].segments)
        print(f"  {d.document_id}: {parsed[0].page_count} pages -> "
              + ", ".join(f"{n} {k.replace('_', ' ')}s" for k, n in kinds.items()))  # fmt: skip

    _stage(2, "Extract tax entities (spaCy + regex, no LLM)", args.step)
    for d in new_docs:
        stats = extract_corpus(settings.processed_dir, settings.extracted_dir, d.document_id)
        top = sorted(stats["by_label"].items(), key=lambda x: -x[1])[:5]
        print(f"  {stats['chunks']} chunks, {stats['entities']} entities in {stats['seconds']} s: "
              + ", ".join(f"{label} {n}" for label, n in top))  # fmt: skip

    with get_driver() as driver:
        driver.verify_connectivity()
        known, _, _ = driver.execute_query(KNOWN_DOCUMENTS)
        doc_ids = {r["document_id"] for r in known} | set(args.documents)
        before, _, _ = driver.execute_query(CURRENT_VERSIONS)
        before = [dict(r) for r in before]

        _stage(3, "Rebuild the rule timelines with the new Act (versioning)", args.step)
        chunks = [c for c in load_extracted(settings.extracted_dir) if c["document_id"] in doc_ids]
        parsed = [c for c in load_chunks(settings.processed_dir) if c.document_id in doc_ids]
        versions = build_versions(parsed)
        changes = version_changes(before, versions)
        for key, vs in versions.items():
            touched = [c for c in changes if c.rule_id.startswith(RULES[key].id_prefix + "-")]
            print(f"  {RULES[key].title}: {len(vs)} versions"
                  + (f", {len(touched)} changed" if touched else ", unchanged"))  # fmt: skip

        _stage(4, "Write the changes to Neo4j (only what is new)", args.step)
        documents = [d for d in registry.documents if d.document_id in doc_ids]
        delta = delta_rows(graph_rows(chunks, versions, documents), set(args.documents))
        removed = [c.rule_id for c in changes if c.kind == "-"]
        if removed:
            driver.execute_query(DROP_VERSIONS, ids=removed)
        load(driver, delta, reset=False)
        print(f"  +{len(delta['documents'])} document, +{len(delta['provisions'])} provisions, "
              f"+{len(delta['chunks'])} chunks, +{len(delta['amends'])} AMENDS links, "
              f"+{len(delta['refers_to'])} REFERS_TO links; "
              f"{len(delta['versions'])} rule versions written")  # fmt: skip

    _stage(5, "Change report", args.step)
    print(report(new_docs, delta, changes))
    print(f"\nGraph updated in {time.perf_counter() - started:.1f} s. If the API is running, "
          "reload it: POST http://localhost:8003/reload")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
