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
5. print a change report: which versions were added or replaced, and their dates;
6. keep it as a `law_change` record (GET /notifications on the API; see
   c3_common.notifications). --no-notify skips this.

Typical use: data_collection.monitor (scheduled) spots a new Act on the IRD site and notifies
the admin -> the admin checks it, adds it to corpus_sources.json -> download -> this command.
The demo: build the graph as it was known before 2025
(python -m knowledge_graph.build --until 2025-01-01), then add Act No. 2 of 2025.
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
       v.cap AS cap, v.rate_percent AS rate_percent, v.source_document_id AS source,
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
    if v.get("cap") is not None:
        return f"up to Rs. {int(v['cap']):,}"
    if v.get("rate_percent") is not None:
        return f"{float(v['rate_percent']):g}%"
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
                "amount": v.value if v.value_kind == "amount" else None,
                "cap": v.value if v.value_kind == "cap" else None,
                "rate_percent": v.value if v.value_kind == "rate" else None,
                "bands": v.value if v.value_kind == "bands" else None,
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


def _money(v: Version) -> str:
    if v.value_kind == "cap":
        return f"up to Rs. {int(v.value):,}"
    if v.value_kind == "rate":
        return f"{float(v.value):g}%"
    if v.value_kind == "bands":
        return "/".join(f"{float(b[2]):g}%" for b in v.value) + f" ({len(v.value)} bands)"
    return f"Rs. {int(v.value):,}"


def rule_changes(changes: list[Change], versions: dict[str, list[Version]]) -> list:
    """The change report in plain words, one RuleChange per changed rule version."""
    from c3_common.notifications import RuleChange

    by_id = {v.rule_id: v for vs in versions.values() for v in vs}
    out = []
    for c in changes:
        v = by_id.get(c.rule_id)
        key = v.rule_key if v else c.rule_id
        title = RULES[key].title if key in RULES else c.rule_id
        if c.kind == "-" or v is None:
            out.append(RuleChange(c.rule_id, key, title, "removed",
                                  f"{title}: {c.rule_id} is no longer in the law"))  # fmt: skip
            continue
        since, until = v.valid_from.isoformat(), v.valid_to.isoformat() if v.valid_to else None
        if c.kind == "+":
            what = (
                "new relief"
                if v.version == 1 and RULES[key].rule_type == "relief"
                else ("new rule" if v.version == 1 else "new version")
            )
            summary = f"{title}: {what}, {_money(v)}, from {since}" + (
                f" to {until}" if until else ""
            )
            out.append(RuleChange(v.rule_id, key, title, "added", summary, since, until,
                                  f'"{v.matched_text}"'))  # fmt: skip
        elif v.status == "outdated":
            out.append(RuleChange(v.rule_id, key, title, "ended",
                                  f"{title}: withdrawn, it does not apply after {until}",
                                  since, until, c.evidence))  # fmt: skip
        elif v.status == "replaced":
            out.append(RuleChange(v.rule_id, key, title, "replaced",
                                  f"{title}: {_money(v)} now ends on {until}, replaced by "
                                  f"{v.superseded_by}", since, until, c.evidence))  # fmt: skip
        else:
            out.append(RuleChange(v.rule_id, key, title, "changed", f"{title}: {c.detail}",
                                  since, until, c.evidence))  # fmt: skip
    return out


def notify_law_change(new_documents: list, changes: list[Change],
                      versions: dict[str, list[Version]]):  # fmt: skip
    """Keep the change report as a notification for users (only if a rule changed)."""
    from c3_common import notifications

    items = rule_changes(changes, versions)
    if not items:
        return None
    names = "; ".join(d.title for d in new_documents)
    n = notifications.make(
        "law_change", "users", f"Tax law updated: {names}",
        f"{names} changed {len(items)} rule version(s). Calculations for the years concerned "
        "may change: please recalculate.",
        document_id=new_documents[0].document_id, url=new_documents[0].url, changes=items,
    )  # fmt: skip
    return notifications.add(n)


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


def apply_update(document_ids: list[str], *, skip_parse: bool = False, step: bool = False,
                 notify: bool = True) -> list[Change]:  # fmt: skip
    """Add these registered (and downloaded) Acts to the graph in Neo4j: parse, extract,
    rebuild the timelines, write only what changed, report it and notify users.
    Raises UpdateError when a document is missing; Neo4j errors propagate."""
    from collections import Counter

    from c3_common.config import get_settings
    from data_collection.parse import parse_corpus
    from data_collection.sources import load_registry
    from entity_recognition.chunking import load_chunks
    from entity_recognition.extract import extract_corpus
    from knowledge_graph.connection import get_driver

    settings = get_settings()
    registry = load_registry()
    new_docs = [registry.get(doc_id) for doc_id in document_ids]
    if any(d.document_id == PRINCIPAL for d in new_docs):
        raise UpdateError("The principal Act is the base of the graph: rebuild with "
                          "knowledge_graph.build")  # fmt: skip
    started = time.perf_counter()
    print("New Act: " + "; ".join(f"{d.title} (certified {d.certified_date})" for d in new_docs))

    _stage(1, "Parse the PDF into its legal structure", step)
    for d in new_docs:
        if skip_parse:
            print(f"  {d.document_id}: already parsed")
            continue
        parsed = parse_corpus(settings.raw_docs_dir, settings.processed_dir, d.document_id)
        if not parsed:
            raise UpdateError(f"{d.document_id} is not downloaded. Run: python -m "
                              "data_collection.download")  # fmt: skip
        kinds = Counter(seg.kind for seg in parsed[0].segments)
        print(f"  {d.document_id}: {parsed[0].page_count} pages -> "
              + ", ".join(f"{n} {k.replace('_', ' ')}s" for k, n in kinds.items()))  # fmt: skip

    _stage(2, "Extract tax entities (spaCy + regex, no LLM)", step)
    for d in new_docs:
        stats = extract_corpus(settings.processed_dir, settings.extracted_dir, d.document_id)
        top = sorted(stats["by_label"].items(), key=lambda x: -x[1])[:5]
        print(f"  {stats['chunks']} chunks, {stats['entities']} entities in {stats['seconds']} s: "
              + ", ".join(f"{label} {n}" for label, n in top))  # fmt: skip

    with get_driver() as driver:
        driver.verify_connectivity()
        known, _, _ = driver.execute_query(KNOWN_DOCUMENTS)
        doc_ids = {r["document_id"] for r in known} | set(document_ids)
        before, _, _ = driver.execute_query(CURRENT_VERSIONS)
        before = [dict(r) for r in before]

        _stage(3, "Rebuild the rule timelines with the new Act (versioning)", step)
        chunks = [c for c in load_extracted(settings.extracted_dir) if c["document_id"] in doc_ids]
        parsed = [c for c in load_chunks(settings.processed_dir) if c.document_id in doc_ids]
        versions = build_versions(parsed)
        changes = version_changes(before, versions)
        for key, vs in versions.items():
            touched = [c for c in changes if c.rule_id.startswith(RULES[key].id_prefix + "-")]
            print(f"  {RULES[key].title}: {len(vs)} versions"
                  + (f", {len(touched)} changed" if touched else ", unchanged"))  # fmt: skip

        _stage(4, "Write the changes to Neo4j (only what is new)", step)
        documents = [d for d in registry.documents if d.document_id in doc_ids]
        delta = delta_rows(graph_rows(chunks, versions, documents), set(document_ids))
        removed = [c.rule_id for c in changes if c.kind == "-"]
        if removed:
            driver.execute_query(DROP_VERSIONS, ids=removed)
        load(driver, delta, reset=False)
        print(f"  +{len(delta['documents'])} document, +{len(delta['provisions'])} provisions, "
              f"+{len(delta['chunks'])} chunks, +{len(delta['amends'])} AMENDS links, "
              f"+{len(delta['refers_to'])} REFERS_TO links; "
              f"{len(delta['versions'])} rule versions written")  # fmt: skip

    _stage(5, "Change report", step)
    print(report(new_docs, delta, changes))
    if notify:
        note = notify_law_change(new_docs, changes, versions)
        if note:
            print(f"\nNotification {note.id} recorded for users (GET /notifications).")
    print(f"\nGraph updated in {time.perf_counter() - started:.1f} s. If the API is running, "
          "reload it: POST http://localhost:8003/reload")  # fmt: skip
    return changes


class UpdateError(RuntimeError):
    """A document cannot be added (not registered, not downloaded, the principal Act)."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add newly published Acts to the graph.")
    parser.add_argument("documents", nargs="+", metavar="DOC_ID")
    parser.add_argument("--skip-parse", action="store_true", help="the Act is already parsed")
    parser.add_argument("--step", action="store_true", help="pause after each stage (for a demo)")
    parser.add_argument("--no-notify", action="store_true", help="do not record a notification")
    args = parser.parse_args(argv)
    try:
        apply_update(args.documents, skip_parse=args.skip_parse, step=args.step,
                     notify=not args.no_notify)  # fmt: skip
    except UpdateError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
