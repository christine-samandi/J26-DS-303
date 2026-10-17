"""GraphRAG retrieval over the C3 knowledge graph (C3_MODE=graphrag).

    python -m retrieval_api.graph_backend "personal relief" --date 2024-06-01 [--history]

How a request is answered:
1. Text match: score every chunk against the question (retrieval_api.lexical).
2. Graph match: each rule has a neighbourhood in the graph -- the chunks that DEFINE its
   versions (direct evidence) and the chunks that belong to, AMEND or REFER TO the same
   provisions (one hop away). A rule's relevance is its best-matching chunk (one hop away
   counts 0.8). Plain keywords ("tax slabs", "personal relief") also select a rule.
3. Time: of each relevant rule, the version in force on `as_of_date` is returned (with
   `include_history`, the other versions too, marked not in force).
4. Every result carries structured values (amount or bands), the citation, validity dates,
   the evidence (chunk IDs, verbatim text, graph path) and a confidence score.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import cache

from j26_contracts.retrieval import (
    SCHEMA_VERSION,
    AppliesTo,
    Confidence,
    ConfidenceComponents,
    Evidence,
    GraphEdge,
    MatchedEntity,
    Metadata,
    Money,
    QueryEcho,
    RetrievalOutput,
    RetrievalRequest,
    RetrievedRule,
    RuleValues,
    SlabBand,
    Source,
    Validity,
)
from retrieval_api.lexical import LexicalIndex, tokens

MIN_RELEVANCE = 0.6  # below this a rule is not returned for a text query
NEAR_WEIGHT = 0.8  # a chunk one hop away in the graph counts a little less than evidence
WEIGHTS = {"semantic_similarity": 0.35, "graph_match": 0.25, "extraction_confidence": 0.15,
           "temporal_match": 0.25}  # fmt: skip
CONFIDENCE_METHOD = (
    "weighted sum (provisional, calibrated in Phase 5): 0.35 text coverage + 0.25 graph match "
    "+ 0.15 extraction confidence + 0.25 temporal match; high >= 0.80, medium >= 0.50"
)
KEYWORDS = {  # words that name a rule directly
    "personal_relief": ["personal relief", "relief", "allowance", "tax free"],
    "individual_income_tax_slabs": ["slab", "band", "bracket", "tax rate", "rates of tax",
                                    "progressive", "tax table", "income tax rate", "how much tax",
                                    "tax payable", "tax on my", "tax do i pay"],
}  # fmt: skip
RATE_TYPES = {"tax_slab": {"tax_slab", "tax_rate"}}  # a slab rule also answers "tax_rate"


@dataclass
class GraphData:
    """What retrieval needs from the graph, loaded once."""

    versions: list[dict]
    chunks: dict[str, dict]
    neighbourhood: dict[str, dict[str, set[str]]]  # rule_key -> {"direct": ..., "near": ...}
    corpus_version: str
    index: LexicalIndex = field(init=False)

    def __post_init__(self):
        self.index = LexicalIndex({cid: c["text"] for cid, c in self.chunks.items()})


# --- loading ------------------------------------------------------------------------------

VERSIONS_QUERY = """
MATCH (v:RuleVersion)-[:VERSION_OF]->(r:Rule)
MATCH (v)-[:DEFINED_IN {primary: true}]->(c:Chunk)-[:PART_OF]->(:Provision)-[:PART_OF]->(d:Document)
OPTIONAL MATCH (v)-[:HAS_BAND]->(b:Band)
OPTIONAL MATCH (v)-[:APPLIES_TO]->(t:TaxpayerCategory)
OPTIONAL MATCH (v)-[:DEFINED_IN]->(e:Chunk)
OPTIONAL MATCH (v)-[:SUPERSEDES]->(prev:RuleVersion)
OPTIONAL MATCH (nxt:RuleVersion)-[:SUPERSEDES]->(v)
OPTIONAL MATCH (c)-[:AMENDS]->(ap:Provision)
RETURN v {.*} AS v, r {.*} AS rule, d {.*} AS doc,
       c {.chunk_id, .section, .subsection, .schedule, .paragraph, .page_start, .text} AS chunk,
       collect(DISTINCT b {.`order`, .lower, .upper, .rate_percent}) AS bands,
       collect(DISTINCT t.name) AS categories, collect(DISTINCT e.chunk_id) AS evidence,
       prev.rule_id AS supersedes, nxt.rule_id AS superseded_by,
       collect(DISTINCT ap.segment_id) AS amends
"""
NEIGHBOURHOOD_QUERY = """
MATCH (r:Rule)<-[:VERSION_OF]-(:RuleVersion)-[:DEFINED_IN]->(e:Chunk)
OPTIONAL MATCH (e)-[:PART_OF|AMENDS]->(:Provision)<-[:PART_OF|AMENDS|REFERS_TO]-(n:Chunk)
RETURN r.rule_key AS rule_key, collect(DISTINCT e.chunk_id) AS direct,
       collect(DISTINCT n.chunk_id) AS near
"""
CHUNKS_QUERY = "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.text AS text"


def _native(value):
    return value.to_native() if hasattr(value, "to_native") else value


def from_neo4j(driver) -> GraphData:
    rows, _, _ = driver.execute_query(VERSIONS_QUERY)
    versions = []
    for r in rows:
        # Neo4j does not store null properties: a missing valid_to / amount means null
        v = {"valid_to": None, "amount": None} | {k: _native(x) for k, x in r["v"].items()}
        versions.append(
            v | {"rule": r["rule"], "doc": {k: _native(x) for k, x in r["doc"].items()},
                 "chunk": r["chunk"], "bands": sorted(r["bands"], key=lambda b: b["order"]),
                 "categories": sorted(r["categories"]), "evidence": r["evidence"],
                 "supersedes": r["supersedes"], "superseded_by": r["superseded_by"],
                 "amends": r["amends"]}
        )  # fmt: skip
    hood, _, _ = driver.execute_query(NEIGHBOURHOOD_QUERY)
    chunks, _, _ = driver.execute_query(CHUNKS_QUERY)
    return GraphData(
        versions=versions,
        chunks={c["chunk_id"]: {"text": c["text"]} for c in chunks},
        neighbourhood={
            h["rule_key"]: {"direct": set(h["direct"]), "near": set(h["near"]) - set(h["direct"])}
            for h in hood
        },
        corpus_version=f"neo4j:{len({v['doc']['document_id'] for v in versions})}-acts-with-rules"
        f":{len(chunks)}-chunks",
    )


def from_rows(rows: dict) -> GraphData:
    """The same data straight from knowledge_graph.build.graph_rows (tests, no database)."""
    chunks = {c["chunk_id"]: c for c in rows["chunks"]}
    docs = {d["document_id"]: d for d in rows["documents"]}
    rules = {r["rule_key"]: r for r in rows["rules"]}
    links = [(c["chunk_id"], c["segment_id"]) for c in rows["chunks"]]
    links += [(a["chunk_id"], a["segment_id"]) for a in rows["amends"]]
    links_with_refs = links + [(x["chunk_id"], x["segment_id"]) for x in rows["refers_to"]]
    by_id = {v["rule_id"]: v for v in rows["versions"]}
    versions = []
    for v in rows["versions"]:
        rid = v["rule_id"]
        c = chunks[v["source_chunk_id"]]
        versions.append(
            v | {"valid_from": date.fromisoformat(v["valid_from"]),
                 "valid_to": date.fromisoformat(v["valid_to"]) if v["valid_to"] else None,
                 "rule": rules[v["rule_key"]], "doc": docs.get(c["document_id"], {}),
                 "chunk": c,
                 "bands": sorted((b for b in rows["bands"] if b["rule_id"] == rid),
                                 key=lambda b: b["order"]),
                 "categories": sorted(a["category"] for a in rows["applies_to"]
                                      if a["rule_id"] == rid),
                 "evidence": [d["chunk_id"] for d in rows["defined_in"] if d["rule_id"] == rid],
                 "supersedes": next((s["previous"] for s in rows["supersedes"]
                                     if s["rule_id"] == rid), None),
                 "superseded_by": next((s["rule_id"] for s in rows["supersedes"]
                                        if s["previous"] == rid), None),
                 "amends": [a["segment_id"] for a in rows["amends"]
                            if a["chunk_id"] == c["chunk_id"]]}
        )  # fmt: skip
    hood: dict[str, dict[str, set[str]]] = {}
    for d in rows["defined_in"]:
        key = by_id[d["rule_id"]]["rule_key"]
        hood.setdefault(key, {"direct": set(), "near": set()})["direct"].add(d["chunk_id"])
    for h in hood.values():
        provisions = {seg for cid, seg in links if cid in h["direct"]}
        h["near"] = {cid for cid, seg in links_with_refs if seg in provisions} - h["direct"]
    return GraphData(versions, {k: {"text": c["text"]} for k, c in chunks.items()}, hood,
                     f"rows:{len(chunks)}-chunks")  # fmt: skip


@cache
def graph() -> GraphData:
    """Loaded on the first request (restart the API after rebuilding the graph)."""
    from knowledge_graph.connection import get_driver

    with get_driver() as driver:
        driver.verify_connectivity()
        return from_neo4j(driver)


# --- scoring ------------------------------------------------------------------------------


@dataclass
class Match:
    relevance: float
    coverage: float  # semantic_similarity component
    graph_match: float
    chunk_id: str | None  # best matching chunk


def _rule_matches(req: RetrievalRequest, data: GraphData) -> dict[str, Match]:
    """rule_key -> how well the rule matches the request (before looking at dates)."""
    keys = {v["rule_key"]: v["rule"] for v in data.versions}
    if req.rule_types:
        wanted = set(req.rule_types)
        keys = {k: r for k, r in keys.items()
                if wanted & RATE_TYPES.get(r["rule_type"], {r["rule_type"]})}  # fmt: skip
    if not req.query_text:
        return {k: Match(1.0, 1.0, 1.0, None) for k in keys}

    text = req.query_text.lower()
    matches = {}
    for key in keys:
        hood = data.neighbourhood.get(key, {"direct": set(), "near": set()})
        best = Match(0.0, 0.0, 0.0, None)
        for cid in hood["direct"] | hood["near"]:
            cov = data.index.coverage(req.query_text, cid)
            weight = 1.0 if cid in hood["direct"] else NEAR_WEIGHT
            if cov * weight > best.relevance:
                best = Match(cov * weight, cov, weight, cid)
        stems = " ".join(tokens(text))
        hits = [k for k in KEYWORDS.get(key, []) if k in text or k in stems]
        if hits:  # a phrase ("personal relief") names the rule more surely than one word
            best.relevance = max(best.relevance, 0.9 if any(" " in k for k in hits) else 0.7)
            best.graph_match = best.graph_match or 0.6
        if best.relevance >= MIN_RELEVANCE:
            matches[key] = best
    return matches


def _confidence(match: Match, extraction: float, in_force: bool) -> Confidence:
    parts = {
        "semantic_similarity": round(match.coverage, 3),
        "graph_match": round(match.graph_match, 3),
        "extraction_confidence": round(extraction, 3),
        "temporal_match": 1.0 if in_force else 0.0,
    }
    score = round(sum(WEIGHTS[k] * v for k, v in parts.items()), 3)
    level = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return Confidence(score=score, level=level, components=ConfidenceComponents(**parts))


# --- building one result ------------------------------------------------------------------


def _values(v: dict) -> RuleValues:
    if v.get("bands"):
        bands = [
            SlabBand(order=b["order"], lower=b["lower"], upper=b["upper"],
                     rate_percent=b["rate_percent"])
            for b in v["bands"]
        ]  # fmt: skip
        return RuleValues(bands=bands)
    return RuleValues(amount=Money(amount=v["amount"], currency="LKR", period="year_of_assessment"))


def _text(v: dict) -> str:
    if v.get("bands"):
        parts = []
        for b in v["bands"]:
            if b["upper"] is None:
                parts.append(f"{b['rate_percent']:g}% above Rs. {b['lower']:,}")
            else:
                parts.append(f"{b['rate_percent']:g}% on Rs. {b['lower']:,} - {b['upper']:,}")
        return ("The taxable income of a resident or non-resident individual for a year of "
                "assessment is taxed at: " + "; ".join(parts) + ".")  # fmt: skip
    return (f"A resident individual is entitled to a personal relief of Rs. {v['amount']:,} for "
            "each year of assessment.")  # fmt: skip


def _source(v: dict) -> Source:
    doc, c = v["doc"], v["chunk"]
    if c.get("schedule"):
        where = f"{c['schedule'].title()} para {c['paragraph']}"
    else:
        where = f"s.{c['section']}" + (f"({c['subsection']})" if c.get("subsection") else "")
    citation = f"{doc['title']}, {where}"
    if v.get("amends"):
        citation += f" (amending {v['rule']['provision']} of Act No. 24 of 2017)"
    citation += f", p.{c['page_start']}"
    return Source(
        document_id=doc["document_id"], document_title=doc["title"],
        document_type=doc["document_type"], act_number=doc.get("act_number"),
        section=c.get("section"), subsection=c.get("subsection"),
        paragraph=c.get("paragraph"), schedule=c.get("schedule"), page=c.get("page_start"),
        citation=citation,
        published_date=date.fromisoformat(str(doc["certified_date"]))
        if doc.get("certified_date") else None,
        url=doc.get("url"),
    )  # fmt: skip


def _entities(text: str) -> tuple[list[MatchedEntity], float]:
    from entity_recognition.extract import extract_entities

    found = extract_entities(text)
    if not found:
        return [], 0.9
    return (
        [MatchedEntity(text=e.text, label=e.label) for e in found],
        sum(e.confidence for e in found) / len(found),
    )


def _result(v: dict, match: Match, day: date) -> RetrievedRule:
    in_force = v["valid_from"] <= day and (v["valid_to"] is None or day <= v["valid_to"])
    entities, extraction = _entities(v["matched_text"])
    c = v["chunk"]
    path = [
        GraphEdge(from_=v["rule_id"], relation="DEFINED_IN", to=c["chunk_id"]),
        GraphEdge(from_=c["chunk_id"], relation="PART_OF", to=v["doc"]["document_id"]),
    ]
    path += [GraphEdge(from_=c["chunk_id"], relation="AMENDS", to=seg) for seg in v["amends"]]
    path += [GraphEdge(from_=v["rule_id"], relation="APPLIES_TO", to=t) for t in v["categories"]]
    if v.get("supersedes"):
        path.append(GraphEdge(from_=v["rule_id"], relation="SUPERSEDES", to=v["supersedes"]))
    evidence = list(dict.fromkeys([c["chunk_id"]] + list(v["evidence"])))
    return RetrievedRule(
        rank=1,
        rule_id=v["rule_id"],
        rule_key=v["rule_key"],
        rule_type=v["rule"]["rule_type"],
        title=v["rule"]["title"],
        text=_text(v),
        values=_values(v),
        applies_to=AppliesTo(
            income_types=list(v["income_types"]), taxpayer_categories=v["categories"] or ["any"]
        ),  # fmt: skip
        source=_source(v),
        validity=Validity(
            valid_from=v["valid_from"],
            valid_to=v["valid_to"],
            status=v["status"],
            in_force_on_as_of_date=in_force,
            superseded_by=v.get("superseded_by"),
            supersedes=v.get("supersedes"),
            changed_by_document_id=v["source_document_id"],
        ),  # fmt: skip
        confidence=_confidence(match, extraction, in_force),
        evidence=Evidence(
            chunk_ids=evidence,
            matched_text=v["matched_text"],
            graph_path=path,
            matched_entities=entities or None,
        ),  # fmt: skip
    )


# --- the request --------------------------------------------------------------------------


def _applies(v: dict, req: RetrievalRequest) -> bool:
    if req.taxpayer_category and req.taxpayer_category != "any":
        if v["categories"] and req.taxpayer_category not in v["categories"]:
            return False
    if req.income_types and not set(req.income_types) & set(v["income_types"]):
        return False
    return True


def retrieve(req: RetrievalRequest, data: GraphData | None = None) -> RetrievalOutput:
    started = time.perf_counter()
    data = data or graph()
    day = req.as_of_date
    warnings: list[str] = []

    if req.rule_ids:
        pool = [v for v in data.versions if v["rule_id"] in req.rule_ids]
        matches = {v["rule_key"]: Match(1.0, 1.0, 1.0, None) for v in pool}
    else:
        matches = _rule_matches(req, data)
        pool = [v for v in data.versions if v["rule_key"] in matches and _applies(v, req)]

    results = [_result(v, matches[v["rule_key"]], day) for v in pool]
    results.sort(key=lambda r: (not r.validity.in_force_on_as_of_date,
                                -matches[r.rule_key].relevance, r.rule_key or "",
                                r.validity.valid_from))  # fmt: skip
    shown = (
        results
        if req.include_history
        else [r for r in results if r.validity.in_force_on_as_of_date]
    )
    shown = shown[: req.top_k]
    for rank, r in enumerate(shown, start=1):
        r.rank = rank

    tracked = {v["rule"]["rule_type"] for v in data.versions} | {"tax_rate"}
    if not pool:
        warnings += ["NO_MATCH", "PARTIAL_CORPUS_COVERAGE"]
    elif not any(r.validity.in_force_on_as_of_date for r in results):
        warnings.append("NO_RULE_IN_FORCE_ON_DATE")
        if day < min(v["valid_from"] for v in pool):
            warnings.append("AS_OF_DATE_BEFORE_CORPUS")
    if req.rule_types and set(req.rule_types) - tracked:
        warnings.append("PARTIAL_CORPUS_COVERAGE")
    if len(matches) > 1 and not req.rule_ids and not req.rule_types:
        top = sorted((m.relevance for m in matches.values()), reverse=True)
        if top[0] - top[1] < 0.1:
            warnings.append("AMBIGUOUS_QUERY")
    if any(r.validity.in_force_on_as_of_date and r.confidence.level == "low" for r in shown):
        warnings.append("LOW_CONFIDENCE")

    echo = req.model_dump(
        include={"query_text", "income_types", "taxpayer_category", "rule_types", "rule_ids"},
        exclude_unset=True,
    )
    return RetrievalOutput(
        schema_version=SCHEMA_VERSION,
        request_id=req.request_id,
        generated_at=datetime.now(UTC),
        query=QueryEcho(
            **echo, as_of_date=day, top_k=req.top_k, include_history=req.include_history
        ),
        results=shown,
        warnings=list(dict.fromkeys(warnings)),
        metadata=Metadata(
            retrieval_mode="graphrag",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            corpus_version=data.corpus_version,
            embedding_model="none (TF-IDF term coverage; dense embeddings planned)",
            candidates_considered=len(pool),
            confidence_method=CONFIDENCE_METHOD,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the knowledge graph a question.")
    parser.add_argument("question")
    parser.add_argument("--date", default=date.today().isoformat(), help="as_of_date, YYYY-MM-DD")
    parser.add_argument("--history", action="store_true", help="also show other versions")
    args = parser.parse_args(argv)
    req = RetrievalRequest(schema_version="1.0.0", request_id="cli", query_text=args.question,
                           as_of_date=date.fromisoformat(args.date),
                           include_history=args.history)  # fmt: skip
    out = retrieve(req)
    print(f"Q: {args.question!r} as of {args.date}   ({out.metadata.latency_ms} ms)")
    for r in out.results:
        flag = "IN FORCE" if r.validity.in_force_on_as_of_date else "not in force"
        print(f"\n#{r.rank} {r.rule_id}  [{flag}]  confidence {r.confidence.score} "
              f"({r.confidence.level})")  # fmt: skip
        print(f"   {r.text}")
        print(f"   valid {r.validity.valid_from} .. {r.validity.valid_to or 'now'}  "
              f"status today: {r.validity.status}")  # fmt: skip
        print(f"   source:   {r.source.citation}")
        print(f'   evidence: {r.evidence.chunk_ids[0]}: "{r.evidence.matched_text}"')
    if out.warnings:
        print(f"\nwarnings: {', '.join(out.warnings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
