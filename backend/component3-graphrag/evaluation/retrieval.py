"""Phase 5: does C3 return the law that applied on the date asked? GraphRAG vs flat RAG.

    python -m evaluation.retrieval                 # uses the graph in Neo4j
    python -m evaluation.retrieval --source files  # rebuilds the graph data from data/ (no Neo4j)

Development set: ground_truth/retrieval_questions.csv -- everyday questions, each with an as-of date
(24 answerable: both PP1 rules on dates in every version, including the first and last day of
each version; 6 that the system must refuse). The correct answer for each question comes from
the hand-verified version tables in ground_truth/, never from the system. The development set
was used while building retrieval; for an unbiased score, run once on questions written by
people who have not seen the system:  --questions ground_truth/retrieval_questions_heldout.csv

Systems compared on the SAME chunks of the SAME seven Acts:
- Flat RAG: BM25 over all chunks (the standard sparse retriever), no graph, no dates;
- GraphRAG (C3): retrieval_api.graph_backend (text match + graph neighbourhood + date filter).
Both use word matching, so the difference measures what the graph and versioning add.

Metrics:
- correct evidence @1 / @3: the first (first three) chunks returned include the provision that
  sets the version in force on the date (from the ground-truth source provision);
- MRR: 1 / rank of the first correct chunk (0 if not in the top 5);
- outdated law @1: the first chunk is the provision of a DIFFERENT version of the same rule
  (an older or newer law than the one in force) -- the failure versioning is meant to prevent;
- correct answer: the returned value (amount or every band) and its validity dates equal the
  ground truth (GraphRAG only: flat RAG returns text, not values);
- refused out-of-scope: no result for questions the corpus cannot answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from j26_contracts.retrieval import RetrievalRequest
from retrieval_api.lexical import BM25

K = 5
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]


@dataclass
class GoldVersion:
    rule_key: str
    version: str
    value: int | list[list]
    valid_from: date
    valid_to: date | None
    document_id: str
    provision: str


@dataclass
class Question:
    qid: str
    text: str
    day: date
    rule_key: str | None  # None: the system should refuse
    why: str
    gold: GoldVersion | None = None


@dataclass
class Outcome:
    qid: str
    system: str
    chunks: list[str]
    hit1: bool = False
    hit3: bool = False
    rr: float = 0.0
    outdated1: bool = False
    answered: bool = False
    answer_correct: bool | None = None
    latency_ms: float = 0.0
    note: str = ""
    extra: dict = field(default_factory=dict)


# --- ground truth -------------------------------------------------------------------------


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _d(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


def gold_versions(gt_dir: Path) -> list[GoldVersion]:
    out = [
        GoldVersion("personal_relief", r["version"], int(r["amount_lkr"]), _d(r["valid_from"]),
                    _d(r["valid_to"]), r["source_document_id"], r["source_provision"])
        for r in _rows(gt_dir / "personal_relief_versions.csv")
    ]  # fmt: skip
    bands: dict[str, GoldVersion] = {}
    for r in _rows(gt_dir / "individual_tax_bands.csv"):
        g = bands.setdefault(
            r["version"],
            GoldVersion("individual_income_tax_slabs", r["version"], [], _d(r["valid_from"]),
                        _d(r["valid_to"]), r["source_document_id"], r["source_provision"]),
        )  # fmt: skip
        upper = int(r["upper_lkr"]) if r["upper_lkr"] else None
        g.value.append([int(r["lower_lkr"]), upper, float(r["rate_percent"])])
    return out + list(bands.values())


def load_questions(gt_dir: Path, path: Path | None = None) -> list[Question]:
    golds = gold_versions(gt_dir)
    questions = []
    for r in _rows(path or gt_dir / "retrieval_questions.csv"):
        key = None if r["expected_rule_key"] == "none" else r["expected_rule_key"]
        q = Question(r["qid"], r["question"], date.fromisoformat(r["as_of_date"]), key, r["why"])
        if key:
            (q.gold,) = [
                g
                for g in golds
                if g.rule_key == key
                and g.valid_from <= q.day
                and (g.valid_to is None or q.day <= g.valid_to)
            ]
        questions.append(q)
    return questions


def provision_chunks(g: GoldVersion, chunks: list[dict]) -> set[str]:
    """Chunks of the provision that sets this version, read from the ground-truth row:
    "s.55(2)(a) amending ..." -> section 55 of that Act; "Fifth Schedule para 2(a)" -> that
    Schedule paragraph of the 2017 Act."""
    m = re.match(r"s\.\s*(\d+)", g.provision)
    if m:
        return {c["chunk_id"] for c in chunks
                if c["document_id"] == g.document_id and c.get("section") == m[1]}  # fmt: skip
    m = re.match(r"(\w+)\s+Schedule\s+para\s+(\d+)", g.provision, re.I)
    schedule = f"{m[1].upper()} SCHEDULE"
    return {
        c["chunk_id"]
        for c in chunks
        if c["document_id"] == g.document_id
        and c.get("schedule") == schedule
        and c.get("paragraph") == m[2]
    }


# --- scoring one answer -------------------------------------------------------------------


def _rank_metrics(o: Outcome, right: set[str], other: set[str]) -> None:
    top = o.chunks[:K]
    o.hit1 = bool(top) and top[0] in right
    o.hit3 = any(c in right for c in top[:3])
    o.rr = next((1 / (i + 1) for i, c in enumerate(top) if c in right), 0.0)
    o.outdated1 = bool(top) and top[0] in other and top[0] not in right


def _same_value(result, gold: GoldVersion) -> bool:
    if isinstance(gold.value, int):
        return result.values.amount is not None and result.values.amount.amount == gold.value
    bands = [[b.lower, b.upper, float(b.rate_percent)] for b in result.values.bands or []]
    return bands == gold.value


# --- the two systems ----------------------------------------------------------------------


def run_flat(questions, chunks, golds_by_rule) -> list[Outcome]:
    index = BM25({c["chunk_id"]: c["text"] for c in chunks})
    out = []
    for q in questions:
        started = time.perf_counter()
        hits = index.search(q.text, K)
        o = Outcome(q.qid, "flat_rag", [cid for cid, _ in hits])
        o.latency_ms = (time.perf_counter() - started) * 1000
        o.answered = bool(hits)
        if q.gold:
            right = provision_chunks(q.gold, chunks)
            other = set().union(*(provision_chunks(g, chunks) for g in golds_by_rule[q.rule_key]
                                  if g.version != q.gold.version))  # fmt: skip
            _rank_metrics(o, right, other)
        out.append(o)
    return out


def run_graphrag(questions, chunks, golds_by_rule, data) -> list[Outcome]:
    from retrieval_api.graph_backend import retrieve

    out = []
    for q in questions:
        req = RetrievalRequest(schema_version="1.0.0", request_id=q.qid, query_text=q.text,
                               as_of_date=q.day, caller="evaluation")  # fmt: skip
        started = time.perf_counter()
        res = retrieve(req, data)
        evidence = list(dict.fromkeys(c for r in res.results for c in r.evidence.chunk_ids))
        o = Outcome(q.qid, "graphrag", evidence)
        o.latency_ms = (time.perf_counter() - started) * 1000
        o.answered = bool(res.results)
        o.extra = {"warnings": res.warnings,
                   "top": res.results[0].rule_id if res.results else ""}  # fmt: skip
        if q.gold:
            right = provision_chunks(q.gold, chunks)
            other = set().union(*(provision_chunks(g, chunks) for g in golds_by_rule[q.rule_key]
                                  if g.version != q.gold.version))  # fmt: skip
            _rank_metrics(o, right, other)
            top = res.results[0] if res.results else None
            o.answer_correct = bool(
                top
                and top.rule_key == q.rule_key
                and top.validity.in_force_on_as_of_date
                and top.validity.valid_from == q.gold.valid_from
                and top.validity.valid_to == q.gold.valid_to
                and _same_value(top, q.gold)
            )
        out.append(o)
    return out


# --- summary ------------------------------------------------------------------------------


def summarise(questions, outcomes: list[Outcome]) -> dict:
    by_q = {q.qid: q for q in questions}
    pos = [o for o in outcomes if by_q[o.qid].rule_key]
    neg = [o for o in outcomes if not by_q[o.qid].rule_key]

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    s = {
        "answerable_questions": len(pos),
        "out_of_scope_questions": len(neg),
        "correct_evidence_at_1": rate([o.hit1 for o in pos]),
        "correct_evidence_at_3": rate([o.hit3 for o in pos]),
        "mrr": rate([o.rr for o in pos]),
        "outdated_law_at_1": rate([o.outdated1 for o in pos]),
        "refused_out_of_scope": rate([not o.answered for o in neg]),
        "mean_latency_ms": round(statistics.mean(o.latency_ms for o in outcomes), 2),
    }
    if any(o.answer_correct is not None for o in pos):
        s["correct_answer"] = rate([bool(o.answer_correct) for o in pos])
        s["answered_out_of_scope"] = [o.qid for o in neg if o.answered]
        s["wrong_answers"] = [o.qid for o in pos if not o.answer_correct]
    return s


def report(flat: dict, graph: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x * 100:5.1f}%"

    rows = [
        ("Correct evidence @1", pct(flat["correct_evidence_at_1"]),
         pct(graph["correct_evidence_at_1"])),
        ("Correct evidence @3", pct(flat["correct_evidence_at_3"]),
         pct(graph["correct_evidence_at_3"])),
        ("MRR (top 5)", f"{flat['mrr']:.3f}", f"{graph['mrr']:.3f}"),
        ("Outdated law @1 (lower is better)", pct(flat["outdated_law_at_1"]),
         pct(graph["outdated_law_at_1"])),
        ("Correct answer (value + dates)", "n/a (text only)", pct(graph.get("correct_answer"))),
        ("Out-of-scope questions refused", pct(flat["refused_out_of_scope"]),
         pct(graph["refused_out_of_scope"])),
        ("Mean latency per question (ms)", f"{flat['mean_latency_ms']:.1f}",
         f"{graph['mean_latency_ms']:.1f}"),
    ]  # fmt: skip
    head = (f"{graph['answerable_questions']} answerable + {graph['out_of_scope_questions']} "
            "out-of-scope questions, each with an as-of date (2019-2026)\n")  # fmt: skip
    lines = [head, f"{'':36} {'Flat RAG (BM25)':>17} {'GraphRAG (C3)':>15}"]
    lines += [f"{name:36} {a:>17} {b:>15}" for name, a, b in rows]
    if graph.get("wrong_answers"):
        lines.append(f"\nGraphRAG wrong answers: {', '.join(graph['wrong_answers'])}")
    if graph.get("answered_out_of_scope"):
        lines.append(f"GraphRAG answered out-of-scope: {', '.join(graph['answered_out_of_scope'])}")
    return "\n".join(lines)


def _load_data(source: str):
    from c3_common.config import get_settings
    from knowledge_graph.build import load_extracted

    settings = get_settings()
    chunks = load_extracted(settings.extracted_dir)
    if source == "neo4j":
        from retrieval_api.graph_backend import graph

        return chunks, graph()
    from data_collection.sources import load_registry
    from entity_recognition.chunking import load_chunks
    from knowledge_graph.build import graph_rows
    from knowledge_graph.versions import build_versions
    from retrieval_api.graph_backend import from_rows

    versions = build_versions(load_chunks(settings.processed_dir))
    return chunks, from_rows(graph_rows(chunks, versions, load_registry().documents))


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import COMPONENT_DIR

    parser = argparse.ArgumentParser(description="Evaluate retrieval: GraphRAG vs flat RAG.")
    parser.add_argument("--source", choices=["neo4j", "files"], default="neo4j")
    parser.add_argument("--out", type=Path, default=COMPONENT_DIR / "data" / "evaluation")
    parser.add_argument(
        "--questions", type=Path, help="another question file, same columns (e.g. a held-out set)"
    )
    args = parser.parse_args(argv)

    gt_dir = COMPONENT_DIR / "ground_truth"
    questions = load_questions(gt_dir, args.questions)
    golds_by_rule: dict[str, list[GoldVersion]] = {}
    for g in gold_versions(gt_dir):
        golds_by_rule.setdefault(g.rule_key, []).append(g)
    chunks, data = _load_data(args.source)
    if not chunks:
        print("No extracted chunks. Run: python -m entity_recognition.extract")
        return 1

    run_graphrag(questions[:1], chunks, golds_by_rule, data)  # warm-up (spaCy start-up)
    flat = run_flat(questions, chunks, golds_by_rule)
    graph = run_graphrag(questions, chunks, golds_by_rule, data)
    s_flat, s_graph = summarise(questions, flat), summarise(questions, graph)
    print(report(s_flat, s_graph))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval_scores.json").write_text(
        json.dumps({"flat_rag": s_flat, "graphrag": s_graph}, indent=2) + "\n"
    )
    with (args.out / "retrieval_per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "question", "as_of_date", "expected", "system", "hit@1", "hit@3",
                    "outdated@1", "answered", "answer_correct", "top_chunks", "note"])  # fmt: skip
        by_q = {q.qid: q for q in questions}
        for o in flat + graph:
            q = by_q[o.qid]
            expected = f"{q.rule_key} {q.gold.version}" if q.gold else "refuse"
            w.writerow([o.qid, q.text, q.day, expected, o.system, o.hit1, o.hit3, o.outdated1,
                        o.answered, "" if o.answer_correct is None else o.answer_correct,
                        " ".join(o.chunks[:3]), o.extra.get("top", "")])  # fmt: skip
    print(f"\nScores and per-question results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
