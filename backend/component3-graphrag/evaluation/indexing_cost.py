"""Indexing cost: this lightweight pipeline vs LLM-based extraction (proposal section 3.4).

    python -m evaluation.indexing_cost                  # Part A: this pipeline, no key needed
    python -m evaluation.indexing_cost --llm gemini     # + Part B: Gemini on the gold chunks

Part A runs the real indexing pipeline on the whole corpus and times every stage:
parse the PDFs -> chunk -> extract entities -> rebuild rule versions -> graph rows. It also runs
extraction twice to check the output is identical (deterministic). Nothing is written to
data/processed or Neo4j: parsing goes to a temporary folder.

Part B sends each hand-annotated gold chunk to an LLM (evaluation.llm_baseline) and scores it
with the same scorer as this extractor. From the measured tokens and time per chunk it projects
the cost of LLM extraction for the whole corpus, i.e. every time the index is rebuilt.

Honest limits, also written into the report: the LLM is measured on the gold chunks only and
scaled up; the result depends on the model and prompt (both recorded); prices are not built in,
pass them with --price-in / --price-out (USD per million tokens) from the provider's price page.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Stage:
    name: str
    seconds: float
    detail: str = ""


@dataclass
class PipelineCost:
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    characters: int = 0
    entities: int = 0
    stages: list[Stage] = field(default_factory=list)
    deterministic: bool = True

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.stages)

    @property
    def extraction_ms_per_chunk(self) -> float:
        s = next((s for s in self.stages if s.name == "extract entities"), None)
        return 1000 * s.seconds / self.chunks if s and self.chunks else 0.0


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def measure_pipeline(raw_dir: Path) -> PipelineCost:
    """Part A: the real pipeline on the whole corpus, timed stage by stage."""
    from data_collection.parse import parse_corpus
    from data_collection.sources import load_registry
    from entity_recognition.chunking import load_chunks
    from entity_recognition.extract import extract_entities
    from knowledge_graph.build import graph_rows
    from knowledge_graph.versions import build_versions

    cost = PipelineCost()
    with tempfile.TemporaryDirectory() as tmp:
        parsed, t = _timed(lambda: parse_corpus(raw_dir, Path(tmp)))
        cost.documents, cost.pages = len(parsed), sum(p.page_count for p in parsed)
        cost.stages.append(Stage("parse PDFs", t, f"{cost.documents} Acts, {cost.pages} pages"))
        chunks, t = _timed(lambda: load_chunks(Path(tmp)))
    cost.chunks, cost.characters = len(chunks), sum(len(c.text) for c in chunks)
    cost.stages.append(Stage("chunk", t, f"{cost.chunks} chunks, {cost.characters:,} characters"))

    extract_entities("warm-up")  # loads spaCy once; not part of the per-chunk time
    first, t = _timed(lambda: [extract_entities(c.text) for c in chunks])
    cost.entities = sum(len(e) for e in first)
    cost.stages.append(Stage("extract entities", t, f"{cost.entities:,} entities"))
    second = [extract_entities(c.text) for c in chunks]
    key = lambda ents: [(e.label, e.text, e.start, e.end) for e in ents]  # noqa: E731
    cost.deterministic = all(key(a) == key(b) for a, b in zip(first, second, strict=True))

    versions, t = _timed(lambda: build_versions(chunks))
    n_versions = sum(len(v) for v in versions.values())
    cost.stages.append(Stage("rebuild rule versions", t, f"{n_versions} versions"))
    from dataclasses import asdict

    extracted = [c.to_dict() | {"entities": [asdict(e) for e in ents]}
                 for c, ents in zip(chunks, first, strict=True)]  # fmt: skip
    rows, t = _timed(lambda: graph_rows(extracted, versions, load_registry().documents))
    n_rows = sum(len(v) for v in rows.values())
    cost.stages.append(Stage("graph rows", t, f"{n_rows:,} nodes and relationships"))
    return cost


@dataclass
class Projection:
    """LLM extraction for the whole corpus, scaled up from the measured gold chunks."""

    input_tokens: int
    output_tokens: int
    hours: float
    usd: float | None


def project(baseline, gold_texts: dict[str, str], pipeline: PipelineCost,
            price_in: float | None, price_out: float | None) -> Projection:  # fmt: skip
    """Tokens and time scale with the corpus: input with its characters (prompt overhead per
    chunk + text), output with the number of entities to return, time with the chunk count."""
    from evaluation.llm_baseline import PROMPT

    calls = [c for calls in baseline.calls.values() for c in calls if c.input_tokens]
    if not calls:
        return Projection(0, 0, 0.0, None)
    chars = sum(len(gold_texts[cid]) for cid, cs in baseline.calls.items() for c in cs
                if c.input_tokens)  # fmt: skip
    tokens_in = sum(c.input_tokens for c in calls)
    prompt_chars = len(PROMPT) - len("{text}")
    per_char = tokens_in / (chars + prompt_chars * len(calls))  # same tokenizer for both parts
    input_tokens = round(per_char * (pipeline.characters + prompt_chars * pipeline.chunks))
    entities = sum(len(c.entities) for c in calls) or 1
    output_tokens = round(sum(c.output_tokens for c in calls) / entities * pipeline.entities)
    hours = sum(c.seconds for c in calls) / len(calls) * pipeline.chunks / 3600
    usd = None
    if price_in is not None and price_out is not None:
        usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return Projection(input_tokens, output_tokens, hours, usd)


def report(pipeline: PipelineCost, ours=None, llm=None, baseline=None, consistency=None,
           invented=None, projection=None, gold_chunks: int = 0) -> str:  # fmt: skip
    from entity_recognition.evaluate import _total

    lines = [
        "PART A  This pipeline (spaCy + regex, no LLM), whole corpus",
        f"  {pipeline.documents} Acts, {pipeline.pages} pages, {pipeline.chunks} chunks, "
        f"{pipeline.characters:,} characters",
    ]
    for s in pipeline.stages:
        lines.append(f"  {s.name:24} {s.seconds:8.2f} s   {s.detail}")
    lines += [
        f"  {'TOTAL':24} {pipeline.total_seconds:8.2f} s",
        f"  extraction: {pipeline.extraction_ms_per_chunk:.2f} ms per chunk; "
        f"same output on a second run: {'yes' if pipeline.deterministic else 'NO'}",
        "  tokens: 0, API cost: 0, runs offline",
    ]
    if llm is None:
        return "\n".join(lines)
    calls = [c for cs in baseline.calls.values() for c in cs]
    ok = [c for c in calls if c.input_tokens]
    share, compared = consistency
    o, m = _total(ours.strict), _total(llm.strict)
    ol, ml = _total(ours.lenient), _total(llm.lenient)
    sec = sum(c.seconds for c in ok) / len(ok) if ok else 0.0
    lines += [
        "",
        f"PART B  LLM extraction ({baseline.model}), {gold_chunks} gold chunks x "
        f"{len(next(iter(baseline.calls.values()), []))} runs",
        f"  calls: {len(calls)}, answered: {len(ok)}, failed after retries: "
        f"{len(calls) - len(ok)} (provider errors), entities not found in the text: {invented}",
        f"  scored on the first answer per chunk; chunks never answered: {baseline.unanswered()}",
        "",
        f"  {'':30} {'spaCy + regex':>15} {'LLM':>15}",
        f"  {'precision (strict)':30} {o.precision:15.3f} {m.precision:15.3f}",
        f"  {'recall (strict)':30} {o.recall:15.3f} {m.recall:15.3f}",
        f"  {'F1 (strict)':30} {o.f1:15.3f} {m.f1:15.3f}",
        f"  {'F1 (lenient: boundaries)':30} {ol.f1:15.3f} {ml.f1:15.3f}",
        f"  {'time per chunk':30} {pipeline.extraction_ms_per_chunk:12.2f} ms "
        f"{sec * 1000:12.0f} ms",
        f"  {'tokens per chunk (in / out)':30} {'0 / 0':>15} "
        f"{(sum(c.input_tokens for c in ok) // max(len(ok), 1))!s:>7} / "
        f"{(sum(c.output_tokens for c in ok) // max(len(ok), 1))!s:<6}",
        f"  {'same answer on every run':30} {'100%':>15} {share:15.0%}"
        f"  ({compared} chunks answered twice)",
        "  Note: the extractor was refined on these chunks after its blind first score",
        "  (P 0.97 / R 1.00); the LLM sees them for the first time (zero-shot, same guide).",
    ]
    if projection and projection.input_tokens:
        cost = f"USD {projection.usd:,.2f}" if projection.usd is not None else \
            "set --price-in/--price-out"  # fmt: skip
        tokens = projection.input_tokens + projection.output_tokens
        lines += [
            "",
            f"PROJECTION  whole corpus ({pipeline.chunks} chunks), every time it is re-indexed",
            f"  {'':30} {'spaCy + regex':>15} {'LLM (projected)':>18}",
            f"  {'extraction time':30} {pipeline.stages[2].seconds:13.2f} s "
            f"{projection.hours * 60:14.1f} min",
            f"  {'tokens (in + out)':30} {0:15} {tokens:18,}",
            f"  {'API cost':30} {'0':>15} {cost:>18}",
            "  (LLM time = sequential calls at the measured average; scaled from the gold chunks)",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import COMPONENT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Indexing cost: this pipeline vs an LLM.")
    parser.add_argument("--llm", choices=["gemini"], help="also run the LLM baseline (Part B)")
    parser.add_argument("--model", default=None, help="LLM model (default: GEMINI_MODEL or "
                        "the same default as C1)")  # fmt: skip
    parser.add_argument("--runs", type=int, default=2, help="LLM runs per chunk (consistency)")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between LLM calls")
    parser.add_argument("--price-in", type=float, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens")
    parser.add_argument("--rescore", action="store_true", help="rebuild Part B from the saved "
                        "indexing_cost.json instead of calling the LLM again")  # fmt: skip
    parser.add_argument(
        "--sheet", type=Path, default=COMPONENT_DIR / "ground_truth" / "entity_annotations.xlsx"
    )
    parser.add_argument("--out", type=Path, default=COMPONENT_DIR / "data" / "evaluation")
    args = parser.parse_args(argv)
    settings = get_settings()

    print("Part A: timing the indexing pipeline on the whole corpus ...")
    pipeline = measure_pipeline(settings.raw_docs_dir)
    payload: dict = {"pipeline": {
        **{k: getattr(pipeline, k) for k in ("documents", "pages", "chunks", "characters",
                                             "entities", "deterministic")},
        "stages": [vars(s) for s in pipeline.stages],
        "total_seconds": round(pipeline.total_seconds, 3),
        "extraction_ms_per_chunk": round(pipeline.extraction_ms_per_chunk, 3),
    }}  # fmt: skip
    text = report(pipeline)

    saved = None
    if args.rescore:
        path = args.out / "indexing_cost.json"
        saved = json.loads(path.read_text()).get("llm") if path.exists() else None
        if not saved:
            print(f"No saved LLM results in {path}. Run with --llm gemini first.")
            return 1
    if args.llm or saved:
        from entity_recognition.annotation import read_annotations
        from entity_recognition.evaluate import _total, evaluate
        from evaluation.llm_baseline import (
            DEFAULT_MODEL,
            Baseline,
            GeminiExtractor,
            consistency,
            not_in_text,
            run_baseline,
        )

        texts, gold = read_annotations(args.sheet)
        if saved:
            baseline = Baseline.from_json(saved)
            model, runs = baseline.model, saved.get("runs", args.runs)
            print(f"Part B: rescoring the saved {model} answers (no LLM calls) ...")
        else:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                print("Part B needs GEMINI_API_KEY in .env (see README).")
                return 1
            model, runs = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL), args.runs
            print(f"Part B: {model} on {len(texts)} gold chunks x {runs} runs "
                  f"(pause {args.pause:g} s between calls, not counted) ...")  # fmt: skip
            baseline = run_baseline(texts, GeminiExtractor(key, model), runs, args.pause)
        ours = evaluate(texts, gold)
        first = baseline.first_answer()
        llm = evaluate(texts, gold, lambda cid, _t: first.get(cid, []))
        cons, invented = consistency(baseline), not_in_text(baseline, texts)
        proj = project(baseline, texts, pipeline, args.price_in, args.price_out)
        text = report(pipeline, ours, llm, baseline, cons, invented, proj, len(texts))
        payload["llm"] = {
            "model": model,
            "gold_chunks": len(texts),
            "runs": runs,
            "errors": baseline.errors,
            "scores_strict": {"ours": _total(ours.strict).as_dict(),
                              "llm": _total(llm.strict).as_dict()},
            "scores_lenient": {"ours": _total(ours.lenient).as_dict(),
                               "llm": _total(llm.lenient).as_dict()},
            "consistency": round(cons[0], 4),
            "consistency_chunks": cons[1],
            "unanswered_chunks": baseline.unanswered(),
            "entities_not_in_text": invented,
            "calls": {cid: [vars(c) for c in cs] for cid, cs in baseline.calls.items()},
            "projection": vars(proj),
            "prices_usd_per_million": {"input": args.price_in, "output": args.price_out},
        }  # fmt: skip
        for e in baseline.errors:
            print("  error:", e)

    print("\n" + text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "indexing_cost.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "indexing_cost.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nWritten to {args.out / 'indexing_cost.json'} and indexing_cost.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
