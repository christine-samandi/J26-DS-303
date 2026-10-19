"""Indexing cost experiment: LLM answer parsing, scoring with the shared scorer, consistency,
projection and the report. A fake LLM stands in for Gemini, so no key or network is needed."""

import pytest

pytest.importorskip("spacy")

from entity_recognition.evaluate import _total, evaluate  # noqa: E402
from evaluation.indexing_cost import PipelineCost, Stage, project, report  # noqa: E402
from evaluation.llm_baseline import (  # noqa: E402
    PROMPT,
    Call,
    consistency,
    not_in_text,
    parse_answer,
    run_baseline,
)

TEXTS = {
    "C1": "the personal relief of Rs. 1,800,000 shall apply from April 1, 2025",
    "C2": "a resident individual shall pay tax at 6% on employment income",
}
GOLD = {
    "C1": [("RELIEF", "personal relief"), ("AMOUNT", "Rs. 1,800,000"), ("DATE", "April 1, 2025")],
    "C2": [("TAXPAYER_CATEGORY", "resident individual"), ("RATE", "6%"),
           ("INCOME_TYPE", "employment income")],
}  # fmt: skip


class FakeLLM:
    """Answers like an LLM that misses one entity, invents one, and changes its mind once."""

    model = "fake-llm"

    def __init__(self):
        self.calls = 0

    def extract(self, text: str) -> Call:
        self.calls += 1
        if "personal relief" in text:
            ents = [("RELIEF", "personal relief"), ("AMOUNT", "Rs. 1,800,000")]  # misses DATE
        else:
            ents = [("TAXPAYER_CATEGORY", "resident individual"), ("RATE", "6%"),
                    ("INCOME_TYPE", "employment income"), ("AMOUNT", "Rs. 6")]  # fmt: skip
            if self.calls > 2:  # second run: a different answer
                ents = ents[:3]
        return Call(entities=ents, seconds=1.5, input_tokens=900, output_tokens=60)


def test_prompt_carries_the_annotation_guide_and_the_passage():
    assert "TAX_SLAB" in PROMPT and "Longest phrase wins" in PROMPT
    assert PROMPT.rstrip().endswith("{text}")


@pytest.mark.parametrize(
    "raw, entities, invalid",
    [
        ('[{"label": "rate", "text": " 6% "}]', [("RATE", "6%")], 0),
        ('```json\n[{"label": "AMOUNT", "text": "Rs. 5"}]\n```', [("AMOUNT", "Rs. 5")], 0),
        ('Sure! [{"label": "DATE", "text": "April 1, 2025"}]', [("DATE", "April 1, 2025")], 0),
        ('[{"label": "PERSON", "text": "x"}, {"label": "RATE", "text": ""}]', [], 2),
        ("not json at all", [], 1),
    ],
)
def test_answers_are_parsed_defensively(raw, entities, invalid):
    assert parse_answer(raw) == (entities, invalid)


def test_llm_is_scored_with_the_same_scorer_as_the_extractor():
    baseline = run_baseline(TEXTS, FakeLLM(), runs=2, pause_s=0)
    llm = _total(evaluate(TEXTS, GOLD, lambda cid, _t: baseline.run(0)[cid]).strict)
    assert (llm.tp, llm.fp, llm.fn) == (5, 1, 1)  # missed the date, invented "Rs. 6"
    ours = _total(evaluate(TEXTS, GOLD).strict)
    assert ours.recall == 1.0  # the extractor finds all six
    assert consistency(baseline) == 0.5  # C2 changed between runs
    assert not_in_text(baseline, TEXTS) == 1  # "Rs. 6" does not occur in the text


def test_entities_not_in_the_text_are_counted():
    class Inventor(FakeLLM):
        def extract(self, text):
            return Call(entities=[("AMOUNT", "Rs. 999")], seconds=1.0, input_tokens=10)

    assert not_in_text(run_baseline(TEXTS, Inventor(), runs=1, pause_s=0), TEXTS) == 2


def test_failed_calls_are_recorded_not_fatal():
    class Down(FakeLLM):
        def extract(self, text):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    baseline = run_baseline(TEXTS, Down(), runs=1, pause_s=0)
    assert len(baseline.errors) == 2 and baseline.run(0) == {"C1": [], "C2": []}


def _pipeline():
    p = PipelineCost(documents=7, pages=425, chunks=1000, characters=400_000, entities=3000)
    p.stages = [Stage("parse PDFs", 2.5), Stage("chunk", 0.01), Stage("extract entities", 0.5),
                Stage("rebuild rule versions", 0.01), Stage("graph rows", 0.01)]  # fmt: skip
    return p


def test_projection_scales_with_corpus_size():
    baseline = run_baseline(TEXTS, FakeLLM(), runs=1, pause_s=0)
    proj = project(baseline, TEXTS, _pipeline(), price_in=0.5, price_out=2.0)
    assert proj.input_tokens > 1000 * 300  # prompt overhead is paid once per chunk
    assert proj.output_tokens == round(120 / 6 * 3000)  # tokens per entity x corpus entities
    assert proj.hours == pytest.approx(1.5 * 1000 / 3600)
    assert proj.usd == pytest.approx((proj.input_tokens * 0.5 + proj.output_tokens * 2) / 1e6)
    assert project(baseline, TEXTS, _pipeline(), None, None).usd is None


def test_report_shows_both_sides():
    baseline = run_baseline(TEXTS, FakeLLM(), runs=2, pause_s=0)
    ours = evaluate(TEXTS, GOLD)
    llm = evaluate(TEXTS, GOLD, lambda cid, _t: baseline.run(0)[cid])
    proj = project(baseline, TEXTS, _pipeline(), None, None)
    text = report(_pipeline(), ours, llm, baseline, 0.5, 0, proj, len(TEXTS))
    assert "PART A" in text and "PART B" in text and "PROJECTION" in text
    assert "0.50 ms" in text and "1500 ms" in text and "set --price-in/--price-out" in text
    assert "PART B" not in report(_pipeline())
