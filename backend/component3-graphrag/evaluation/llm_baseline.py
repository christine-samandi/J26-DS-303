"""LLM baseline for entity extraction: the "LLM-indexed GraphRAG" approach, measured.

Existing GraphRAG systems (e.g. Microsoft GraphRAG) send every chunk to an LLM to find the
entities. This module does exactly that for the hand-annotated gold chunks, with the same
annotation guide a human annotator got, and scores the answers with the SAME scorer as the
spaCy + regex extractor (entity_recognition.evaluate), so the two are directly comparable.

Per call it records the time and the tokens the provider reports. Each chunk is sent `runs`
times (default 2) to measure whether the LLM gives the same answer twice.

Only the provider call is LLM-specific (`GeminiExtractor`); tests use a fake with the same
`extract(text) -> Call` method, so no key or network is needed in CI.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from entity_recognition.annotation import GUIDE, LABELS

DEFAULT_MODEL = "gemini-3.8-flash"  # same default as C1's explainer (GEMINI_MODEL overrides)

_START, _END = GUIDE.index("LABELS (what to mark, and the exact words to copy)"), None
PROMPT = "\n".join(
    [
        "You extract tax-law entities from a passage of a Sri Lankan Inland Revenue Act.",
        "",
        *GUIDE[_START:_END],
        "",
        "Return ONLY a JSON array, one object per entity, in the order they appear:",
        '[{"label": "AMOUNT", "text": "Rs. 1,800,000"}, ...]',
        f"label must be one of: {', '.join(LABELS)}. text must be copied exactly from the passage.",
        "If there are no entities, return [].",
        "",
        "PASSAGE:",
        "{text}",
    ]
)


class QuotaExhausted(RuntimeError):
    """The provider's quota is used up (e.g. free tier: N requests per day); retrying is useless."""


@dataclass
class Call:
    """One LLM answer for one chunk."""

    entities: list[tuple[str, str]]
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0  # includes any "thinking" tokens the provider bills
    invalid: int = 0  # items dropped: unknown label, empty text, unparsable


@dataclass
class Baseline:
    model: str
    calls: dict[str, list[Call]] = field(default_factory=dict)  # chunk_id -> one Call per run
    errors: list[str] = field(default_factory=list)

    def run(self, n: int) -> dict[str, list[tuple[str, str]]]:
        return {cid: calls[n].entities for cid, calls in self.calls.items() if len(calls) > n}

    def first_answer(self) -> dict[str, list[tuple[str, str]]]:
        """Per chunk, the first call that succeeded (a failed call is not an empty answer)."""
        return {cid: next((c.entities for c in calls if answered(c)), [])
                for cid, calls in self.calls.items()}  # fmt: skip

    def unanswered(self) -> int:
        """Chunks for which no call succeeded (they count as all-missed in the score)."""
        return sum(1 for calls in self.calls.values() if not any(answered(c) for c in calls))

    @classmethod
    def from_json(cls, data: dict) -> Baseline:
        """Rebuild from the "llm" part of indexing_cost.json (rescore without calling again)."""
        calls = {cid: [Call(**{**c, "entities": [tuple(e) for e in c["entities"]]}) for c in cs]
                 for cid, cs in data["calls"].items()}  # fmt: skip
        return cls(model=data["model"], calls=calls, errors=list(data.get("errors", [])))


def answered(call: Call) -> bool:
    """A successful call always reports input tokens; a failed one is recorded with 0."""
    return call.input_tokens > 0


def parse_answer(raw: str) -> tuple[list[tuple[str, str]], int]:
    """The JSON array the model returned -> [(label, text)], plus how many items were invalid."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            return [], 1
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], 1
    if not isinstance(items, list):
        return [], 1
    out, invalid = [], 0
    for item in items:
        label = str(item.get("label", "")).strip().upper() if isinstance(item, dict) else ""
        text = " ".join(str(item.get("text", "")).split()) if isinstance(item, dict) else ""
        if label in LABELS and text:
            out.append((label, text))
        else:
            invalid += 1
    return out, invalid


class GeminiExtractor:
    """Calls Gemini through google-genai (pip install -e ".[llm]"), temperature 0, JSON mode."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, retries: int = 4):
        from google import genai
        from google.genai import types

        self.model = model
        self.retries = retries
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"
        )

    def extract(self, text: str) -> Call:
        prompt = PROMPT.replace("{text}", text)
        for attempt in range(self.retries + 1):
            try:
                start = time.perf_counter()
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config
                )
                seconds = time.perf_counter() - start
                break
            except Exception as e:  # rate limit (429) or a transient error: wait and retry
                if "RESOURCE_EXHAUSTED" in str(e) and "PerDay" in str(e):
                    raise QuotaExhausted(f"daily quota of {self.model} used up") from e
                if attempt == self.retries:
                    raise
                wait = 15 * (attempt + 1)
                print(f"    {type(e).__name__}: retry in {wait} s")
                time.sleep(wait)
        usage = response.usage_metadata
        entities, invalid = parse_answer(response.text or "")
        return Call(
            entities=entities,
            seconds=seconds,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
            invalid=invalid,
        )


def run_baseline(texts: dict[str, str], extractor, runs: int = 2, pause_s: float = 4.0) -> Baseline:
    """Send every chunk `runs` times. The pause (free-tier rate limit) is not counted in time."""
    result = Baseline(model=getattr(extractor, "model", "?"))
    stopped = False
    for n in range(runs):
        for i, (chunk_id, text) in enumerate(texts.items(), 1):
            print(f"  run {n + 1}/{runs}, chunk {i}/{len(texts)}: {chunk_id}")
            if stopped:
                result.calls.setdefault(chunk_id, []).append(Call(entities=[], seconds=0.0))
                continue
            try:
                call = extractor.extract(text)
            except QuotaExhausted as e:
                print(f"    {e}: no more calls; the rest are recorded as not answered")
                result.errors.append(f"{chunk_id} (run {n + 1}): {e}")
                stopped = True
                call = Call(entities=[], seconds=0.0, invalid=1)
            except Exception as e:
                result.errors.append(f"{chunk_id} (run {n + 1}): {type(e).__name__}: {e}")
                call = Call(entities=[], seconds=0.0, invalid=1)
            result.calls.setdefault(chunk_id, []).append(call)
            if pause_s and not stopped and (n, i) != (runs - 1, len(texts)):
                time.sleep(pause_s)
    return result


def consistency(baseline: Baseline) -> tuple[float, int]:
    """Share of chunks for which every successful run returned exactly the same entities, and
    how many chunks that is based on (only chunks answered in at least two runs count)."""
    multi = [[c for c in calls if answered(c)] for calls in baseline.calls.values()]
    multi = [calls for calls in multi if len(calls) > 1]
    if not multi:
        return 1.0, 0
    same = sum(1 for calls in multi if all(sorted(c.entities) == sorted(calls[0].entities)
                                           for c in calls[1:]))  # fmt: skip
    return same / len(multi), len(multi)


def not_in_text(baseline: Baseline, texts: dict[str, str]) -> int:
    """Entities whose text does not occur in the chunk at all (invented or reworded)."""
    norm = lambda s: " ".join(s.split()).casefold()  # noqa: E731
    return sum(
        1
        for cid, calls in baseline.calls.items()
        for c in calls
        for _, t in c.entities
        if norm(t) not in norm(texts[cid])
    )
