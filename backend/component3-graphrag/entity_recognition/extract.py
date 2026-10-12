"""Rule-based tax entity extraction (spaCy + regex), with no LLM at indexing time.

Pipeline for one chunk of text:
1. spaCy tokenises the text (spacy.blank("en"): rules only, no statistical model to download).
2. Regex rules (patterns.REGEX_RULES) find formatted values: amounts, rates, dates, tax bands,
   legal references. Their character spans are aligned to spaCy tokens.
3. spaCy's PhraseMatcher finds lexicon terms (patterns.LEXICON): reliefs, deductions,
   exemptions, income types, taxpayer categories.
4. Overlaps are resolved with spacy.util.filter_spans (longest span wins), except TAX_SLAB,
   which is kept alongside the AMOUNTs inside it (a band and its limits are both facts).
5. Every entity gets a normalised value (60000, 18.0, "2025-04-01", "section 52(2)") and,
   for dates, the direction of the time condition ("on_or_after", "prior_to", ...).

Usage:  python -m entity_recognition.extract        (reads data/processed, writes data/extracted)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import cache
from pathlib import Path

from entity_recognition.chunking import Chunk, load_chunks
from entity_recognition.patterns import COMPILED, LEXICON, SCALES, TENS, UNITS


@dataclass
class Entity:
    label: str  # contract EntityLabel
    text: str  # verbatim from the chunk
    start: int  # character offsets in the chunk text
    end: int
    value: str | float | int | None  # normalised value
    rule: str  # which pattern found it (for error analysis)
    confidence: float
    qualifier: str | None = None  # DATE only: on_or_after | prior_to | from | on | None


# --- normalisation ------------------------------------------------------------------------


def words_to_number(text: str) -> int | None:
    """'one hundred and eighty thousand' -> 180000."""
    total, current = 0, 0
    for word in re.split(r"[\s-]+", text.lower()):
        if word == "and" or not word:
            continue
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in SCALES:
            total += (current or 1) * SCALES[word]
            current = 0
        else:
            return None
    return total + current


def _figure(text: str) -> float:
    return float(text.replace(",", ""))


def _int_if_whole(x: float) -> int | float:
    return int(x) if x == int(x) else x


def normalise_amount(text: str) -> int | float | None:
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)(?:\s+(million|billion))?", text)
    if m:
        value = _figure(m[1]) * (SCALES[m[2]] if m[2] else 1)
        return _int_if_whole(value)
    words = re.sub(r"\s+rupees\b", "", text, flags=re.I)
    return words_to_number(words)


def normalise_rate(text: str) -> float | None:
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m[1])
    n = words_to_number(re.sub(r"\s+per\s*cent(?:um)?|\s+percent", "", text, flags=re.I))
    return float(n) if n is not None else None


def normalise_date(text: str) -> str | None:
    clean = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    clean = re.sub(r"\bof\s+", "", clean).replace(",", " ")
    clean = " ".join(clean.split())
    for fmt in ("%B %d %Y", "%d %B %Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalise_slab(text: str) -> str:
    figures = [normalise_amount(rs) for rs in re.findall(r"Rs\.?\s*[^R]+?(?=\s+but\b|$)", text)]
    if text.lower().startswith("not"):
        return f"0-{figures[0]}"
    if len(figures) == 2:
        return f"{figures[0]}-{figures[1]}"
    return f"{figures[0]}-"


def normalise_reference(text: str) -> str:
    return " ".join(text.split()).lower().replace("sub-", "sub")


# words just before a date that say which way the time condition points
_QUALIFIERS = [
    ("on_or_after", r"(?:on\s+or\s+after|commencing\s+(?:on\s+or\s+after|from)|"
                    r"with\s+effect\s+from|from)\s*$"),
    ("prior_to", r"(?:prior\s+to|before|ending\s+(?:on\s+)?)\s*$"),
    ("on_or_before", r"on\s+or\s+before\s*$"),
]  # fmt: skip


def date_qualifier(text_before: str) -> str | None:
    tail = text_before[-40:].lower()
    for name, pattern in _QUALIFIERS:
        if re.search(pattern, tail):
            return name
    return None


_NORMALISERS = {
    "AMOUNT": normalise_amount,
    "RATE": normalise_rate,
    "DATE": normalise_date,
    "TAX_SLAB": normalise_slab,
    "SECTION_REF": normalise_reference,
}

# --- spaCy pipeline -----------------------------------------------------------------------


@cache
def _nlp():
    import spacy
    from spacy.matcher import PhraseMatcher

    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms: dict[str, tuple[str, str, float]] = {}
    for i, (label, value, phrases, confidence) in enumerate(LEXICON):
        key = f"LEX{i}"
        terms[key] = (label, value, confidence)
        matcher.add(key, [nlp.make_doc(p) for p in phrases])
    return nlp, matcher, terms


def extract_entities(text: str) -> list[Entity]:
    from spacy.util import filter_spans

    nlp, matcher, terms = _nlp()
    doc = nlp(text)
    # spaCy spans decide overlaps (token-aligned); the exact text and offsets come from the
    # match itself, so "gaming-40%" (no space in the PDF) still yields "40%" -> 40.0
    meta: dict[tuple[int, int, str], tuple[str, float, object, int, int]] = {}
    spans, slabs = [], []

    for rule, label, pattern, confidence in COMPILED:
        for m in pattern.finditer(text):
            span = doc.char_span(m.start(), m.end(), label=label, alignment_mode="expand")
            if span is None:
                continue
            meta[(span.start, span.end, label)] = (rule, confidence, None, m.start(), m.end())
            (slabs if label == "TAX_SLAB" else spans).append(span)

    for match_id, start, end in matcher(doc):
        label, value, confidence = terms[nlp.vocab.strings[match_id]]
        span = doc[start:end]
        span.label_ = label
        meta[(start, end, label)] = ("lexicon", confidence, value, span.start_char, span.end_char)
        spans.append(span)

    entities = []
    for span in sorted(filter_spans(spans) + filter_spans(slabs), key=lambda s: s.start_char):
        rule, confidence, value, start, end = meta[(span.start, span.end, span.label_)]
        surface = text[start:end]
        if value is None:
            value = _NORMALISERS[span.label_](surface)
        entities.append(
            Entity(
                label=span.label_,
                text=surface,
                start=start,
                end=end,
                value=value,
                rule=rule,
                confidence=confidence,
                qualifier=date_qualifier(text[:start]) if span.label_ == "DATE" else None,
            )
        )
    return entities


# --- corpus run ---------------------------------------------------------------------------


def extract_corpus(processed_dir: Path, out_dir: Path, only: str | None = None) -> dict:
    chunks = load_chunks(processed_dir, only)
    if not chunks:
        raise SystemExit(f"No parsed Acts in {processed_dir}. Run: python -m data_collection.parse")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    by_doc: dict[str, list[dict]] = {}
    labels: Counter[str] = Counter()
    for chunk in chunks:
        entities = extract_entities(chunk.text)
        labels.update(e.label for e in entities)
        record = chunk.to_dict() | {"entities": [asdict(e) for e in entities]}
        by_doc.setdefault(chunk.document_id, []).append(record)
    seconds = time.perf_counter() - started
    for doc_id, records in by_doc.items():
        payload = {"document_id": doc_id, "chunks": records}
        (out_dir / f"{doc_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return {
        "documents": len(by_doc),
        "chunks": len(chunks),
        "characters": sum(len(c.text) for c in chunks),
        "entities": sum(labels.values()),
        "by_label": dict(sorted(labels.items())),
        "seconds": round(seconds, 2),
    }


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import get_settings

    parser = argparse.ArgumentParser(description="Extract tax entities from the parsed Acts.")
    parser.add_argument("--only", metavar="DOC_ID")
    parser.add_argument("--text", help="extract from this text instead (quick check)")
    args = parser.parse_args(argv)

    if args.text:
        for e in extract_entities(args.text):
            extra = f" ({e.qualifier})" if e.qualifier else ""
            print(f"{e.label:18} {e.text!r:45} -> {e.value!r}{extra}  [{e.rule}, {e.confidence}]")
        return 0

    settings = get_settings()
    out_dir = settings.extracted_dir
    stats = extract_corpus(settings.processed_dir, out_dir, args.only)
    print(
        f"{stats['documents']} documents, {stats['chunks']} chunks, "
        f"{stats['characters']:,} characters, {stats['entities']:,} entities "
        f"in {stats['seconds']} s (no LLM calls)"
    )
    for label, count in stats["by_label"].items():
        print(f"  {label:18} {count:6,}")
    print(f"\nJSON written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

__all__ = ["Chunk", "Entity", "date", "extract_entities", "extract_corpus"]
