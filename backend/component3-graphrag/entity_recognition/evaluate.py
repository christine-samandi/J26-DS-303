"""Score the extractor against the hand annotations: precision, recall and F1 per label.

    python -m entity_recognition.evaluate        # reads ground_truth/entity_annotations.xlsx

Only chunks marked "done" in the workbook are scored, and the system is run on the exact chunk
text stored in the workbook, so the score does not change if chunking changes later.

Two matching modes (standard in NER evaluation):
- strict:  same label and the same words (whitespace, letter case and trailing punctuation
           ignored);
- lenient: same label and the texts overlap (one contains the other), e.g. gold "Section 52 of
           the principal enactment" vs system "Section 52". Shows boundary errors separately
           from entities that were missed or invented.

precision = TP / (TP + FP)   how much of what the system found is right
recall    = TP / (TP + FN)   how much of what is there the system found
F1        = 2PR / (P + R)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from entity_recognition.annotation import LABELS, read_annotations
from entity_recognition.extract import extract_entities, normalise_date


def _norm(text: str) -> str:
    return " ".join(text.split()).strip(" .,;:\"'“”‘’").casefold()


def _same(label: str, gold: str, pred: str) -> bool:
    """Exact match; for dates, the same calendar day (Excel may have stored the gold as a date)."""
    if _norm(gold) == _norm(pred):
        return True
    if label == "DATE":
        g = gold if re.fullmatch(r"\d{4}-\d{2}-\d{2}", gold) else normalise_date(gold)
        return g is not None and g == normalise_date(pred)
    return False


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class Result:
    chunks: int = 0
    gold: int = 0
    predicted: int = 0
    strict: dict[str, Tally] = field(default_factory=dict)
    lenient: dict[str, Tally] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)


def _match(gold: list[tuple[str, str]], pred: list[tuple[str, str]], lenient: bool):
    """One-to-one matching; returns (matched pairs, unmatched gold, unmatched predicted).

    Exact matches are paired first, so in lenient mode an overlap can only pair entities that
    have no exact partner (gold "Rs. 600,000" is never spent on a predicted "Rs. 6").
    """
    unmatched_pred = list(pred)
    matched, missed = [], []
    passes = [_same] + ([_overlaps] if lenient else [])
    todo = list(gold)
    for test in passes:
        still = []
        for g in todo:
            hit = next((p for p in unmatched_pred if p[0] == g[0] and test(g[0], g[1], p[1])), None)
            if hit is None:
                still.append(g)
            else:
                unmatched_pred.remove(hit)
                matched.append((g, hit))
        todo = still
    missed = todo
    return matched, missed, unmatched_pred


def _overlaps(label: str, gold: str, pred: str) -> bool:
    a, b = _norm(gold), _norm(pred)
    return bool(a and b) and (a in b or b in a)


def evaluate(texts: dict[str, str], gold: dict[str, list[tuple[str, str]]]) -> Result:
    result = Result(
        strict={label: Tally() for label in LABELS},
        lenient={label: Tally() for label in LABELS},
    )
    for chunk_id, text in texts.items():
        g = gold.get(chunk_id, [])
        p = [(e.label, e.text) for e in extract_entities(text)]
        result.chunks += 1
        result.gold += len(g)
        result.predicted += len(p)
        for mode, tallies in (("strict", result.strict), ("lenient", result.lenient)):
            matched, missed, extra = _match(g, p, lenient=mode == "lenient")
            for (label, _), _ in matched:
                tallies[label].tp += 1
            for label, t in missed:
                tallies[label].fn += 1
                if mode == "strict":
                    result.errors.append(_err(chunk_id, "missed (FN)", label, t, extra))
            for label, t in extra:
                tallies[label].fp += 1
                if mode == "strict":
                    result.errors.append(_err(chunk_id, "extra (FP)", label, t, missed))
    return result


def _err(chunk_id, kind, label, text, others) -> dict:
    near = [
        o[1]
        for o in others
        if o[0] == label and (_norm(o[1]) in _norm(text) or _norm(text) in _norm(o[1]))
    ]
    return {
        "chunk_id": chunk_id,
        "error": kind,
        "label": label,
        "text": text,
        "boundary_partner": near[0] if near else "",
    }


def _total(tallies: dict[str, Tally]) -> Tally:
    return Tally(*(sum(getattr(t, k) for t in tallies.values()) for k in ("tp", "fp", "fn")))


def report(result: Result) -> str:
    lines = [
        f"{result.chunks} annotated chunks, {result.gold} gold entities, "
        f"{result.predicted} predicted",
        "",
        f"{'label':18} {'gold':>5}   {'strict P':>8} {'R':>6} {'F1':>6}   "
        f"{'lenient P':>9} {'R':>6} {'F1':>6}",
    ]
    for label in LABELS:
        s, le = result.strict[label], result.lenient[label]
        support = s.tp + s.fn
        if support == 0 and s.fp == 0:
            continue
        lines.append(
            f"{label:18} {support:5}   {s.precision:8.2f} {s.recall:6.2f} {s.f1:6.2f}   "
            f"{le.precision:9.2f} {le.recall:6.2f} {le.f1:6.2f}"
        )
    s, le = _total(result.strict), _total(result.lenient)
    lines.append(
        f"{'ALL (micro)':18} {s.tp + s.fn:5}   {s.precision:8.2f} {s.recall:6.2f} {s.f1:6.2f}   "
        f"{le.precision:9.2f} {le.recall:6.2f} {le.f1:6.2f}"
    )
    return "\n".join(lines)


def save(result: Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": result.chunks,
        "gold_entities": result.gold,
        "predicted_entities": result.predicted,
        "strict": {k: v.as_dict() for k, v in result.strict.items()}
        | {"ALL": _total(result.strict).as_dict()},
        "lenient": {k: v.as_dict() for k, v in result.lenient.items()}
        | {"ALL": _total(result.lenient).as_dict()},
    }
    (out_dir / "extraction_scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / "extraction_errors.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["chunk_id", "error", "label", "text", "boundary_partner"]
        )
        writer.writeheader()
        writer.writerows(result.errors)


def main(argv: list[str] | None = None) -> int:
    from c3_common.config import COMPONENT_DIR

    parser = argparse.ArgumentParser(description="Score extraction against hand annotations.")
    parser.add_argument(
        "--sheet", type=Path, default=COMPONENT_DIR / "ground_truth" / "entity_annotations.xlsx"
    )
    parser.add_argument("--out", type=Path, default=COMPONENT_DIR / "data" / "evaluation")
    args = parser.parse_args(argv)
    if not args.sheet.exists():
        print(f"No workbook at {args.sheet}. Run: python -m entity_recognition.annotation")
        return 1
    texts, gold = read_annotations(args.sheet)
    if not texts:
        print("No chunk is marked 'done' yet (column D of its grey header row).")
        return 1
    result = evaluate(texts, gold)
    print(report(result))
    save(result, args.out)
    print(f"\nScores and error list written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
