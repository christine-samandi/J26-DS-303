"""Fiscal-arbitrated multi-engine fusion (own method, Gap: single-engine OCR errors on photos).

Why this module exists:
- Two pre-trained readers (Tesseract and PaddleOCR) make DIFFERENT mistakes. Choosing one engine
  for the whole document throws away the other engine's correct values.
- Common ensemble OCR votes by confidence alone, but OCR confidence is often high on a misread
  digit. For tax documents there is a better referee: the document's own arithmetic
  (net = gross - deductions, EPF 8% / 12%, ETF 3%, deductions = EPF + APIT).

Method, per field (values are taken from each engine's extraction, all in the same image coordinates):
  1. Both engines give the same value  -> take it; agreement raises confidence.
  2. Only one engine found the field     -> take it, unchanged.
  3. Engines disagree on AMOUNTS         -> try every combination of the candidates and keep the
                                            one that passes the most hard, then soft, fiscal rules.
  4. No rule separates them (or text)    -> higher OCR confidence wins, confidence is lowered and
                                            the disagreement is reported, so it is flagged for review.
After fusion, the usual fiscal self-correction and confidence scoring run on the fused values, so
the output contract and explanation report stay the same as for a single engine.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from entity_extraction.alignment import extract_aligned
from entity_extraction.extractor import Candidate
from entity_extraction.fiscal import evaluate

AGREE_BOOST = 0.5          # two independent engines agreeing halve the remaining doubt
DISAGREE_FACTOR = 0.85     # unresolved disagreement: lower confidence so the field is reviewed
MAX_COMBINATIONS = 4096    # 2 engines x up to 12 disputed amounts; beyond that, fall back to confidence
FUSION_ENGINE = "fusion(tesseract-5+paddleocr-3)"


@dataclass
class FieldDecision:
    field: str
    chosen: str                       # engine(s) whose value was used, e.g. "tesseract+paddle"
    value: object
    reason: str                       # agreed | only <engine> | arithmetic | higher confidence
    options: Dict[str, object]        # engine -> value it read

    def to_dict(self) -> dict:
        return {"field": self.field, "chosen": self.chosen, "value": self.value, "reason": self.reason,
                "options": self.options}


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 0.01
    return " ".join(str(a).split()).casefold() == " ".join(str(b).split()).casefold()


def _groups(options: List[Tuple[str, Candidate]]) -> List[List[Tuple[str, Candidate]]]:
    """Group the engines' candidates for one field by value (agreeing engines share a group)."""
    groups: List[List[Tuple[str, Candidate]]] = []
    for eng, cand in options:
        for g in groups:
            if _same(g[0][1].value, cand.value):
                g.append((eng, cand))
                break
        else:
            groups.append([(eng, cand)])
    return groups


def _best(group: List[Tuple[str, Candidate]]) -> Candidate:
    return max(group, key=lambda ec: ec[1].ocr_confidence)[1]


def _conf(group) -> float:
    return max(c.ocr_confidence for _, c in group)


def _merged(group, reason: str, n_engines: int) -> Candidate:
    """The fused candidate for a chosen group, with its confidence adjusted for the evidence."""
    c = _best(group)
    conf = c.ocr_confidence
    if len(group) >= 2:
        conf = 1 - (1 - conf) * AGREE_BOOST
    elif reason == "higher confidence" and n_engines >= 2:
        conf = conf * DISAGREE_FACTOR
    return replace(c, ocr_confidence=round(min(conf, 0.99), 4), notes=list(c.notes) + [f"fusion: {reason}"])


def fuse_candidates(per_engine: Dict[str, Dict[str, Candidate]]) -> Tuple[Dict[str, Candidate], List[FieldDecision]]:
    """per_engine: engine -> {field: Candidate}. Returns fused candidates and one decision per field."""
    fields: List[str] = []
    for cands in per_engine.values():
        fields += [f for f in cands if f not in fields]

    fused: Dict[str, Candidate] = {}
    decisions: Dict[str, FieldDecision] = {}
    disputed: Dict[str, List[List[Tuple[str, Candidate]]]] = {}

    for f in fields:
        options = [(e, c[f]) for e, c in per_engine.items() if f in c]
        groups = _groups(options)
        opts = {e: c.value for e, c in options}
        if len(groups) == 1:
            g = groups[0]
            reason = "agreed" if len(g) >= 2 else f"only {g[0][0]}"
            fused[f] = _merged(g, reason, len(options))
            decisions[f] = FieldDecision(f, "+".join(e for e, _ in g), fused[f].value, reason, opts)
        elif all(isinstance(c.value, float) for _, c in options):
            disputed[f] = groups
        else:                                   # disagreeing text (issuer, period): no arithmetic referee
            g = max(groups, key=_conf)
            fused[f] = _merged(g, "higher confidence", len(options))
            decisions[f] = FieldDecision(f, "+".join(e for e, _ in g), fused[f].value, "higher confidence", opts)

    if disputed:
        names = list(disputed)
        base = {f: c.value for f, c in fused.items() if isinstance(c.value, float)}
        by_conf = tuple(max(disputed[f], key=_conf) for f in names)
        combos = list(itertools.product(*(disputed[f] for f in names)))
        if len(combos) > MAX_COMBINATIONS:
            combos = [by_conf]

        def score(choice):
            values = dict(base, **{f: g[0][1].value for f, g in zip(names, choice)})
            checks = evaluate(values)
            hard = sum(c.passed for c in checks if c.hard)
            soft = sum(c.passed for c in checks if not c.hard)
            return hard, soft, sum(_conf(g) for g in choice)

        best = max(combos, key=score)
        decided_by_rules = score(best)[:2] > score(by_conf)[:2]
        for f, g in zip(names, best):
            reason = "arithmetic" if decided_by_rules else "higher confidence"
            fused[f] = _merged(g, reason, 2)
            opts = {e: c.value for grp in disputed[f] for e, c in grp}
            decisions[f] = FieldDecision(f, "+".join(e for e, _ in g), fused[f].value, reason, opts)

    return fused, [decisions[f] for f in fields]


def extract_fused(tokens_by_engine: Dict[str, Sequence], doc_id: str, processing_ms: Optional[int] = None,
                  to_original: Optional[Callable[[List[int]], List[int]]] = None, engine_label: str = FUSION_ENGINE):
    """Classify with every engine (keep the most confident answer), then fuse payslip fields.
    tokens_by_engine must all be in the same image coordinates (the same preprocessed image)."""
    from document_classifier import classify_tokens
    from entity_extraction.payslip import extract_document, extract_payslip

    engines = list(tokens_by_engine)
    classes = {e: classify_tokens(t) for e, t in tokens_by_engine.items()}
    known = [e for e in engines if classes[e].document_type != "unknown"]
    lead = max(known, key=lambda e: classes[e].confidence) if known else engines[0]
    cls = classes[lead]
    classification = {"document_type": cls.document_type, "confidence": cls.confidence, "scores": cls.scores,
                      "evidence": cls.evidence, "reason": f"{cls.reason} (from {lead})",
                      "per_engine": {e: classes[e].document_type for e in engines}}

    if cls.document_type != "payslip":
        out, report = extract_document(tokens_by_engine[lead], doc_id=doc_id, engine=engine_label,
                                       processing_ms=processing_ms, to_original=to_original)
        report["classification"] = classification
        report["fusion"] = {"note": f"no field fusion for {cls.document_type}; used {lead}"}
        return out, report

    per_engine = {e: extract_aligned(t)[0] for e, t in tokens_by_engine.items()}   # rows re-paired per engine
    fused, decisions = fuse_candidates(per_engine)
    # Sinhala/Tamil detection uses Tesseract's tokens when present (PaddleOCR has no Sinhala model)
    lang_tokens = tokens_by_engine.get("tesseract", tokens_by_engine[lead])
    out, report = extract_payslip(lang_tokens, doc_id, engine=engine_label, processing_ms=processing_ms,
                                  to_original=to_original, document_type="payslip", candidates=fused)
    report["classification"] = classification
    report["fusion"] = {"decisions": [d.to_dict() for d in decisions],
                        "summary": {r: sum(d.reason == r for d in decisions) for r in
                                    sorted({d.reason for d in decisions})}}
    return out, report
