"""Fiscal row-alignment: decide which printed amount belongs to which label (own method).

Why this module exists:
- On real payslips the amount column is often printed (or photographed) slightly lower or higher
  than the label column. OCR then puts each amount on the line of the NEXT or PREVIOUS label, so
  "Total Deductions" receives the PAYE amount and "Net Salary" receives the total deductions.
  The reading is right; the pairing is wrong. Found on a real Sri Lankan payslip (PS_001).
- Geometry alone cannot settle it: the amount sits between two rows. The document's own tax
  arithmetic can: only one pairing makes  net = gross - total deductions  hold.

Method:
  1. Start from the normal same-line pairing.
  2. For every amount field also consider the amount on the row below and the row above its label.
  3. Keep the pairing that passes the most HARD, then SOFT, fiscal rules, using as few shifted rows
     as possible; a shift is accepted only if it repairs a hard rule and is the unique best answer.
  4. The same misalignment holds within one printed block, so fields in the same contiguous block
     of label+amount rows get the same shift (marked "propagated", lower confidence).
Shifted values are marked realigned, so their confidence is capped and the user reviews them.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from entity_extraction.extractor import Candidate, Line, _amount_candidate, extract_fields, group_lines, split_amount
from entity_extraction.fiscal import RULES, evaluate

MAX_FIELDS = 9                 # 3^9 = 19,683 pairings at most: fast enough on CPU
PROPAGATED_FACTOR = 0.9        # inferred from neighbours, not checked by a rule itself
SHIFT_NAME = {1: "row below", -1: "row above"}


def _shifted(lines: List[Line], c: Candidate, s: int) -> Candidate | None:
    j = c.line_index + s
    if c.line_index < 0 or not 0 <= j < len(lines) or j == c.amount_line:
        return None
    _, amount = split_amount(lines[j])
    if not amount:
        return None
    label, _ = split_amount(lines[c.line_index])
    alt = _amount_candidate(c.field, label or lines[c.line_index].tokens, amount, c.match_score)
    if alt is None:
        return None
    alt.line_index, alt.amount_line, alt.realigned = c.line_index, j, True
    alt.notes.append(f"value taken from the {SHIFT_NAME[s]} its label (row alignment)")
    return alt


def _score(choice: Dict[str, Candidate]) -> Tuple[int, int]:
    checks = evaluate({f: c.value for f, c in choice.items()})
    return sum(c.passed for c in checks if c.hard), sum(c.passed for c in checks if not c.hard)


def _is_row(line: Line) -> bool:
    label, amount = split_amount(line)
    return bool(label) and bool(amount)


def realign(lines: List[Line], base: Dict[str, Candidate]) -> Tuple[Dict[str, Candidate], dict]:
    fields = [f for f, c in base.items() if isinstance(c.value, float) and c.line_index >= 0][:MAX_FIELDS]
    options = {f: [(0, base[f])] + [(s, alt) for s in (1, -1) if (alt := _shifted(lines, base[f], s))]
               for f in fields}
    report = {"applied": False, "reason": "same-line pairing kept"}
    if not fields or all(len(o) == 1 for o in options.values()):
        return base, report

    base_hard, base_soft = _score({f: base[f] for f in fields})
    scored = []
    for combo in itertools.product(*(options[f] for f in fields)):
        used = [c.amount_line for _, c in combo]
        if len(set(used)) < len(used):            # one printed amount cannot belong to two labels
            continue
        hard, soft = _score({f: c for f, (_, c) in zip(fields, combo)})
        shifts = sum(s != 0 for s, _ in combo)
        scored.append(((hard, soft, -shifts), combo))
    best_key, best = max(scored, key=lambda kc: kc[0])
    rivals = [c for k, c in scored if k == best_key and c is not best]
    if best_key[0] <= base_hard or rivals:
        report["reason"] = ("no other pairing repairs a hard rule" if best_key[0] <= base_hard
                            else "several pairings fit equally well; kept the printed pairing")
        return base, report

    out = dict(base)
    moved = {}
    for f, (s, c) in zip(fields, best):
        if s:
            out[f], moved[f] = c, s
    directions = set(moved.values())
    propagated = []
    if len(directions) == 1:
        d = directions.pop()
        rows = [out[f].line_index for f in moved]
        lo, hi = min(rows), max(rows)
        # Fields in a hard rule that the arithmetic confirmed IN PLACE (e.g. gross) are anchors: the
        # shift never spreads across them (a slip whose rows are all label+amount would otherwise
        # shift everything, including the verified gross).
        hard_fields = {f for r in RULES if r.hard for f in r.fields}
        anchors = [base[f].line_index for f, (s, _) in zip(fields, best) if s == 0 and f in hard_fields]
        floor = max([a for a in anchors if a < lo], default=-1)
        ceil = min([a for a in anchors if a > hi], default=len(lines))
        while lo - 1 > floor and _is_row(lines[lo - 1]):
            lo -= 1
        while hi + 1 < ceil and _is_row(lines[hi + 1]):
            hi += 1
        for f in fields:
            if f in moved or not lo <= base[f].line_index <= hi:
                continue
            alt = dict(options[f]).get(d)
            if alt is not None and alt.amount_line not in {c.amount_line for c in out.values()}:
                alt = replace(alt, ocr_confidence=round(alt.ocr_confidence * PROPAGATED_FACTOR, 4),
                              notes=alt.notes + ["same shift as the verified rows in this block"])
                out[f] = alt
                propagated.append(f)
    report = {"applied": True,
              "reason": f"arithmetic: {best_key[0]} hard rule(s) pass after re-pairing (was {base_hard})",
              "moved": {f: SHIFT_NAME[s] for f, s in moved.items()}, "propagated": propagated}
    return out, report


def extract_aligned(tokens: Sequence) -> Tuple[Dict[str, Candidate], dict]:
    """extract_fields + row-alignment check. Returns (candidates, alignment report)."""
    return realign(group_lines(tokens), extract_fields(tokens))
