"""Relief discovery for invoices and receipts: what to ask C3, and whether a relief applies.

Plain functions, no framework and no C3 call (that is `stages.discover_reliefs`). Nothing
here knows any particular relief except CONDITION_QUESTIONS: C3's conditions are plain text,
so C1 keeps the yes/no questions that decide them, per rule_key. Everything else (value,
kind, income types, citation) is whatever C3 returned.

A relief is applied only when nothing is assumed:
  - the income it applies to is one C1 calculates (employment), else not_supported,
  - the user confirmed the document date and it falls inside the YA, else needs_review /
    not_applied,
  - for a cap, the user confirmed the amount paid (C2's total only pre-fills it),
  - every condition question is answered "yes"; any "no" -> not_applied, any unanswered or
    no questions known for the relief -> needs_review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from pii import strip_identifiers
from tax_calculation_agent.engine import ReliefClaim, remaining_claimable
from tax_calculation_agent.ruleset import ReliefRule

CALCULATED_INCOME = frozenset({"employment"})  # income types the engine calculates (v1)
MAX_QUERY_CHARS = 2000  # RetrievalRequest.query_text limit


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    if_no: str  # why the relief cannot apply when the answer is "no"


# rule_key -> the questions that decide C3's conditions text for that relief
CONDITION_QUESTIONS: dict[str, tuple[Question, ...]] = {
    "solar_panel_relief": (
        Question(
            "resident",
            "Were you resident in Sri Lanka in this year of assessment?",
            "the relief is only for a resident individual",
        ),
        Question(
            "on_premises",
            "Are the solar panels fixed on your own premises?",
            "the solar panels must be fixed on your premises",
        ),
        Question(
            "national_grid",
            "Are the solar panels connected to the national grid?",
            "the solar panels must be connected to the national grid",
        ),
    ),
}


def relief_query(issuer: str | None, payment_for: str | None) -> str | None:
    """What to ask C3 for one document: the issuer and the user's "what was this for?".

    E-mail, NIC and phone numbers are stripped; the business name stays, since that is what
    names the relief ("... Solar ..."). Never adds the word "relief": a bare "relief" makes
    C3 list every relief in force. None when there is nothing to ask with.
    """
    parts = [strip_identifiers(x) for x in (issuer, payment_for) if x and x.strip()]
    text = " ".join(p for p in parts if p)[:MAX_QUERY_CHARS].strip()
    return text or None


@dataclass(frozen=True)
class Assessment:
    status: str  # a ReliefStatus
    reasons: tuple[str, ...]
    answers: dict[str, str | None]  # question_id -> "yes" / "no" / None, in question order
    claim: ReliefClaim | None  # set only when status is "applied"
    remaining_claimable: Decimal | None


def assess(
    rule: ReliefRule,
    *,
    document_type: str,
    document_ref: str,
    amount_paid: Decimal | None,
    document_date: date | None,
    answers: dict[str, str],
    ya_start: date,
    ya_end: date,
    employment_income: Decimal,
) -> Assessment:
    """Decide one relief for one document. Reasons are shown to the user, so no figures from
    the document beyond its date and no names."""
    questions = CONDITION_QUESTIONS.get(rule.rule_key, ())
    given = {q.question_id: answers.get(q.question_id) for q in questions}

    unsupported = []
    if not set(rule.income_types) & CALCULATED_INCOME:
        income = " or ".join(rule.income_types) or "another type of"
        unsupported.append(f"needs {income} income, which C1 does not calculate yet")
    elif rule.value_kind == "rate" and set(rule.income_types) != CALCULATED_INCOME:
        unsupported.append("is a share of an income C1 does not calculate yet")
    if unsupported:
        return Assessment("not_supported", tuple(unsupported), given, None, None)

    failed, missing = [], []
    if document_date is None:
        missing.append(f"confirm the date on the {document_type}")
    elif not ya_start <= document_date <= ya_end:
        failed.append(
            f"the {document_type} is dated {document_date}, outside this year of assessment "
            f"({ya_start} to {ya_end})"
        )
    if rule.value_kind == "cap" and amount_paid is None:
        missing.append(f"confirm the amount paid on the {document_type}")
    if rule.ref.valid_to is not None and rule.ref.valid_to < ya_end:
        missing.append(f"the rule ends on {rule.ref.valid_to}, before the year of assessment ends")
    if not questions:
        missing.append("C1 has no questions for this relief's conditions yet: a person must check")
    for q in questions:
        if given[q.question_id] == "no":
            failed.append(q.if_no)
        elif given[q.question_id] is None:
            missing.append(f"answer: {q.text}")

    if failed:
        return Assessment("not_applied", tuple(failed + missing), given, None, None)
    if missing:
        return Assessment("needs_review", tuple(missing), given, None, None)

    base = {"cap": amount_paid, "rate": employment_income, "amount": None}[rule.value_kind]
    claim = ReliefClaim(
        rule=rule,
        base=base,
        base_source=None if base is None else "user",
        document_ids=(document_ref,),
        document_type=document_type,
    )
    return Assessment("applied", (), given, claim, remaining_claimable(claim))


def combine(claims: list[ReliefClaim]) -> ReliefClaim:
    """Several documents claiming the same cap relief: one claim on the total paid (the cap
    is per year of assessment, not per document). Rate and fixed reliefs count once."""
    first = claims[0]
    if len(claims) == 1:
        return first
    docs = tuple(d for c in claims for d in c.document_ids)
    if first.rule.value_kind != "cap":
        return first.model_copy(update={"document_ids": docs})
    total = sum((c.base for c in claims if c.base is not None), Decimal(0))
    return first.model_copy(update={"base": total, "document_ids": docs})
