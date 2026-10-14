"""The pipeline stages, as plain functions: `stage(state, deps) -> dict of state updates`.

No framework imports here. `orchestrator.graph` wires these into LangGraph; they can be
unit-tested, or driven by a plain loop, without it. A stage that cannot continue raises
StageError; the runner records it and goes straight to `deliver`.

    extract -> normalise -> retrieve -> calculate -> explain -> score -> assure
    assure: PASS / HUMAN_REVIEW -> deliver
            REGENERATE (attempt 1) -> regenerate -> rerun from the stage C4 names
            REGENERATE (attempt 2) -> deliver, needs review

The INTERPRET stage (LLM classification of ambiguous items) is not in v1: employment income
and APIT need no interpretation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from j26_contracts.retrieval import RetrievalRequest

from orchestrator.clients import AssuranceClient, OcrClient, RetrievalClient
from orchestrator.confidence import REVIEW_THRESHOLD, c1_confidence
from orchestrator.models import (
    MAX_ATTEMPTS,
    Claim,
    Explanation,
    OcrResult,
    OrchestratorResult,
    PipelineState,
    PipelineTrace,
    ProfileValue,
    RerunStage,
    TaxpayerProfile,
)
from tax_calculation_agent.engine import CalculationInputError, calculate
from tax_calculation_agent.ruleset import (
    RuleSet,
    RuleSetError,
    load_seed_rules,
    mid_year_changes,
    rule_conflicts,
    rules_from_retrieval,
    ya_period,
)

ZERO = Decimal(0)
ONE = Decimal(1)
RETRIEVAL_QUERY = (
    "Personal relief and income tax rates for a resident individual's employment income"
)


class StageError(Exception):
    """The request cannot be completed. The message is shown to the user, so no PII."""


@dataclass
class Deps:
    ocr: OcrClient
    retrieval: RetrievalClient
    assurance: AssuranceClient
    seed_rules: Callable[[str], RuleSet] = load_seed_rules
    review_threshold: Decimal = REVIEW_THRESHOLD


Update = dict  # the state keys a stage sets


def extract(state: PipelineState, deps: Deps) -> Update:
    request = state["request"]
    return {
        "ocr_results": tuple(deps.ocr.extract(request.request_id, d) for d in request.documents)
    }


def normalise(state: PipelineState, deps: Deps) -> Update:
    """Map typed input and OCR fields to a TaxpayerProfile. Typed values win."""
    request = state["request"]
    ocr = state.get("ocr_results", ())
    if len(ocr) > 1:
        raise StageError("more than one document: aggregating documents is not supported yet")
    warnings = list(state.get("warnings", ()))

    income = _profile_value(request.employment_income, "employment_income", ocr)
    if income is None:
        raise StageError("no employment income: type it in or upload a payslip")
    apit = _profile_value(request.apit_paid, "apit_paid", ocr)
    if apit is None:
        apit = ProfileValue(value=ZERO, source="user", confidence=ONE)
        warnings.append("APIT paid was not given, so it is taken as 0")
    return {
        "profile": TaxpayerProfile(employment_income=income, apit_paid=apit),
        "warnings": tuple(warnings),
    }


def retrieve(state: PipelineState, deps: Deps) -> Update:
    """Ask C3 for the rules in force at the start of the YA and cross-check the seed rules."""
    request = state["request"]
    ya = request.year_of_assessment
    try:
        as_of, _ = ya_period(ya)
    except RuleSetError as e:
        raise StageError(str(e)) from e
    output = deps.retrieval.retrieve(
        RetrievalRequest(
            schema_version="1.0.0",
            request_id=request.request_id,
            caller="c1-orchestrator",
            query_text=RETRIEVAL_QUERY,
            as_of_date=as_of,
            income_types=["employment"],
            taxpayer_category="resident_individual",
            rule_types=["relief", "tax_slab"],
        )
    )
    if output.request_id != request.request_id:
        raise StageError(f"C3 answered request {output.request_id}, not {request.request_id}")
    try:
        rules = rules_from_retrieval(output, ya)
    except RuleSetError as e:
        raise StageError(f"rules for YA {ya} could not be retrieved: {e}") from e

    warnings = [*state.get("warnings", ()), *(f"C3: {w}" for w in output.warnings)]
    warnings += mid_year_changes(rules)
    try:
        conflicts = rule_conflicts(deps.seed_rules(ya), rules)
    except RuleSetError:
        conflicts = []
        warnings.append(f"no seed rules for YA {ya}, so C3's rules were not cross-checked")
    return {
        "retrieval": output,
        "rules": rules,
        "conflicts": tuple(conflicts),
        "warnings": tuple(warnings),
    }


def calculate_tax(state: PipelineState, deps: Deps) -> Update:
    profile = state["profile"]
    try:
        calc = calculate(profile.employment_income.value, profile.apit_paid.value, state["rules"])
    except CalculationInputError as e:
        raise StageError(str(e)) from e
    return {"calc": calc}


def explain(state: PipelineState, deps: Deps) -> Update:
    """Deterministic skeleton: one line per step, numbers from the engine, citations attached.

    The LLM plain-language rewrite and its post-check come later (milestone 5).
    """
    rules = state["rules"]
    citations = {ref.rule_id: ref.citation for ref in _refs(rules)}
    retrieved = {r.rule_id for r in state["retrieval"].results}
    lines, claims = [], []
    for i, step in enumerate(state["calc"].steps, start=1):
        unsupported = [ref for ref in step.citation_refs if ref not in retrieved]
        if unsupported:  # rule 3: only cite what C3 actually returned
            raise StageError(f"{step.step_id} cites rules C3 did not return: {unsupported}")
        cited = "".join(f" [{citations[ref]}]" for ref in step.citation_refs)
        text = f"{step.description}: Rs. {step.result:,.2f}{cited}"
        lines.append(text)
        claims.append(
            Claim(
                claim_id=f"CL{i}", step_id=step.step_id, text=text, citation_refs=step.citation_refs
            )
        )
    return {"explanation": Explanation(lines=tuple(lines), claims=tuple(claims))}


def score(state: PipelineState, deps: Deps) -> Update:
    profile = state["profile"]
    rules = state["rules"]
    used = {ref.rule_id for ref in _refs(rules)}
    retrieval = [
        Decimal(str(r.confidence.score)) for r in state["retrieval"].results if r.rule_id in used
    ]
    return {
        "confidence": c1_confidence(
            ocr=[profile.employment_income.confidence, profile.apit_paid.confidence],
            retrieval=retrieval,
            interpretation=[],
            explanation_fell_back=state["explanation"].used_fallback,
            has_conflict=bool(state.get("conflicts")),
        )
    }


def assure(state: PipelineState, deps: Deps) -> Update:
    request = state["request"]
    attempt = state["attempt"]
    trace = PipelineTrace(
        request_id=request.request_id,
        attempt=attempt,
        year_of_assessment=request.year_of_assessment,
        profile=state["profile"],
        rules=state["rules"],
        retrieval_rule_ids=tuple(r.rule_id for r in state["retrieval"].results),
        calc_steps=state["calc"].steps,
        claims=state["explanation"].claims,
        confidence=state["confidence"],
        conflicts=state.get("conflicts", ()),
        warnings=state.get("warnings", ()),
    )
    verdict = deps.assurance.assure(trace)
    if (verdict.request_id, verdict.attempt) != (request.request_id, attempt):
        raise StageError(
            f"C4 verdict is for {verdict.request_id} attempt {verdict.attempt}, "
            f"expected {request.request_id} attempt {attempt}"
        )
    return {"trace": trace, "verdicts": (*state.get("verdicts", ()), verdict)}


def after_assure(state: PipelineState) -> Literal["regenerate", "deliver"]:
    verdicts = state.get("verdicts", ())
    if (
        not state.get("error")
        and verdicts
        and verdicts[-1].decision == "REGENERATE"
        and state["attempt"] < MAX_ATTEMPTS
    ):
        return "regenerate"
    return "deliver"


def regenerate(state: PipelineState, deps: Deps) -> Update:
    """Start attempt 2 from the stage C4 named.

    The failed-claim feedback stays in `verdicts` for the LLM stages to use once they exist;
    the deterministic stages have nothing to change, so they simply run again.
    """
    return {"attempt": state["attempt"] + 1, "rerun_from": state["verdicts"][-1].rerun_from}


def rerun_from(state: PipelineState) -> RerunStage:
    return state["rerun_from"]


def deliver(state: PipelineState, deps: Deps) -> Update:
    request = state["request"]
    verdicts = state.get("verdicts", ())
    confidence = state.get("confidence")
    conflicts = state.get("conflicts", ())
    reasons: list[str] = []

    if error := state.get("error"):
        status = "failed"
        reasons.append(error)
    else:
        last = verdicts[-1]
        if last.decision == "HUMAN_REVIEW":
            reasons += ["C4 asked for human review", *last.reasons]
        elif last.decision == "REGENERATE":
            reasons += ["C4 still found problems after regenerating once", *last.reasons]
        if confidence is not None and confidence.score < deps.review_threshold:
            reasons.append(
                f"C1 confidence {confidence.score} is below the review threshold "
                f"{deps.review_threshold}"
            )
        reasons += [f"seed rules and C3 disagree on {c}" for c in conflicts]
        status = "needs_review" if reasons else "delivered"

    return {
        "result": OrchestratorResult(
            request_id=request.request_id,
            status=status,
            attempt=state["attempt"],
            calc=None if status == "failed" else state.get("calc"),
            explanation=None if status == "failed" else state.get("explanation"),
            confidence=confidence,
            verdicts=verdicts,
            review_reasons=tuple(reasons),
            conflicts=conflicts,
            warnings=tuple(dict.fromkeys(state.get("warnings", ()))),
            stages_run=(*state.get("stages_run", ()), "deliver"),
        )
    }


def _profile_value(
    typed: Decimal | None, field_name: str, ocr: tuple[OcrResult, ...]
) -> ProfileValue | None:
    if typed is not None:
        return ProfileValue(value=typed, source="user", confidence=ONE)
    for doc in ocr:
        for f in doc.fields:
            if f.name == field_name:
                return ProfileValue(
                    value=f.value,
                    source="ocr",
                    confidence=f.confidence,
                    document_id=doc.document_id,
                )
    return None


def _refs(rules: RuleSet):
    return (rules.personal_relief.ref, rules.tax_bands.ref)
