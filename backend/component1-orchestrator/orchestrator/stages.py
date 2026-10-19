"""The pipeline stages, as plain functions: `stage(state, deps) -> dict of state updates`.

No framework imports here. `orchestrator.graph` wires these into LangGraph; they can be
unit-tested, or driven by a plain loop, without it. A stage that cannot continue raises
StageError; the runner records it and goes straight to `deliver`.

    extract -> normalise -> retrieve -> calculate -> explain -> score -> assure
    assure: PASS / HUMAN_REVIEW -> deliver
            REGENERATE (attempt 1) -> regenerate -> rerun from the stage C4 names
            REGENERATE (attempt 2) -> deliver, needs review

The INTERPRET stage (LLM classification of ambiguous items) is not in v1: employment income
and APIT need no interpretation. The only LLM is the optional plain-language rewrite in
EXPLAIN (`Deps.explainer_llm`), checked by the explanation agent before it is shown.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from j26_contracts.retrieval import RetrievalRequest

from explanation_agent.explainer import TextLLM
from explanation_agent.explainer import explain as explain_steps
from orchestrator.clients import (
    AssuranceClient,
    ComponentUnavailable,
    OcrClient,
    RetrievalClient,
)
from orchestrator.confidence import REVIEW_THRESHOLD, c1_confidence
from orchestrator.models import (
    MAX_ATTEMPTS,
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
    explainer_llm: TextLLM | None = None  # None: the deterministic template is shown


Update = dict  # the state keys a stage sets


def extract(state: PipelineState, deps: Deps) -> Update:
    request = state["request"]
    try:
        return {
            "ocr_results": tuple(deps.ocr.extract(request.request_id, d) for d in request.documents)
        }
    except ComponentUnavailable as e:
        raise StageError(str(e)) from e


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
    retrieval_request = RetrievalRequest(
        schema_version="1.0.0",
        request_id=request.request_id,
        caller="c1-orchestrator",
        query_text=RETRIEVAL_QUERY,
        as_of_date=as_of,
        income_types=["employment"],
        taxpayer_category="resident_individual",
        rule_types=["relief", "tax_slab"],
    )
    try:
        output = deps.retrieval.retrieve(retrieval_request)
    except ComponentUnavailable as e:
        raise StageError(str(e)) from e
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
    """Template text from the engine's steps, reworded by the LLM if one is configured.

    The rewrite is shown only if it keeps every number and citation (explanation_agent's
    post-check); otherwise the template is. Citations are limited to rules C3 returned. The
    prompt holds only engine figures and step descriptions, no personal data.
    """
    retrieved = {r.rule_id for r in state["retrieval"].results}
    rules_used = {ref.rule_id: ref.citation for ref in _refs(state["rules"])}
    rules_used = {k: v for k, v in rules_used.items() if k in retrieved}
    try:
        outcome = explain_steps(
            state["calc"].steps,
            rules_used,
            deps.explainer_llm,
            year_of_assessment=state["request"].year_of_assessment,
        )
    except ValueError as e:  # a step cites a rule C3 did not return (rule 3)
        raise StageError(str(e)) from e
    return {
        "explanation": outcome.explanation,
        "explain_warnings": outcome.warnings,
        "llm_calls": () if outcome.llm_call is None else (outcome.llm_call,),
    }


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
        explanation=state["explanation"],
        llm_calls=state.get("llm_calls", ()),
        confidence=state["confidence"],
        conflicts=state.get("conflicts", ()),
        warnings=_warnings(state),
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
            rules=None if status == "failed" else state.get("rules"),
            explanation=None if status == "failed" else state.get("explanation"),
            confidence=confidence,
            verdicts=verdicts,
            review_reasons=tuple(reasons),
            conflicts=conflicts,
            warnings=_warnings(state),
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


def _warnings(state: PipelineState) -> tuple[str, ...]:
    """Pipeline warnings plus the latest explain run's, without repeats."""
    return tuple(dict.fromkeys((*state.get("warnings", ()), *state.get("explain_warnings", ()))))


def _refs(rules: RuleSet):
    return (rules.personal_relief.ref, rules.tax_bands.ref)
