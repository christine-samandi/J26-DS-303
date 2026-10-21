"""The pipeline stages, as plain functions: `stage(state, deps) -> dict of state updates`.

No framework imports here. `orchestrator.graph` wires these into LangGraph; they can be
unit-tested, or driven by a plain loop, without it. A stage that cannot continue raises
StageError; the runner records it and goes straight to `deliver`.

    extract -> normalise -> retrieve [-> discover_reliefs] -> calculate -> explain -> score
            -> assure        (discover_reliefs only when an invoice/receipt was uploaded)
    assure: PASS / HUMAN_REVIEW -> deliver
            REGENERATE (attempt 1) -> regenerate -> rerun from the stage C4 names
            REGENERATE (attempt 2) -> deliver, needs review
            C4 unreachable          -> deliver, marked as not checked by C4

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
    MockAssuranceClient,
    OcrClient,
    RetrievalClient,
)
from orchestrator.confidence import REVIEW_THRESHOLD, c1_confidence
from orchestrator.models import (
    MAX_ATTEMPTS,
    DocumentInput,
    DocumentSummary,
    OcrResult,
    OrchestratorResult,
    PipelineState,
    PipelineTrace,
    ProfileValue,
    ReliefFinding,
    ReliefQuestion,
    RerunStage,
    TaxpayerProfile,
)
from orchestrator.reliefs import CONDITION_QUESTIONS, assess, combine, relief_query
from tax_calculation_agent.engine import (
    CalculationInputError,
    ReliefClaim,
    calculate,
    remaining_claimable,
)
from tax_calculation_agent.ruleset import (
    RELIEF_KEY,
    ReliefRule,
    RuleSet,
    RuleSetError,
    load_seed_reliefs,
    load_seed_rules,
    mid_year_changes,
    relief_conflicts,
    relief_from_retrieval,
    rule_conflicts,
    rules_from_retrieval,
    ya_period,
)

ZERO = Decimal(0)
ONE = Decimal(1)
RETRIEVAL_QUERY = (
    "Personal relief and income tax rates for a resident individual's employment income"
)
RELIEF_DOCUMENTS = ("invoice", "receipt")  # checked for reliefs; other non-payslips are listed


class StageError(Exception):
    """The request cannot be completed. The message is shown to the user, so no PII."""


@dataclass
class Deps:
    ocr: OcrClient
    retrieval: RetrievalClient
    assurance: AssuranceClient
    seed_rules: Callable[[str], RuleSet] = load_seed_rules
    seed_reliefs: Callable[[str], tuple[ReliefRule, ...]] = load_seed_reliefs
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
    """Sort the documents by what C1 does with them, then map typed input and payslip OCR
    fields to a TaxpayerProfile. Typed values win.

    payslip -> income and APIT; invoice/receipt -> relief candidates (discover_reliefs);
    anything else (bank statement, unknown) -> listed as not analysed yet, otherwise ignored.
    """
    request = state["request"]
    documents, payslips, candidates = [], [], []
    for ref, doc in zip(request.documents, state.get("ocr_results", ()), strict=True):
        if doc.document_type.startswith("payslip"):
            payslips.append(doc)
            documents.append(
                DocumentSummary(document_id=ref, document_type=doc.document_type, role="payslip")
            )
        elif doc.document_type in RELIEF_DOCUMENTS:
            candidates.append((ref, doc))
            documents.append(
                DocumentSummary(
                    document_id=ref,
                    document_type=doc.document_type,
                    role="relief_candidate",
                    note="checked for reliefs",
                )
            )
        else:
            documents.append(
                DocumentSummary(
                    document_id=ref,
                    document_type=doc.document_type,
                    role="not_analysed",
                    note=f"not analysed yet: C1 does not use {doc.document_type} documents",
                )
            )
    if len(payslips) > 1:
        raise StageError("more than one payslip: aggregating payslips is not supported yet")
    ocr = tuple(payslips)
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
        "documents": tuple(documents),
        "relief_candidates": tuple(candidates),
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


def has_relief_candidates(state: PipelineState) -> Literal["discover_reliefs", "calculate"]:
    return "discover_reliefs" if state.get("relief_candidates") else "calculate"


def discover_reliefs(state: PipelineState, deps: Deps) -> Update:
    """For each invoice/receipt, ask C3 which reliefs it names and decide each one.

    The query is the issuer C2 read plus the user's "What was this payment for?"; whatever
    reliefs C3 returns are assessed the same way (orchestrator.reliefs). Nothing is assumed:
    a missing answer, amount or date leaves the relief out and asks for review. One
    document backs at most one relief, and a cap counts once per YA across documents.
    """
    request = state["request"]
    ya = request.year_of_assessment
    ya_start, ya_end = ya_period(ya)
    inputs = {d.document: d for d in request.document_inputs}
    income = state["profile"].employment_income.value
    warnings = list(state.get("warnings", ()))
    conflicts = list(state.get("conflicts", ()))
    try:
        seed = deps.seed_reliefs(ya)
    except RuleSetError:
        seed = ()
        warnings.append(f"no seed reliefs for YA {ya}, so C3's reliefs were not cross-checked")

    findings: list[ReliefFinding] = []
    claims: dict[str, list[ReliefClaim]] = {}
    retrievals = []
    for ref, doc in state["relief_candidates"]:
        user = inputs.get(ref, DocumentInput(document=ref))
        kind = doc.document_type
        query = relief_query(doc.issuer, user.payment_for)
        if query is None:
            findings.append(
                ReliefFinding(
                    document_id=ref,
                    status="needs_review",
                    reasons=(
                        f"the issuer of this {kind} could not be read: "
                        "say what the payment was for",
                    ),
                )
            )
            continue
        try:
            output = deps.retrieval.retrieve(
                RetrievalRequest(
                    schema_version="1.0.0",
                    request_id=request.request_id,
                    caller="c1-orchestrator",
                    query_text=query,
                    as_of_date=ya_start,
                    taxpayer_category="resident_individual",
                    rule_types=["relief"],
                )
            )
        except ComponentUnavailable as e:
            findings.append(
                ReliefFinding(
                    document_id=ref,
                    status="needs_review",
                    query=query,
                    reasons=(f"reliefs could not be checked: {e}",),
                )
            )
            continue
        if output.request_id != request.request_id:
            raise StageError(f"C3 answered request {output.request_id}, not {request.request_id}")
        retrievals.append(output)
        found = [
            r
            for r in output.results
            if r.rule_type == "relief"
            and r.rule_key != RELIEF_KEY
            and r.validity.in_force_on_as_of_date
        ]
        if not found:
            if user.payment_for:
                findings.append(
                    ReliefFinding(
                        document_id=ref,
                        status="no_relief",
                        query=query,
                        reasons=("C3 found no relief for this payment",),
                    )
                )
            else:
                findings.append(
                    ReliefFinding(
                        document_id=ref,
                        status="needs_review",
                        query=query,
                        reasons=(
                            "C3 found no relief for the issuer: say what the payment was for",
                        ),
                    )
                )
            continue

        doc_findings = []
        for r in found:
            base = ReliefFinding(
                document_id=ref,
                status="needs_review",
                query=query,
                rule_id=r.rule_id,
                rule_key=r.rule_key,
                title=r.title,
                citation=r.source.citation,
                rule_text=r.text,
                amount_paid=user.amount_paid,
                retrieval_confidence=Decimal(str(r.confidence.score)),
            )
            try:
                rule = relief_from_retrieval(r)
            except RuleSetError as e:
                doc_findings.append(
                    (
                        base.model_copy(update={"status": "not_supported", "reasons": (str(e),)}),
                        None,
                    )
                )
                continue
            conflicts += relief_conflicts(seed, rule)
            a = assess(
                rule,
                document_type=kind,
                document_ref=ref,
                amount_paid=user.amount_paid,
                document_date=user.document_date,
                answers=user.answers,
                ya_start=ya_start,
                ya_end=ya_end,
                employment_income=income,
            )
            questions = tuple(
                ReliefQuestion(
                    question_id=q.question_id, text=q.text, answer=a.answers[q.question_id]
                )
                for q in CONDITION_QUESTIONS.get(rule.rule_key, ())
            )
            finding = base.model_copy(
                update={
                    "status": a.status,
                    "reasons": a.reasons,
                    "questions": questions,
                    "remaining_claimable": a.remaining_claimable,
                }
            )
            doc_findings.append((finding, a.claim))

        applied = [(f, c) for f, c in doc_findings if c is not None]
        if len(applied) > 1:  # one payment cannot back two reliefs
            reason = "this document matches several reliefs: say what the payment was for"
            doc_findings = [
                (
                    f.model_copy(
                        update={
                            "status": "needs_review",
                            "reasons": (reason,),
                            "remaining_claimable": None,
                        }
                    ),
                    None,
                )
                if c is not None
                else (f, c)
                for f, c in doc_findings
            ]
        for f, c in doc_findings:
            findings.append(f)
            if c is not None:
                claims.setdefault(c.rule.rule_key, []).append(c)

    combined = {key: combine(group) for key, group in claims.items()}
    findings = [_with_combined(f, combined) for f in findings]
    return {
        "relief_claims": tuple(combined.values()),
        "relief_findings": tuple(findings),
        "relief_retrievals": tuple(retrievals),
        "conflicts": tuple(dict.fromkeys(conflicts)),
        "warnings": tuple(warnings),
    }


def _with_combined(finding: ReliefFinding, combined: dict[str, ReliefClaim]) -> ReliefFinding:
    """An applied relief claimed on several documents: show the combined remaining amount."""
    claim = combined.get(finding.rule_key or "")
    if finding.status != "applied" or claim is None or len(claim.document_ids) == 1:
        return finding
    others = len(claim.document_ids) - 1
    return finding.model_copy(
        update={
            "remaining_claimable": remaining_claimable(claim),
            "reasons": (
                f"combined with {others} other document{'s' if others > 1 else ''} for "
                "this relief: the yearly limit applies once",
            ),
        }
    )


def calculate_tax(state: PipelineState, deps: Deps) -> Update:
    profile = state["profile"]
    try:
        calc = calculate(
            profile.employment_income.value,
            profile.apit_paid.value,
            state["rules"],
            state.get("relief_claims", ()),
        )
    except CalculationInputError as e:
        raise StageError(str(e)) from e
    return {"calc": calc}


def explain(state: PipelineState, deps: Deps) -> Update:
    """Template text from the engine's steps, reworded by the LLM if one is configured.

    The rewrite is shown only if it keeps every number and citation (explanation_agent's
    post-check); otherwise the template is. Citations are limited to rules C3 returned. The
    prompt holds only engine figures and step descriptions, and is PII-redacted before it is
    sent. No names are passed: C2's ocr_output v0.1 has no employee-name field.
    """
    retrieved = {r.rule_id for r in state["retrieval"].results}
    retrieved |= {r.rule_id for out in state.get("relief_retrievals", ()) for r in out.results}
    refs = (*_refs(state["rules"]), *(c.rule.ref for c in state.get("relief_claims", ())))
    rules_used = {ref.rule_id: ref.citation for ref in refs}
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
        "pii_redacted": outcome.pii_redacted,
    }


def score(state: PipelineState, deps: Deps) -> Update:
    profile = state["profile"]
    rules = state["rules"]
    used = {ref.rule_id for ref in _refs(rules)}
    retrieval = [
        Decimal(str(r.confidence.score)) for r in state["retrieval"].results if r.rule_id in used
    ]
    retrieval += [
        f.retrieval_confidence
        for f in state.get("relief_findings", ())
        if f.status == "applied" and f.retrieval_confidence is not None
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
    try:
        verdict = deps.assurance.assure(trace)
    except ComponentUnavailable as e:
        # Not fatal: the figures are the engine's either way. deliver says nothing checked them.
        return {"trace": trace, "assurance_error": str(e)}
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

    assurance_error = state.get("assurance_error")
    if assurance_error:
        assurance = "unavailable"
        note = f"Not checked by C4: {assurance_error}. Status is from C1's own checks only."
    elif isinstance(deps.assurance, MockAssuranceClient):
        assurance, note = "mock", "C4 is mocked: its verdicts are scripted, not a real check."
    else:
        assurance, note = "live", "Checked by C4."

    if error := state.get("error"):
        status = "failed"
        reasons.append(error)
    else:
        # Only a verdict on the delivered attempt counts. With C4 down there is none: that
        # alone is no reason for review, so the status rests on C1's own rules below. The
        # exception: C4 flagged attempt 1, so an unverified retry is not delivered as clean.
        last = verdicts[-1] if verdicts and verdicts[-1].attempt == state["attempt"] else None
        if last is not None and last.decision == "HUMAN_REVIEW":
            reasons += ["C4 asked for human review", *last.reasons]
        elif last is not None and last.decision == "REGENERATE":
            reasons += ["C4 still found problems after regenerating once", *last.reasons]
        elif assurance_error and verdicts and verdicts[-1].decision == "REGENERATE":
            reasons += [
                f"C4 flagged attempt {verdicts[-1].attempt} and the retry could not be "
                "verified (C4 unavailable)",
                *verdicts[-1].reasons,
            ]
        if confidence is not None and confidence.score < deps.review_threshold:
            reasons.append(
                f"C1 confidence {confidence.score} is below the review threshold "
                f"{deps.review_threshold}"
            )
        reasons += [f"seed rules and C3 disagree on {c}" for c in conflicts]
        reasons += [
            f"{f.title or 'Relief'} ({f.document_id}): {reason}"
            for f in state.get("relief_findings", ())
            if f.status == "needs_review"
            for reason in f.reasons
        ]
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
            documents=state.get("documents", ()),
            relief_findings=state.get("relief_findings", ()),
            pii_redacted=state.get("pii_redacted", 0),
            assurance=assurance,
            assurance_note=note,
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
