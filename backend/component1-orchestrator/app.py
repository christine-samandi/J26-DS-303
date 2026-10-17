"""C1 PP1 demo: input -> orchestrator -> cited, verified answer."""

import os
import random
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explanation_agent.badge import explanation_badge  # noqa: E402
from explanation_agent.demo_tamper import TamperedRewriteLLM, Tampering  # noqa: E402
from orchestrator.clients import (  # noqa: E402
    C3_DEFAULT_URL,
    HttpRetrievalClient,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
)
from orchestrator.graph import run  # noqa: E402
from orchestrator.llm import explainer_llm_from_env  # noqa: E402
from orchestrator.models import TaxRequest  # noqa: E402
from orchestrator.stages import Deps  # noqa: E402

PAYSLIPS = {
    "Clean payslip": "payslip_clean",
    "Noisy payslip (low OCR confidence)": "payslip_noisy_low_confidence",
    "Refund case": "payslip_refund",
    "High income (all 5 bands)": "payslip_high_income",
}
# YAs the live C3 graph can answer (the mock only has 2025/26)
LIVE_YEARS = ["2025/26", "2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"]
STATUS_BOX = {"delivered": st.success, "needs_review": st.warning, "failed": st.error}


def load_env() -> dict[str, str]:
    """os.environ + the gitignored .env (component folder or repo root)."""
    env = dict(os.environ)
    for p in (HERE / ".env", HERE.parents[1] / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip().strip("\"'"))
    return env


def to_decimal(text: str, label: str) -> Decimal | None:
    text = text.replace(",", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        st.error(f"{label} must be a number")
        st.stop()


def rs(x) -> str:
    return f"Rs {x:,.2f}"


def law_value(rule) -> str:
    """A C3 rule's structured value: an amount, or its band rates."""
    v = rule.values
    if v is None:
        return ""
    if v.amount is not None:
        return rs(v.amount.amount)
    return " / ".join(f"{b.rate_percent:g}%" for b in v.bands or [])


def tampered_markdown(t: Tampering) -> list[str]:
    """The rejected rewrite, one line per step, with the changed figure and citation marked."""
    out = []
    for sid, text in t.lines:
        if sid == t.figure_step:
            text = text.replace(t.new_figure, f":red-background[**{t.new_figure}**]", 1)
        if sid == t.citation_to and t.citation_to != t.citation_from:
            text = text.replace(
                f"[{t.citation}]", f":orange[{t.citation} (moved from {t.citation_from})]"
            )
        out.append(f"- `{sid}` {text}")
    return out


# ---------------- UI: input ----------------
st.set_page_config(page_title="J26-DS-303 Tax Assistant", layout="wide")
st.title("Explainable Agentic AI Tax Assistant")
st.caption("C1 demo · the engine does the math, the LLM only rephrases · law from C3")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Type amounts", "Mock payslip"])
    income = apit = None
    docs: tuple[str, ...] = ()
    if mode == "Type amounts":
        income = to_decimal(st.text_input("Employment income (Rs)", "3000000"), "Income")
        apit = to_decimal(st.text_input("APIT already paid (Rs)", "50000"), "APIT")
    else:
        docs = (PAYSLIPS[st.selectbox("Payslip", list(PAYSLIPS))],)

    st.header("Law (C3)")
    env_now = load_env()
    live_c3 = st.toggle(
        "Live C3 GraphRAG",
        value=env_now.get("C1_RETRIEVAL", "mock").strip().lower() == "http",
        help="On: rules come from C3's knowledge graph (POST /retrieve). Off: mock rules.",
    )
    c3_url = env_now.get("C3_API_URL", C3_DEFAULT_URL)
    if live_c3:
        st.caption(f"C3 API: {c3_url}")
        year = st.selectbox("Year of assessment", LIVE_YEARS)
    else:
        year = "2025/26"
        st.caption("Mock rules · YA 2025/26 only")

    st.header("Options")
    use_gemini = st.toggle("Gemini rewrite of explanation", value=False)
    demo_tamper = st.toggle(
        "Demo: simulate tampered rewrite",
        value=False,
        help="Offline: feeds a deliberately wrong rewrite to the real post-check. No Gemini call.",
    )
    force_regen = st.toggle("C4 mock: force one REGENERATE", value=False)
    go = st.button("Calculate", type="primary", width="stretch")

if not go:
    st.info("Pick an input on the left and press **Calculate**.")
    st.stop()

# ---------------- run ----------------
llm = None
if demo_tamper:  # replaces Gemini entirely: no key, no network
    llm = TamperedRewriteLLM()
elif use_gemini:
    env = load_env()
    env["C1_EXPLAINER_LLM"] = "gemini"
    try:
        llm = explainer_llm_from_env(env)
    except ValueError as e:
        st.error(str(e))
        st.stop()

retrieval = HttpRetrievalClient(c3_url) if live_c3 else MockRetrievalClient()
deps = Deps(  # built fresh each run: the assurance mock consumes its script
    ocr=MockOcrClient(),
    retrieval=retrieval,
    assurance=MockAssuranceClient(["REGENERATE"] if force_regen else []),
    explainer_llm=llm,
)
request = TaxRequest(
    request_id=f"req-{random.randint(1, 9999):04d}",
    year_of_assessment=year,
    employment_income=income,
    apit_paid=apit,
    documents=docs,
)

with st.spinner("Running pipeline..."):
    try:
        result = run(request, deps)
    except Exception as e:  # show it on screen, don't crash the demo
        st.exception(e)
        st.stop()

# ---------------- UI: output ----------------
STATUS_BOX.get(result.status, st.info)(
    f"Status: **{result.status}** · attempt {result.attempt} · {request.request_id}"
)

calc = result.calc
if calc:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employment income", rs(calc.employment_income))
    c2.metric("Total tax", rs(calc.total_tax))
    if calc.refund > 0:
        c3.metric("Refund due", rs(calc.refund))
    else:
        c3.metric("Tax payable", rs(calc.tax_payable))
    if result.confidence:
        c4.metric("Confidence", f"{result.confidence.score:.2f}")

    st.subheader("Calculation steps")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Step": s.step_id,
                    "What": s.description,
                    "Operation": s.operation,
                    "Result (Rs)": f"{s.result:,.2f}",
                    "Citations": ", ".join(s.citation_refs),
                    "Source": str(s.source),
                }
                for s in calc.steps
            ]
        ),
        hide_index=True,
        width="stretch",
    )

law = getattr(retrieval, "last", None)
if law is not None:
    st.subheader(f"Law used · from C3 GraphRAG, in force on {law.query.as_of_date}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rule": r.rule_id,
                    "Value": law_value(r),
                    "Valid": f"{r.validity.valid_from} .. {r.validity.valid_to or 'now'}",
                    "Source": r.source.citation,
                    "C3 confidence": f"{r.confidence.score:.2f}",
                }
                for r in law.results
                if r.validity.in_force_on_as_of_date
            ]
        ),
        hide_index=True,
        width="stretch",
    )

exp = result.explanation
if exp:
    st.subheader("Explanation")
    label, color = explanation_badge(exp, "rewrite" if demo_tamper else "Gemini rewrite")
    st.badge(label, color=color)
    if demo_tamper:
        st.caption("Simulated tampered LLM output (demo mode)")
    for line in exp.lines:
        st.markdown(f"- {line}")
    if demo_tamper and llm.last:
        with st.expander("What the AI tried to say", expanded=False):
            st.caption(
                f"Rejected: {llm.last.figure_step} changed from Rs. {llm.last.old_figure} "
                f"to Rs. {llm.last.new_figure}; citation {llm.last.citation} moved from "
                f"{llm.last.citation_from} to {llm.last.citation_to}."
            )
            for line in tampered_markdown(llm.last):
                st.markdown(line)

left, right = st.columns(2)
with left:
    if result.confidence:
        st.subheader("Confidence breakdown")
        st.dataframe(
            pd.DataFrame(
                [{"Term": k, "Value": str(v)} for k, v in result.confidence.model_dump().items()]
            ),
            hide_index=True,
            width="stretch",
        )
with right:
    st.subheader("Checks")
    for r in result.review_reasons:
        st.warning(f"Review: {r}")
    for c in result.conflicts:
        st.error(f"Conflict: {c}")
    for w in result.warnings:
        st.info(f"Warning: {w}")
    if not (result.review_reasons or result.conflicts or result.warnings):
        st.success("No review reasons, conflicts or warnings")
    st.caption("Stages: " + " → ".join(result.stages_run))

with st.expander("Raw result JSON"):
    st.json(result.model_dump(mode="json"))
