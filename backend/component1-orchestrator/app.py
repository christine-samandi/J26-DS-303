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

from orchestrator.clients import (  # noqa: E402
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


# ---------------- UI: input ----------------
st.set_page_config(page_title="J26-DS-303 Tax Assistant", layout="wide")
st.title("Explainable Agentic AI Tax Assistant")
st.caption("C1 demo · YA 2025/26 · the engine does the math, the LLM only rephrases")

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

    st.header("Options")
    use_gemini = st.toggle("Gemini rewrite of explanation", value=False)
    force_regen = st.toggle("C4 mock: force one REGENERATE", value=False)
    go = st.button("Calculate", type="primary", width="stretch")

if not go:
    st.info("Pick an input on the left and press **Calculate**.")
    st.stop()

# ---------------- run ----------------
llm = None
if use_gemini:
    env = load_env()
    env["C1_EXPLAINER_LLM"] = "gemini"
    try:
        llm = explainer_llm_from_env(env)
    except ValueError as e:
        st.error(str(e))
        st.stop()

deps = Deps(  # built fresh each run: the assurance mock consumes its script
    ocr=MockOcrClient(),
    retrieval=MockRetrievalClient(),
    assurance=MockAssuranceClient(["REGENERATE"] if force_regen else []),
    explainer_llm=llm,
)
request = TaxRequest(
    request_id=f"req-{random.randint(1, 9999):04d}",
    year_of_assessment="2025/26",
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

exp = result.explanation
if exp:
    st.subheader("Explanation")
    if exp.hallucination_flag or exp.used_fallback:
        st.badge("Template · Gemini rewrite rejected by post-check", color="red")
    elif llm:
        st.badge("Gemini rewrite · verified by post-check", color="green")
    else:
        st.badge("Template (deterministic)", color="blue")
    for line in exp.lines:
        st.markdown(f"- {line}")

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
