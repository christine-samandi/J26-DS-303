"""C1 PP1 demo: input -> orchestrator -> cited, verified answer.

Documents are "doc-1", "doc-2"... everywhere past this page (findings, the C4 trace, C2's
upload): only this page shows the file names, by mapping the ids back (`named`).
Invoices and receipts are a two-step flow: Calculate finds the reliefs and their questions;
the user confirms amount and date, answers, and presses Recalculate.
"""

import hashlib
import os
import random
import re
import sys
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from explanation_agent.badge import explanation_badge  # noqa: E402
from explanation_agent.demo_tamper import TamperedRewriteLLM, Tampering  # noqa: E402
from orchestrator.clients import (  # noqa: E402
    C2_DEFAULT_URL,
    C2_TO_C1_FIELDS,
    C3_DEFAULT_URL,
    C4_DEFAULT_URL,
    ComponentUnavailable,
    HttpAssuranceClient,
    HttpOcrClient,
    HttpRetrievalClient,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
    assurance_client_from_env,
)
from orchestrator.graph import run  # noqa: E402
from orchestrator.law_updates import fetch_notifications, impacts  # noqa: E402
from orchestrator.llm import explainer_llm_from_env  # noqa: E402
from orchestrator.models import DocumentInput, OcrResult, TaxRequest  # noqa: E402
from orchestrator.stages import Deps, doc_id  # noqa: E402
from tax_calculation_agent.ruleset import RuleSetError, ya_period  # noqa: E402

PAYSLIPS = {
    "Clean payslip": "payslip_clean",
    "Noisy payslip (low OCR confidence)": "payslip_noisy_low_confidence",
    "Refund case": "payslip_refund",
    "High income (all 5 bands)": "payslip_high_income",
}
# Offline invoices/receipts (mock C2): checked for reliefs against the mock C3 relief answers
MOCK_RELIEF_DOCS = {
    "Solar panel invoice (Rs 800,000, 15 Aug 2025)": "solar_invoice_2025",
    "Medical receipt (Rs 45,000, 10 Sep 2025): relief withdrawn": "medical_receipt_2025",
    "Receipt, issuer unreadable": "receipt_unreadable_issuer",
    "Bank statement (not analysed)": "bank_statement_2025",
}
RELIEF_BADGE = {  # ReliefStatus -> (label, colour)
    "applied": ("Applied", "green"),
    "not_applied": ("Not applied", "gray"),
    "needs_review": ("Needs your input", "orange"),
    "not_supported": ("Not supported yet", "blue"),
    "no_relief": ("No relief", "gray"),
    "withdrawn": ("Relief withdrawn", "red"),
}
ANSWERS = ["Not answered", "Yes", "No"]
# YAs the live C3 graph can answer, and those C1's mock C3 has a file for
LIVE_YEARS = [f"{y}/{(y + 1) % 100:02d}" for y in range(2026, 2018, -1)]  # 2026/27 .. 2019/20
MOCK_YEARS = ["2025/26", "2024/25"]
# "Compare tax years": before and after Act No. 2 of 2025 (relief and bands from 1 April 2025)
COMPARE_YEARS = ("2024/25", "2025/26")
SAME = "same"
CHANGED = "background-color: rgba(250, 180, 0, 0.25)"  # readable in light and dark theme
# How much of the year one uploaded payslip covers -> months_covered for HttpOcrClient
PAYSLIP_COVERS = {"One month (x 12 for the year)": 1, "Full year (annual summary)": 12}
UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf"]
STATUS_BOX = {"delivered": st.success, "needs_review": st.warning, "failed": st.error}
# C4's FindingResponse fields shown in the Assurance panel (component4-assurance/api/main.py)
C4_FINDING_COLUMNS = [
    "check_name",
    "passed",
    "severity",
    "message",
    "expected_value",
    "actual_value",
]


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


EXCERPT_CHARS = 80
NO_RULE = "—"


def md_escape(text: str) -> str:
    """Law text shown as markdown: no accidental bold, links or LaTeX."""
    return "".join(f"\\{c}" if c in "\\`*_[]<>$#|" else c for c in text)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def rule_refs(rules, reliefs=()) -> dict:
    """rule_id -> RuleRef for the rules a result's steps can cite (reliefs: those applied)."""
    if rules is None:
        return {}
    refs = (rules.personal_relief.ref, rules.tax_bands.ref, *(c.rule.ref for c in reliefs))
    return {ref.rule_id: ref for ref in refs}


def cited_refs(steps, refs: dict) -> list:
    """The rules cited by any step, in first-cited order."""
    ids = dict.fromkeys(r for s in steps for r in s.citation_refs if r in refs)
    return [refs[r] for r in ids]


def step_law(step, refs: dict) -> tuple[str, str]:
    """(Source, Evidence) table cells for one step; NO_RULE if it cites no rule."""
    cited = [refs[r] for r in step.citation_refs if r in refs]
    if not cited:
        return NO_RULE, NO_RULE
    source = "; ".join(r.citation for r in cited)
    evidence = " | ".join(
        excerpt(r.evidence) if r.evidence else "(seed rule, no excerpt)" for r in cited
    )
    return source, evidence


def compare_rows(by_year: dict) -> list[dict]:
    """Two runs side by side (earlier YA first): figures, the Act each cites, and the change."""
    (old_ya, old), (new_ya, new) = by_year.items()
    rows = []

    def row(label: str, a, b, change: str) -> None:
        rows.append({"": label, f"YA {old_ya}": a, f"YA {new_ya}": b, "Change": change})

    def money(result, get):
        return get(result) if result.calc else None

    for label, get in (
        ("Personal relief", lambda r: r.rules.personal_relief.amount),
        ("Taxable income", lambda r: r.calc.taxable_income),
        ("Total tax", lambda r: r.calc.total_tax),
        ("Tax payable", lambda r: r.calc.tax_payable),
        ("Refund", lambda r: r.calc.refund),
    ):
        a, b = money(old, get), money(new, get)
        if a is None or b is None:
            change = NO_RULE
        else:
            change = SAME if a == b else f"{b - a:+,.2f}"
        row(label, NO_RULE if a is None else rs(a), NO_RULE if b is None else rs(b), change)

    for label, rule in (("Relief cites", "personal_relief"), ("Bands cite", "tax_bands")):
        a, b = (getattr(r.rules, rule).ref.citation if r.rules else NO_RULE for r in (old, new))
        row(label, a, b, SAME if a == b else "changed")
    return rows


class RecordingOcr:
    """Wraps the OCR client: each document is read once per file (kept in `cache`, the
    session's), so answering a relief question and recalculating does not run C2 again.
    Also what pre-fills the relief form (issuer, date, total)."""

    def __init__(self, inner, cache: dict, keys: dict[str, str]) -> None:
        self.inner, self.cache, self.keys = inner, cache, keys  # keys: ref -> cache key

    def extract(self, request_id: str, document: str) -> OcrResult:
        key = self.keys[document]
        if key not in self.cache:
            out = self.inner.extract(request_id, document)
            raw = getattr(self.inner, "reads", {}).get(document)
            self.cache[key] = (out, raw, list(getattr(self.inner, "notes", [])))
        return self.cache[key][0]

    def read(self, document: str):
        """(OcrResult, C2's raw answer or None, notes) once extracted, else None."""
        return self.cache.get(self.keys.get(document, ""))


class MixedOcr:
    """Uploaded files go to live C2; picked mock invoices/receipts go to the mock client."""

    def __init__(self, http, mock, mock_refs: set[str]) -> None:
        self.http, self.mock, self.mock_refs = http, mock, mock_refs

    def extract(self, request_id: str, document: str) -> OcrResult:
        client = self.mock if document in self.mock_refs else self.http
        return client.extract(request_id, document)

    def __getattr__(self, name):  # reads, notes, visual, etc. come from live C2
        return getattr(self.http, name)


def named(text: str, names: dict[str, str]) -> str:
    """'... (doc-2) ...' -> '... (solar.png, doc-2) ...': file names for this page only."""
    return re.sub(
        r"\bdoc-\d+\b", lambda m: f"{names[m.group()]}, {m.group()}" if m.group() in names
        else m.group(), text,
    )  # fmt: skip


def relief_inputs(token: str, documents) -> tuple[DocumentInput, ...]:
    """What the user entered in the relief panel, for the invoices/receipts of the last run.
    Amount and date count only when the user ticked that they checked them."""
    out = []
    for d in documents:
        if d.role != "relief_candidate":
            continue
        k = f"{token}:{d.document_id}"
        confirmed = st.session_state.get(f"{k}:confirm", False)
        amount = None
        if confirmed:
            text = (st.session_state.get(f"{k}:amount") or "").replace(",", "").strip()
            try:
                amount = Decimal(text) if text else None
            except InvalidOperation:
                amount = None
        answers = {
            key.rsplit(":", 1)[1]: st.session_state[key].lower()
            for key in st.session_state
            if isinstance(key, str)
            and key.startswith(f"{k}:q:")
            and st.session_state[key] in ("Yes", "No")
        }
        out.append(
            DocumentInput(
                document=d.document_id,
                payment_for=(st.session_state.get(f"{k}:for") or "").strip() or None,
                amount_paid=amount if amount is not None and amount >= 0 else None,
                document_date=st.session_state.get(f"{k}:date") if confirmed else None,
                answers=answers,
            )
        )
    return tuple(out)


def recalculate() -> None:
    st.session_state["recalc"] = True


def law_updates_box(c3_url: str, show_impact: bool) -> None:
    """C3's law-change notifications; warns when the last calculation used a rule that changed."""
    try:
        items = fetch_notifications(c3_url)
    except ComponentUnavailable as e:
        st.sidebar.caption(f"Law updates unavailable: {e}")
        return
    last = st.session_state.get("last_calc")
    if show_impact and last:
        newer = [n for n in items if datetime.fromisoformat(n["created_at"]) > last["at"]]
        hits = impacts(newer, last["rule_ids"], last["as_of"])
        if hits:
            lines = "\n".join(f"- {md_escape(h.summary)}" for h in hits)
            st.warning(
                f"**The law changed after your last calculation** (YA {last['ya']}, made "
                f"{last['at']:%H:%M}):\n{lines}\n\nPress **Calculate** again to use the new law."
            )
    if items:
        with st.sidebar.expander(f"Law updates ({len(items)})"):
            for n in items[:3]:
                st.markdown(f"**{md_escape(n['title'])}** · {n['created_at'][:10]}")
                for c in n.get("changes", [])[:8]:
                    st.markdown(f"- {md_escape(c['summary'])}")


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
    mode = st.radio("Source", ["Type amounts", "Mock payslip", "Upload documents (live C2 OCR)"])
    env_now = load_env()
    income = apit = None
    docs: tuple[str, ...] = ()
    uploads = []
    extra_mock: tuple[str, ...] = ()
    show_boxes = False
    c2_url = env_now.get("C2_API_URL", C2_DEFAULT_URL)
    if mode == "Upload documents (live C2 OCR)":
        uploads = st.file_uploader(
            "Payslip, invoices, receipts (photo, scan or PDF)",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="One payslip at most. Invoices and receipts are checked for reliefs.",
        )
        covers = st.selectbox("The payslip covers", list(PAYSLIP_COVERS))
        st.caption("No payslip? Type the income instead (typed amounts win over the payslip).")
        income = to_decimal(st.text_input("Employment income (Rs), optional", ""), "Income")
        apit = to_decimal(st.text_input("APIT already paid (Rs), optional", ""), "APIT")
        show_boxes = st.toggle("Show where C2 read each value", value=True)
        extra_mock = tuple(
            MOCK_RELIEF_DOCS[p]
            for p in st.multiselect(
                "Add mock invoices / receipts (offline)",
                list(MOCK_RELIEF_DOCS),
                help="Backup for the demo: mock documents added to your uploads.",
            )
        )
        st.caption(f"C2 API: {c2_url}")
    else:
        if mode == "Type amounts":
            income = to_decimal(st.text_input("Employment income (Rs)", "3000000"), "Income")
            apit = to_decimal(st.text_input("APIT already paid (Rs)", "50000"), "APIT")
        else:
            docs = (PAYSLIPS[st.selectbox("Payslip", list(PAYSLIPS))],)
        picked = st.multiselect(
            "Invoices / receipts (mock C2)",
            list(MOCK_RELIEF_DOCS),
            help="Offline documents, checked for reliefs. Pick none for the payslip-only flow.",
        )
        docs += tuple(MOCK_RELIEF_DOCS[p] for p in picked)

    st.header("Law (C3)")
    live_c3 = st.toggle(
        "Live C3 GraphRAG",
        value=env_now.get("C1_RETRIEVAL", "mock").strip().lower() == "http",
        help="On: rules come from C3's knowledge graph (POST /retrieve). Off: mock rules.",
    )
    c3_url = env_now.get("C3_API_URL", C3_DEFAULT_URL)
    if live_c3:
        st.caption(f"C3 API: {c3_url}")
        year = st.selectbox("Year of assessment", LIVE_YEARS)
        st.button("Check for law updates", help="Ask C3 whether the law changed (no calculation)")
    else:
        year = st.selectbox("Year of assessment", MOCK_YEARS)
        st.caption("Mock rules (C1 stand-in for C3)")

    st.header("Assurance (C4)")
    live_c4 = st.toggle(
        "Live C4 assurance",
        value=isinstance(assurance_client_from_env(env_now), HttpAssuranceClient),
        help="On: C4's service checks the result. Off: a mock that PASSes unless told not to.",
    )
    c4_env = {**env_now, "C4_LIVE": "1" if live_c4 else "0"}
    force_regen = False
    if live_c4:
        st.caption(f"C4 API: {c4_env.get('C4_API_URL', C4_DEFAULT_URL)}")
    else:
        force_regen = st.toggle("C4 mock: force one REGENERATE", value=False)

    st.header("Options")
    use_gemini = st.toggle("Gemini rewrite of explanation", value=False)
    demo_tamper = st.toggle(
        "Demo: simulate tampered rewrite",
        value=False,
        help="Offline: feeds a deliberately wrong rewrite to the real post-check. No Gemini call.",
    )
    go = st.button("Calculate", type="primary", width="stretch")

# ---------------- run ----------------
# Calculate starts a run; Recalculate (relief panel) runs again with the user's answers. The
# result is kept in the session, so answering a question redraws the page without a run.
recalc = st.session_state.pop("recalc", False)
if go or recalc:
    st.session_state["view"] = None
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

    cache = st.session_state.setdefault("ocr_cache", {})
    if mode.startswith("Upload"):
        if not uploads and not extra_mock:
            st.error("Upload a payslip, invoice or receipt first")
            st.stop()
        http = HttpOcrClient(c2_url, months_covered=PAYSLIP_COVERS[covers])
        uploaded = tuple(http.add(f.name, f.getvalue()) for f in uploads)
        docs = uploaded + extra_mock
        names = {ref: f.name for ref, f in zip(uploaded, uploads, strict=True)}
        names |= {
            doc_id(n): f"{ref}.json (mock)"
            for n, ref in enumerate(docs, start=1)
            if ref in extra_mock
        }
        keys = {
            ref: f"c2:{PAYSLIP_COVERS[covers]}:{hashlib.sha256(f.getvalue()).hexdigest()}"
            for ref, f in zip(uploaded, uploads, strict=True)
        }
        keys |= {ref: f"mock:{ref}" for ref in extra_mock}
        c2 = MixedOcr(http, MockOcrClient(), set(extra_mock)) if extra_mock else http
    else:
        c2 = MockOcrClient()
        names = {doc_id(n): f"{ref}.json (mock)" for n, ref in enumerate(docs, start=1)}
        keys = {ref: f"mock:{ref}" for ref in docs}
    ocr = RecordingOcr(c2, cache, keys)
    # relief answers are kept per set of documents: a new set starts with an empty form
    token = hashlib.sha256(repr(sorted(keys.values())).encode()).hexdigest()[:12]
    previous = st.session_state.get("last_documents")
    document_inputs = relief_inputs(token, previous) if previous is not None else ()

    retrieval = HttpRetrievalClient(c3_url) if live_c3 else MockRetrievalClient()
    assurance = assurance_client_from_env(c4_env)
    if force_regen:
        assurance = MockAssuranceClient(["REGENERATE"])
    deps = Deps(  # built fresh each run: the assurance mock consumes its script
        ocr=ocr,
        retrieval=retrieval,
        assurance=assurance,
        explainer_llm=llm,
    )
    request = TaxRequest(
        request_id=f"req-{random.randint(1, 9999):04d}",
        year_of_assessment=year,
        employment_income=income,
        apit_paid=apit,
        documents=docs,
        document_inputs=document_inputs,
    )

    with st.spinner("Running pipeline..."):
        try:
            result = run(request, deps)
        except Exception as e:  # show it on screen, don't crash the demo
            st.exception(e)
            st.stop()
    st.session_state["last_documents"] = result.documents
    st.session_state["view"] = dict(
        result=result, request=request, ocr=ocr, c2=c2, names=names, token=token, docs=docs,
        retrieval=retrieval, assurance=assurance, llm=llm, demo_tamper=demo_tamper,
        show_boxes=show_boxes, live_c3=live_c3, c3_url=c3_url,
    )  # fmt: skip

if live_c3:
    law_updates_box(c3_url, show_impact=not (go or recalc))

view = st.session_state.get("view")
if view is None:
    st.info("Pick an input on the left and press **Calculate**.")
    st.stop()
# the run being shown: the sidebar may have changed since, and is used from the next run
result, request, ocr, c2, names, token, docs = (
    view[k] for k in ("result", "request", "ocr", "c2", "names", "token", "docs")
)
retrieval, assurance, llm, demo_tamper = (
    view[k] for k in ("retrieval", "assurance", "llm", "demo_tamper")
)
show_boxes, live_c3, c3_url = (view[k] for k in ("show_boxes", "live_c3", "c3_url"))

if (go or recalc) and live_c3 and result.rules is not None:  # per run, for law-change check
    try:
        st.session_state["last_calc"] = {
            "at": datetime.now(UTC),
            "ya": year,
            "as_of": ya_period(year)[0],
            "rule_ids": set(rule_refs(result.rules, result.reliefs)),  # incl. reliefs
        }  # fmt: skip
    except RuleSetError:
        pass

# ---------------- UI: output ----------------
STATUS_BOX.get(result.status, st.info)(
    f"Status: **{result.status}** · attempt {result.attempt} · {request.request_id}"
)

calc = result.calc
refs = rule_refs(result.rules, result.reliefs)
if calc:
    m1, m2, m3, m4 = st.columns(4)  # not c2: that name is the C2 client used below
    m1.metric("Employment income", rs(calc.employment_income))
    m2.metric("Total tax", rs(calc.total_tax))
    if calc.refund > 0:
        m3.metric("Refund due", rs(calc.refund))
    else:
        m3.metric("Tax payable", rs(calc.tax_payable))
    if result.confidence:
        m4.metric("Confidence", f"{result.confidence.score:.2f}")

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
                    "Source": source,
                    "Evidence": evidence,
                    "Computed by": str(s.source),
                }
                for s in calc.steps
                for source, evidence in [step_law(s, refs)]
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Source": st.column_config.TextColumn(help="The provision each cited rule comes from"),
            "Evidence": st.column_config.TextColumn(
                help=f"First {EXCERPT_CHARS} characters of the source text; full text below"
            ),
        },
    )
    with st.expander("Evidence in full"):
        for ref in cited_refs(calc.steps, refs):
            st.markdown(f"**{md_escape(ref.citation)}** · `{ref.rule_id}`")
            if ref.evidence:
                st.text(ref.evidence)
                st.caption("Chunks: " + ", ".join(ref.evidence_chunk_ids))
            else:
                st.caption("Seed rule: no source excerpt")

    with st.expander("Compare tax years"):
        by_year = {}
        for ya in COMPARE_YEARS:
            # same figures as above, no Gemini, C4 mock passes: only the law differs
            retrieval_ya = HttpRetrievalClient(c3_url) if live_c3 else MockRetrievalClient()
            deps_ya = Deps(MockOcrClient(), retrieval_ya, MockAssuranceClient())
            by_year[ya] = run(
                TaxRequest(
                    request_id=f"req-{random.randint(1, 9999):04d}",
                    year_of_assessment=ya,
                    employment_income=calc.employment_income,
                    apit_paid=calc.apit_paid,
                ),
                deps_ya,
            )
        st.caption(
            f"Same income ({rs(calc.employment_income)}) and APIT ({rs(calc.apit_paid)}), "
            f"each year under the law in force on 1 April of that year "
            f"({'live C3' if live_c3 else 'C1 mock C3'}). Highlighted rows changed."
        )
        st.dataframe(
            pd.DataFrame(compare_rows(by_year)).style.apply(
                lambda row: [CHANGED if row["Change"] != SAME else "" for _ in row], axis=1
            ),
            hide_index=True,
            width="stretch",
        )
        for ya, r in by_year.items():
            if r.calc is None:
                st.error(f"YA {ya}: {'; '.join(r.review_reasons) or r.status}")

if result.documents:
    st.subheader("Documents and reliefs")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "File": names.get(d.document_id, ""),
                    "Id": d.document_id,
                    "Type": d.document_type,
                    "Used as": d.role.replace("_", " "),
                    "Note": d.note,
                }
                for d in result.documents
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Ids, not file names, are what C1 passes on (C2, C3, C4, messages). "
        "A relief is applied only on the amount and date you confirm and with every "
        "question answered Yes. Change anything, then press **Recalculate**."
    )
    for d in (x for x in result.documents if x.role == "relief_candidate"):
        found = [f for f in result.relief_findings if f.document_id == d.document_id]
        read = ocr.read(docs[result.documents.index(d)])  # doc-<n> is the n-th of docs
        pre = read[0] if read else None
        k = f"{token}:{d.document_id}"
        with st.container(border=True):
            st.markdown(f"**{md_escape(names.get(d.document_id, d.document_id))}** · "
                        f"`{d.document_id}` · {d.document_type}")  # fmt: skip
            if pre is not None and pre.issuer:
                st.caption(f"Issuer read by C2: {pre.issuer}")
            for f in found:
                label, color = RELIEF_BADGE[f.status]
                st.badge(f"{f.title or 'Relief'} · {label}" if f.title else label, color=color)
                for reason in f.reasons:
                    st.markdown(f"- {md_escape(named(reason, names))}")
                if f.status == "applied" and f.remaining_claimable:
                    st.caption(
                        f"{rs(f.remaining_claimable)} paid above this year's limit: shown only, "
                        "not applied to any other year."
                    )
                if f.citation:
                    st.caption(f"Law: {f.citation}" + (f" · `{f.rule_id}`" if f.rule_id else ""))
                if f.rule_text:
                    with st.expander("C3's text, conditions included"):
                        st.text(f.rule_text)
            left_r, right_r = st.columns(2)
            left_r.text_input(
                "What was this payment for?",
                key=f"{k}:for",
                help="Sent to C3 with the issuer to find the relief (contact details removed).",
            )
            right_r.text_input(
                "Amount paid (Rs)",
                value=f"{pre.total_amount:.2f}" if pre is not None and pre.total_amount else "",
                key=f"{k}:amount",
            )
            left_r.date_input(
                f"Date on the {d.document_type}",
                value=pre.document_date if pre is not None else None,
                min_value=date(2018, 4, 1),
                key=f"{k}:date",
            )
            right_r.checkbox(
                f"I checked the amount and date against the {d.document_type}",
                key=f"{k}:confirm",
                help="C2's reading only pre-fills these; it is not used until you confirm.",
            )
            questions = {q.question_id: q for f in found for q in f.questions}
            for q in questions.values():
                st.radio(
                    q.text,
                    ANSWERS,
                    index=ANSWERS.index(q.answer.capitalize()) if q.answer else 0,
                    horizontal=True,
                    key=f"{k}:q:{q.question_id}",
                )
    st.button("Recalculate with these answers", type="primary", on_click=recalculate)

for ref in docs:
    read = ocr.read(ref)
    if read is None or read[1] is None:  # mock C2, or C2 did not answer
        continue
    out, raw, notes = read
    n = docs.index(ref) + 1
    st.subheader(
        f"Document read by C2 OCR · {names.get(doc_id(n), ref)} ({doc_id(n)}, "
        f"{raw.get('document_type')})"
    )
    left_doc, right_doc = st.columns([3, 2]) if show_boxes else (st.container(), None)
    with left_doc:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Field": f["field_name"],
                        "Value": f"{f['value']:,.2f}"
                        if isinstance(f["value"], (int, float))
                        else str(f["value"]),
                        "Confidence": f"{f['confidence']:.3f}",
                        "Needs review": "yes" if f["needs_review"] else "",
                        "Used by C1 as": C2_TO_C1_FIELDS.get(f["field_name"], ""),
                    }
                    for f in raw["extracted_fields"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"C2 overall confidence {raw.get('overall_confidence')} · "
            f"needs review: {'yes' if raw.get('needs_review') else 'no'}"
        )
        for note in notes:
            st.info(f"Annualised: {note}")
    if right_doc is not None:
        with right_doc:
            boxes = st.session_state.setdefault("c2_boxes", {})
            key = ocr.keys[ref]
            try:
                if key not in boxes:
                    boxes[key] = c2.visual(ref)
                st.image(boxes[key], caption="Green = trusted, orange = needs review")
            except ComponentUnavailable as e:
                st.caption(f"Boxes not available: {e}")

# C3's first answer is the rules (personal relief, bands); the rest are relief queries
law = next(iter(getattr(retrieval, "answers", [])), None)
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
    if exp.rewrite_status != "not_requested":  # only a prompt that was sent is redacted
        n = result.pii_redacted
        st.caption(f"PII redacted: {n} item{'' if n == 1 else 's'} before the LLM call")
    if demo_tamper:
        st.caption("Simulated tampered LLM output (demo mode)")
    for line in exp.lines:
        st.markdown(f"- {line}")
    if calc:
        st.caption("Sources")
        for ref in cited_refs(calc.steps, refs):
            quote = f' — "{md_escape(excerpt(ref.evidence))}"' if ref.evidence else ""
            st.markdown(f"- {md_escape(ref.citation)}{quote}")
    if demo_tamper and llm.last:
        with st.expander("What the AI tried to say", expanded=False):
            st.caption(
                f"Rejected: {llm.last.figure_step} changed from Rs. {llm.last.old_figure} "
                f"to Rs. {llm.last.new_figure}; citation {llm.last.citation} moved from "
                f"{llm.last.citation_from} to {llm.last.citation_to}."
            )
            for line in tampered_markdown(llm.last):
                st.markdown(line)

if result.status != "failed":
    st.subheader("Assurance")
    if result.assurance == "unavailable":
        st.warning(f"**Assurance unavailable** · {result.assurance_note}")
    if result.verdicts:
        last = result.verdicts[-1]
        a1, a2, a3 = st.columns(3)
        a1.metric("C4 decision", last.decision)
        a2.metric("Attempt", last.attempt)
        a3.metric("Source", result.assurance)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Attempt": v.attempt, "Decision": v.decision, "Reasons": "; ".join(v.reasons)}
                    for v in result.verdicts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if result.assurance != "unavailable":
        st.caption(result.assurance_note)
    checked = getattr(assurance, "last", None)
    if checked is not None:  # the last answer C4 gave
        findings = checked["response"].get("findings") or []
        st.caption(
            f"Findings from C4 `{checked['endpoint']}` on attempt "
            f"{checked['request']['regeneration_attempts'] + 1}"
        )
        if findings:
            st.dataframe(
                pd.DataFrame(findings).reindex(columns=C4_FINDING_COLUMNS),
                hide_index=True,
                width="stretch",
            )

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
        st.warning(f"Review: {named(r, names)}")
    for c in result.conflicts:
        st.error(f"Conflict: {c}")
    for w in result.warnings:
        st.info(f"Warning: {w}")
    if not (result.review_reasons or result.conflicts or result.warnings):
        st.success("No review reasons, conflicts or warnings")
    st.caption("Stages: " + " → ".join(result.stages_run))

with st.expander("Raw result JSON"):
    st.json(result.model_dump(mode="json"))
