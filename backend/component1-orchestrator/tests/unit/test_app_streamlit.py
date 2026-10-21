"""The Streamlit demo (app.py) end to end. Needs the demo extra; skipped without it (CI)."""

import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import orchestrator.llm  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"


class ClientError(Exception):
    """Stands in for google.genai.errors.ClientError (429 quota exhausted)."""


class QuotaExhaustedLLM:
    model = "fake-gemini"
    temperature = 0.0

    def generate(self, prompt: str) -> tuple[str, str]:
        raise ClientError("429 RESOURCE_EXHAUSTED")


class EchoLLM:
    """An honest rewrite: the step lines it was sent, unchanged."""

    model = "fake-gemini"
    temperature = 0.0

    def generate(self, prompt: str) -> tuple[str, str]:
        steps = json.loads(prompt.split("Steps:\n", 1)[1].rsplit("\n\nAnswer with", 1)[0])
        return json.dumps({"lines": steps}), "fake-gemini-001"


def badges(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if "-badge[" in m.value]


def test_app_shows_gray_badge_when_gemini_call_fails(monkeypatch):
    """The bug: Gemini on but failing showed the green "verified" badge over template text."""
    monkeypatch.setattr(orchestrator.llm, "explainer_llm_from_env", lambda env: QuotaExhaustedLLM())
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(t for t in at.toggle if t.label == "Gemini rewrite of explanation").set_value(True)
    next(b for b in at.button if b.label == "Calculate").click()
    at.run()

    assert not at.exception
    assert badges(at) == [":gray-badge[Template · Gemini unavailable]"]
    assert any(
        "explanation LLM call failed (ClientError); template text shown" in i.value for i in at.info
    )


def test_steps_table_shows_each_steps_law_and_evidence():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(b for b in at.button if b.label == "Calculate").click()
    at.run()

    assert not at.exception
    steps = at.dataframe[0].value.set_index("Step")
    assert {"Source", "Evidence", "Computed by"} <= set(steps.columns)
    assert "s.5, inserting item (v) in Fifth Schedule para 2(a)" in steps.loc["S1_RELIEF", "Source"]
    assert "s.3(1)(b), inserting First Schedule para 1(1D)" in steps.loc["S3_BAND_1", "Source"]
    assert "personal relief of Rs. 1,800,000" in steps.loc["S1_RELIEF", "Evidence"]
    assert all(len(e) <= 80 for e in steps["Evidence"])
    for sid in ("S5_APIT_CREDIT", "S6_PAYABLE"):  # no APIT-credit rule yet
        assert steps.loc[sid, "Source"] == steps.loc[sid, "Evidence"] == "—"

    full = [t.value for t in at.text]
    assert any("personal relief of Rs. 1,800,000" in t for t in full)
    sources = [m.value for m in at.markdown if "Fifth Schedule para 2(a)" in m.value]
    assert any("1,800,000" in s for s in sources)


def test_compare_tax_years_shows_both_laws_and_the_change():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(b for b in at.button if b.label == "Calculate").click()  # 3,000,000 income, 50,000 APIT
    at.run()

    assert not at.exception
    (table,) = [d.value for d in at.dataframe if "Change" in d.value.columns]
    table = table.set_index("")
    assert list(table.columns) == ["YA 2024/25", "YA 2025/26", "Change"]
    assert table.loc["Personal relief"].tolist() == [
        "Rs 1,200,000.00",
        "Rs 1,800,000.00",
        "+600,000.00",
    ]
    assert table.loc["Taxable income"].tolist() == [
        "Rs 1,800,000.00",
        "Rs 1,200,000.00",
        "-600,000.00",
    ]
    assert table.loc["Total tax"].tolist() == ["Rs 252,000.00", "Rs 96,000.00", "-156,000.00"]
    assert table.loc["Tax payable"].tolist() == ["Rs 202,000.00", "Rs 46,000.00", "-156,000.00"]
    assert table.loc["Refund", "Change"] == "same"
    assert "Act No. 45 of 2022" in table.loc["Relief cites", "YA 2024/25"]
    assert "Act No. 2 of 2025" in table.loc["Relief cites", "YA 2025/26"]
    assert table.loc["Bands cite", "Change"] == "changed"


def calculate(monkeypatch, llm=None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if llm is not None:
        monkeypatch.setattr(orchestrator.llm, "explainer_llm_from_env", lambda env: llm)
        next(t for t in at.toggle if t.label == "Gemini rewrite of explanation").set_value(True)
    next(b for b in at.button if b.label == "Calculate").click()
    return at.run()


def pii_notes(at: AppTest) -> list[str]:
    return [c.value for c in at.caption if c.value.startswith("PII redacted")]


def test_explanation_panel_shows_how_much_pii_was_redacted(monkeypatch):
    at = calculate(monkeypatch, EchoLLM())
    assert not at.exception
    assert pii_notes(at) == ["PII redacted: 0 items before the LLM call"]


def test_no_pii_note_when_no_llm_was_asked(monkeypatch):
    at = calculate(monkeypatch)
    assert not at.exception
    assert pii_notes(at) == []


def toggle_labels(at: AppTest) -> list[str]:
    return [t.label for t in at.toggle]


def test_c4_mock_by_default_offers_forced_regenerate(monkeypatch):
    monkeypatch.delenv("C4_LIVE", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not next(t for t in at.toggle if t.label == "Live C4 assurance").value
    assert "C4 mock: force one REGENERATE" in toggle_labels(at)

    next(t for t in at.toggle if t.label == "C4 mock: force one REGENERATE").set_value(True)
    next(b for b in at.button if b.label == "Calculate").click()
    at.run()
    assert not at.exception
    verdicts = next(d.value for d in at.dataframe if "Decision" in d.value.columns)
    assert list(verdicts["Decision"]) == ["REGENERATE", "PASS"]
    assert any("scripted" in c.value for c in at.caption)


def test_live_c4_unreachable_warns_and_still_shows_the_result(monkeypatch):
    monkeypatch.setenv("C4_LIVE", "1")
    monkeypatch.setenv("C4_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert next(t for t in at.toggle if t.label == "Live C4 assurance").value
    assert "C4 mock: force one REGENERATE" not in toggle_labels(at)
    assert any(c.value == "C4 API: http://127.0.0.1:9" for c in at.sidebar.caption)

    next(b for b in at.button if b.label == "Calculate").click()
    at.run()
    assert not at.exception
    assert any(w.value.startswith("**Assurance unavailable**") for w in at.warning)
    assert any("C4 is not reachable" in w.value for w in at.warning)
    assert any(s.value.startswith("Status: **delivered**") for s in at.success)
    assert "S3_BAND_1" in set(at.dataframe[0].value["Step"])  # the rest of the page renders


def test_live_c4_findings_are_shown_with_c4s_columns(monkeypatch):
    from test_c4_http_client import FakeC4, answers, c4_answer

    c4 = FakeC4(answers((200, c4_answer("PASS", "all checks passed"))))
    try:
        monkeypatch.setenv("C4_LIVE", "1")
        monkeypatch.setenv("C4_API_URL", c4.url)
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        next(b for b in at.button if b.label == "Calculate").click()
        at.run()
    finally:
        c4.close()

    assert not at.exception
    findings = next(d.value for d in at.dataframe if "check_name" in d.value.columns)
    assert list(findings.columns) == [
        "check_name", "passed", "severity", "message", "expected_value", "actual_value"
    ]  # fmt: skip
    assert list(findings["check_name"]) == ["line_item_sum"]
    assert any("/assurance/combined-check` on attempt 1" in c.value for c in at.caption)


# --- invoices and receipts: relief panel (offline: mock C2 and C3) ---------------------------


def pick_mock_documents(at: AppTest, *labels: str) -> AppTest:
    next(m for m in at.multiselect if m.label == "Invoices / receipts (mock C2)").set_value(
        list(labels)
    )
    next(b for b in at.button if b.label == "Calculate").click()
    return at.run()


SOLAR = "Solar panel invoice (Rs 800,000, 15 Aug 2025)"
MEDICAL = "Medical receipt (Rs 45,000, 10 Sep 2025): relief withdrawn"


def test_solar_invoice_asks_then_applies_after_confirming_and_answering_yes():
    at = pick_mock_documents(AppTest.from_file(str(APP), default_timeout=60).run(), SOLAR)
    assert not at.exception
    assert any(s.value.startswith("Status: **needs_review**") for s in at.warning)
    assert ":orange-badge[Solar panel relief · Needs your input]" in badges(at)
    docs = next(d.value for d in at.dataframe if "Used as" in d.value.columns)
    assert list(docs["File"]) == ["solar_invoice_2025.json (mock)"]
    assert list(docs["Id"]) == ["doc-1"]
    # C2's reading pre-fills the form; nothing is used until confirmed
    assert next(t for t in at.text_input if t.label == "Amount paid (Rs)").value == "800000.00"
    assert any("(solar_invoice_2025.json (mock), doc-1)" in w.value for w in at.warning)

    next(c for c in at.checkbox if c.label.startswith("I checked the amount")).check()
    for r in at.radio:
        if r.label.endswith("?") and r.label != "Source":
            r.set_value("Yes")
    next(b for b in at.button if b.label == "Recalculate with these answers").click()
    at.run()
    assert not at.exception
    assert any(s.value.startswith("Status: **delivered**") for s in at.success)
    assert ":green-badge[Solar panel relief · Applied]" in badges(at)
    steps = at.dataframe[0].value.set_index("Step")
    assert steps.loc["S1_SOLAR_PANEL_RELIEF", "Result (Rs)"] == "600,000.00"
    assert "paragraph 2(g)" in steps.loc["S1_SOLAR_PANEL_RELIEF", "Source"]
    assert steps.loc["S1_SOLAR_PANEL_RELIEF", "Evidence"] != "—"
    assert any("Rs 200,000.00 paid above this year's limit" in c.value for c in at.caption)
    assert any(m.label == "Refund due" for m in at.metric)


def test_one_no_answer_leaves_the_relief_out_and_still_delivers():
    at = pick_mock_documents(AppTest.from_file(str(APP), default_timeout=60).run(), SOLAR)
    next(c for c in at.checkbox if c.label.startswith("I checked the amount")).check()
    for r in at.radio:
        if r.label.endswith("?") and r.label != "Source":
            r.set_value("No" if "national grid" in r.label else "Yes")
    next(b for b in at.button if b.label == "Recalculate with these answers").click()
    at.run()
    assert not at.exception
    assert ":gray-badge[Solar panel relief · Not applied]" in badges(at)
    assert "S1_SOLAR_PANEL_RELIEF" not in set(at.dataframe[0].value["Step"])


def test_medical_receipt_shows_the_relief_was_withdrawn():
    at = pick_mock_documents(AppTest.from_file(str(APP), default_timeout=60).run(), MEDICAL)
    assert not at.exception
    assert any(s.value.startswith("Status: **delivered**") for s in at.success)
    assert any(b.endswith("· Relief withdrawn]") for b in badges(at))
    assert any("last valid to 2022-12-31" in m.value for m in at.markdown)
    assert any("Act, No. 10 of 2021" in c.value for c in at.caption)
