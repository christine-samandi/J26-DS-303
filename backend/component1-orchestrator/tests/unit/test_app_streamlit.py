"""The Streamlit demo (app.py) end to end. Needs the demo extra; skipped without it (CI)."""

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
