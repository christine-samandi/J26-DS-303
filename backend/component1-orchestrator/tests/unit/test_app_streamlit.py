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
