"""The Gemini explainer client, against a fake google-genai SDK: no network, no API key,
and no google-genai install needed (CI installs only the dev extra)."""

import ast
import json
import sys
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator
from orchestrator.clients import MockAssuranceClient, MockOcrClient, MockRetrievalClient
from orchestrator.graph import run
from orchestrator.llm import DEFAULT_TIMEOUT_MS, GeminiTextLLM, explainer_llm_from_env
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps

KEY = "test-key-not-real"


def reword(prompt: str, cheat: dict | None = None) -> str:
    """What an honest (or cheating) model would answer to the explain prompt."""
    steps = json.loads(prompt.split("Steps:\n", 1)[1].split("\n\nAnswer with JSON", 1)[0])
    texts = {s["step_id"]: f"Put simply: {s['text']}" for s in steps}
    texts.update(cheat or {})
    return json.dumps({"lines": [{"step_id": k, "text": v} for k, v in texts.items()]})


class FakeSDK:
    """Stands in for (google.genai, google.genai.types) and records every call."""

    def __init__(self, respond=reword, model_version="gemini-3.8-flash-001", error=None):
        self.clients: list[dict] = []
        self.requests: list[dict] = []
        sdk = self

        class Models:
            def generate_content(self, **kwargs):
                sdk.requests.append(kwargs)
                if error:
                    raise error
                text = respond(kwargs["contents"])
                return SimpleNamespace(text=text, model_version=model_version)

        class Client:
            def __init__(self, **kwargs):
                sdk.clients.append(kwargs)
                self.models = Models()

        self.genai = SimpleNamespace(Client=Client)
        self.types = SimpleNamespace(
            HttpOptions=lambda **kw: {"HttpOptions": kw},
            GenerateContentConfig=lambda **kw: {"GenerateContentConfig": kw},
        )

    @property
    def modules(self):
        return self.genai, self.types


def gemini(fake: FakeSDK, **kw) -> GeminiTextLLM:
    return GeminiTextLLM(KEY, "gemini-3.8-flash", sdk=fake.modules, **kw)


# --- the client ---------------------------------------------------------------------------


def test_client_is_deterministic_json_with_a_timeout_and_calls_nothing_when_built():
    fake = FakeSDK()
    llm = gemini(fake, timeout_ms=5000)
    assert fake.clients == [{"api_key": KEY, "http_options": {"HttpOptions": {"timeout": 5000}}}]
    assert fake.requests == []
    assert llm.temperature == 0 and llm.model == "gemini-3.8-flash"


def test_generate_sends_the_prompt_and_returns_text_and_answering_model():
    fake = FakeSDK(respond=lambda prompt: '{"ok": true}')
    assert gemini(fake).generate("the prompt") == ('{"ok": true}', "gemini-3.8-flash-001")
    (request,) = fake.requests
    assert request == {
        "model": "gemini-3.8-flash",
        "contents": "the prompt",
        "config": {
            "GenerateContentConfig": {
                "temperature": 0.0,
                "seed": 0,
                "response_mime_type": "application/json",
            }
        },
    }


def test_empty_reply_and_missing_model_version_are_tolerated():
    fake = FakeSDK(respond=lambda prompt: None, model_version=None)
    assert gemini(fake).generate("p") == ("", "gemini-3.8-flash")


def test_missing_sdk_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[llm\]'"):
        GeminiTextLLM(KEY)


# --- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize("env", [{}, {"C1_EXPLAINER_LLM": "none"}, {"C1_EXPLAINER_LLM": " "}])
def test_explainer_is_off_unless_configured(env):
    assert explainer_llm_from_env({**env, "GEMINI_API_KEY": KEY}, sdk=FakeSDK().modules) is None


def test_gemini_explainer_from_env():
    fake = FakeSDK()
    env = {
        "C1_EXPLAINER_LLM": "Gemini",
        "GEMINI_API_KEY": KEY,
        "GEMINI_MODEL": "gemini-other",
        "C1_EXPLAINER_TIMEOUT_MS": "7000",
    }
    llm = explainer_llm_from_env(env, sdk=fake.modules)
    assert llm.model == "gemini-other"
    assert fake.clients[0]["http_options"] == {"HttpOptions": {"timeout": 7000}}


def test_gemini_explainer_defaults():
    fake = FakeSDK()
    env = {"C1_EXPLAINER_LLM": "gemini", "GEMINI_API_KEY": KEY}
    llm = explainer_llm_from_env(env, sdk=fake.modules)
    assert llm.model == "gemini-3.8-flash"
    assert fake.clients[0]["http_options"] == {"HttpOptions": {"timeout": DEFAULT_TIMEOUT_MS}}


@pytest.mark.parametrize(
    "env,message",
    [
        ({"C1_EXPLAINER_LLM": "gemini"}, "needs GEMINI_API_KEY"),
        ({"C1_EXPLAINER_LLM": "gpt", "GEMINI_API_KEY": KEY}, "unknown C1_EXPLAINER_LLM 'gpt'"),
    ],
)
def test_bad_configuration_is_refused(env, message):
    with pytest.raises(ValueError, match=message):
        explainer_llm_from_env(env, sdk=FakeSDK().modules)


# --- through the pipeline -----------------------------------------------------------------


def pipeline(fake: FakeSDK):
    d = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient())
    d.explainer_llm = gemini(fake)
    request = TaxRequest(
        request_id="req-0001",
        year_of_assessment="2025/26",
        employment_income=D(3000000),
        apit_paid=D(50000),
    )
    return run(request, d), d.assurance.traces[0]


def test_gemini_rewrite_runs_through_the_pipeline_and_is_traced():
    fake = FakeSDK()
    result, trace = pipeline(fake)
    assert result.status == "delivered"
    assert all(line.startswith("Put simply: ") for line in result.explanation.lines)
    (call,) = trace.llm_calls
    assert (call.model, call.temperature) == ("gemini-3.8-flash-001", 0)
    assert len(fake.requests) == 1


def test_cheating_gemini_is_caught_in_the_pipeline():
    fake = FakeSDK(respond=lambda p: reword(p, {"S6_PAYABLE": "You owe Rs. 40,000."}))
    result, trace = pipeline(fake)
    assert result.explanation.hallucination_flag and result.status == "needs_review"
    assert result.calc.tax_payable == D(46000)
    assert "Rs. 40,000" in trace.llm_calls[0].output


def test_gemini_failure_shows_the_template_and_leaks_nothing():
    fake = FakeSDK(error=RuntimeError(f"503 UNAVAILABLE (key={KEY})"))
    result, trace = pipeline(fake)
    assert result.status == "delivered" and trace.llm_calls == ()
    assert result.warnings == ("explanation LLM call failed (RuntimeError); template text shown",)
    assert KEY not in result.model_dump_json() and KEY not in trace.model_dump_json()


def test_only_the_llm_module_imports_an_llm_sdk():
    package = Path(orchestrator.__file__).parent
    importers = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "google":
                importers.add(path.name)
            if isinstance(node, ast.Import) and any(
                a.name.split(".")[0] == "google" for a in node.names
            ):
                importers.add(path.name)
    assert importers == {"llm.py"}
