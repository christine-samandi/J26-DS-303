"""Gemini behind the explanation agent's `TextLLM` interface.

The only C1 module that talks to an LLM provider. The SDK (google-genai, the `llm` extra) is
imported lazily, so the rest of C1, and CI, run without it. Off unless configured:

    C1_EXPLAINER_LLM          none (default: template text only) | gemini
    GEMINI_API_KEY            required for gemini
    GEMINI_MODEL              default gemini-3.8-flash
    C1_EXPLAINER_TIMEOUT_MS   default 20000; on timeout the template is shown

Temperature 0, seed 0, JSON output. No retries: a failed call falls back to the template,
and retries would spend the free tier's small daily quota. The explanation prompt holds only
engine figures and step descriptions, never personal data, and is never logged here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_MS = 20_000


class GeminiTextLLM:
    temperature = 0.0

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        sdk: tuple[Any, Any] | None = None,  # (genai, types); tests pass a fake
    ) -> None:
        genai, types = sdk or _import_sdk()
        self.model = model
        # Without a timeout a dead connection hangs forever (seen with broken IPv6 routes).
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms)
        )
        self._config = types.GenerateContentConfig(
            temperature=self.temperature, seed=0, response_mime_type="application/json"
        )

    def generate(self, prompt: str) -> tuple[str, str]:
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config
        )
        return response.text or "", response.model_version or self.model


def explainer_llm_from_env(
    env: Mapping[str, str] = os.environ, sdk: tuple[Any, Any] | None = None
) -> GeminiTextLLM | None:
    """The configured explainer LLM, or None to show the template text."""
    provider = env.get("C1_EXPLAINER_LLM", "none").strip().lower()
    if provider in ("", "none"):
        return None
    if provider != "gemini":
        raise ValueError(f"unknown C1_EXPLAINER_LLM {provider!r}: use 'none' or 'gemini'")
    api_key = env.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("C1_EXPLAINER_LLM=gemini needs GEMINI_API_KEY")
    return GeminiTextLLM(
        api_key,
        env.get("GEMINI_MODEL", DEFAULT_MODEL),
        int(env.get("C1_EXPLAINER_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)),
        sdk=sdk,
    )


def _import_sdk() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "the Gemini explainer needs the llm extra: pip install -e '.[llm]'"
        ) from e
    return genai, types
