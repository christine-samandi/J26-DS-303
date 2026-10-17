"""Adapters for the other components. Each has a Mock client now and an HTTP client later.

The orchestrator only talks to the Protocols, so swapping mock for HTTP is a config change.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json
from pydantic import ValidationError

from orchestrator.models import AssuranceVerdict, Decision, OcrResult, PipelineTrace

MOCK_DIR = Path(__file__).resolve().parent / "mock_data"
C3_DEFAULT_URL = "http://localhost:8003"


class ComponentUnavailable(Exception):
    """Another component could not be reached or sent an invalid answer. The request fails."""


class OcrClient(Protocol):  # C2
    def extract(self, request_id: str, document: str) -> OcrResult: ...


class RetrievalClient(Protocol):  # C3
    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput: ...


class AssuranceClient(Protocol):  # C4
    def assure(self, trace: PipelineTrace) -> AssuranceVerdict: ...


class MockOcrClient:
    """Reads `mock_data/ocr/<document>.json`; the document ref is the file's stem."""

    def __init__(self, directory: Path = MOCK_DIR / "ocr") -> None:
        self.directory = directory

    def extract(self, request_id: str, document: str) -> OcrResult:
        path = self.directory / f"{document}.json"
        return OcrResult.model_validate_json(path.read_text(encoding="utf-8"))


class MockRetrievalClient:
    """Answers every query with one fixed RetrievalOutput, echoing the caller's request_id."""

    def __init__(self, path: Path = MOCK_DIR / "retrieval_ya2025_26.json") -> None:
        self.path = path
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        self.requests.append(request)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["request_id"] = request.request_id
        return RetrievalOutput.model_validate(data)


class HttpRetrievalClient:
    """C3's real GraphRAG service: POST <base_url>/retrieve with the frozen retrieval contract.

    Both directions are validated with the shared j26_contracts models. Keeps the last answer
    (`last`) so the UI can show which law was used. Standard library only (no new dependency).
    """

    def __init__(self, base_url: str = C3_DEFAULT_URL, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last: RetrievalOutput | None = None

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        url = f"{self.base_url}/retrieve"
        body = json.dumps(to_contract_json(request)).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise ComponentUnavailable(f"C3 at {url} answered HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ComponentUnavailable(f"C3 is not reachable at {url} ({e})") from e
        try:
            self.last = RetrievalOutput.model_validate_json(raw)
        except ValidationError as e:
            raise ComponentUnavailable(f"C3's answer breaks the retrieval contract: {e}") from e
        return self.last


def retrieval_client_from_env(env: Mapping[str, str]) -> MockRetrievalClient | HttpRetrievalClient:
    """C1_RETRIEVAL=http -> the real C3 service at C3_API_URL (default localhost:8003).

    Anything else (or unset) -> the mock, so tests and offline demos need no running service.
    """
    if env.get("C1_RETRIEVAL", "mock").strip().lower() == "http":
        return HttpRetrievalClient(env.get("C3_API_URL", C3_DEFAULT_URL))
    return MockRetrievalClient()


class MockAssuranceClient:
    """Plays back scripted decisions in order, then PASSes. Keeps every trace it was sent.

    A script entry is a Decision, or a full AssuranceVerdict to control the feedback.
    """

    def __init__(self, script: Sequence[Decision | AssuranceVerdict] = ()) -> None:
        self.script = list(script)
        self.traces: list[PipelineTrace] = []

    def assure(self, trace: PipelineTrace) -> AssuranceVerdict:
        self.traces.append(trace)
        step = self.script.pop(0) if self.script else "PASS"
        if isinstance(step, AssuranceVerdict):
            return step
        return AssuranceVerdict(
            request_id=trace.request_id,
            attempt=trace.attempt,
            decision=step,
            reasons=() if step == "PASS" else (f"mock C4 scripted {step}",),
        )
