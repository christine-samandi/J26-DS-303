"""Adapters for the other components. Each has a Mock client now and an HTTP client later.

The orchestrator only talks to the Protocols, so swapping mock for HTTP is a config change.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest

from orchestrator.models import AssuranceVerdict, Decision, OcrResult, PipelineTrace

MOCK_DIR = Path(__file__).resolve().parent / "mock_data"


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
