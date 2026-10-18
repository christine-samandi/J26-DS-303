"""Adapters for the other components. Each has a Mock client now and an HTTP client later.

The orchestrator only talks to the Protocols, so swapping mock for HTTP is a config change.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json
from pydantic import ValidationError

from orchestrator.models import AssuranceVerdict, Decision, OcrField, OcrResult, PipelineTrace

MOCK_DIR = Path(__file__).resolve().parent / "mock_data"
C2_DEFAULT_URL = "http://localhost:8000"
C3_DEFAULT_URL = "http://localhost:8003"


class ComponentUnavailable(Exception):
    """Another component could not be reached, or its answer cannot be used. The request fails."""


def _http(component: str, req: urllib.request.Request, timeout_s: float) -> bytes:
    """Send one request; any failure becomes ComponentUnavailable with a readable reason."""
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ComponentUnavailable(
            f"{component} at {req.full_url} answered HTTP {e.code}: {detail}"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ComponentUnavailable(f"{component} is not reachable at {req.full_url} ({e})") from e


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
        raw = _http("C3", req, self.timeout_s)
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


# C2's field names (contracts/schemas/ocr_output_schema.json v0.1) -> the names C1 uses
C2_TO_C1_FIELDS = {"gross_income": "employment_income", "apit": "apit_paid"}
OCR_LANGUAGES = ("sin", "tam", "eng")  # asked of C2 when installed there, in this order


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    """One file as multipart/form-data, which is what C2's /extract expects."""
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
        f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


class HttpOcrClient:
    """C2's real OCR service: uploads a payslip to POST <base_url>/extract.

    The UI registers the uploaded files first (`add`); the document ref in the TaxRequest is
    the name `add` returns. C2's answer (ocr_output v0.1) is mapped to C1's OcrResult:
    gross_income -> employment_income, apit -> apit_paid, with C2's per-field confidence.

    A payslip is usually one month, while the tax is for the whole year: `months_covered`
    (1 = monthly payslip, 12 = annual summary) scales the amounts to a year. The scaling is an
    assumption, so it is written to `notes` and the UI shows it. Keeps C2's raw answer in `last`.
    """

    def __init__(
        self, base_url: str = C2_DEFAULT_URL, months_covered: int = 1, timeout_s: float = 60.0
    ) -> None:
        if not 1 <= months_covered <= 12:
            raise ValueError("months_covered must be between 1 and 12")
        self.base_url = base_url.rstrip("/")
        self.months_covered = months_covered
        self.timeout_s = timeout_s
        self.uploads: dict[str, tuple[str, bytes]] = {}
        self.last: dict | None = None
        self.notes: list[str] = []
        self._lang: str | None = None

    def add(self, filename: str, data: bytes) -> str:
        """Register an uploaded file; returns the document ref to put in the TaxRequest."""
        ref = Path(filename).stem or "upload"
        self.uploads[ref] = (filename, data)
        return ref

    def languages(self) -> str:
        """The OCR languages C2 has installed, e.g. 'sin+tam+eng' (asked once via /health)."""
        if self._lang is None:
            req = urllib.request.Request(f"{self.base_url}/health")
            health = json.loads(_http("C2", req, self.timeout_s))
            installed = set(health.get("languages") or ())
            chosen = [code for code in OCR_LANGUAGES if code in installed]
            if not chosen:
                raise ComponentUnavailable(
                    f"C2 has none of the OCR languages {', '.join(OCR_LANGUAGES)} installed "
                    f"({health.get('error') or health.get('status')})"
                )
            self._lang = "+".join(chosen)
        return self._lang

    def _post(self, path: str, document: str) -> bytes:
        if document not in self.uploads:
            raise ComponentUnavailable(f"no uploaded file for document {document!r}")
        filename, data = self.uploads[document]
        body, content_type = _multipart("file", filename, data)
        query = urllib.parse.urlencode({"lang": self.languages(), "doc_id": document})
        req = urllib.request.Request(
            f"{self.base_url}{path}?{query}",
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        return _http("C2", req, self.timeout_s)

    def extract(self, request_id: str, document: str) -> OcrResult:
        try:
            self.last = out = json.loads(self._post("/extract", document))
        except json.JSONDecodeError as e:
            raise ComponentUnavailable(f"C2 did not answer with JSON: {e}") from e
        self.notes = []
        if out.get("schema_version") != "0.1" or not isinstance(out.get("extracted_fields"), list):
            raise ComponentUnavailable("C2's answer is not ocr_output v0.1")
        if out.get("document_type") != "payslip":
            raise ComponentUnavailable(
                f"C2 read {document!r} as {out.get('document_type')!r}, not a payslip"
            )
        factor = Decimal(12) / Decimal(self.months_covered)
        fields = []
        for f in out["extracted_fields"]:
            name = C2_TO_C1_FIELDS.get(f.get("field_name"))
            if name is None or f.get("value") in (None, ""):
                continue
            try:
                amount = Decimal(str(f["value"]))
                confidence = Decimal(str(f["confidence"]))
            except (InvalidOperation, KeyError, TypeError) as e:
                raise ComponentUnavailable(f"C2 sent an unreadable {name}: {f}") from e
            fields.append(
                OcrField(
                    name=name,
                    value=(amount * factor).quantize(Decimal("0.01")),
                    confidence=confidence,
                )  # fmt: skip
            )
            if factor != 1:
                self.notes.append(
                    f"{name}: Rs {amount:,.2f} on the payslip x {factor:g} "
                    f"= Rs {amount * factor:,.2f} for the year (payslip covers "
                    f"{self.months_covered} month{'s' if self.months_covered > 1 else ''})"
                )
        return OcrResult(
            document_id=out.get("doc_id") or document,
            document_type=out["document_type"],
            fields=tuple(fields),
        )

    def visual(self, document: str) -> bytes:
        """The uploaded image with a box on every field C2 read (PNG)."""
        return self._post("/extract/visual", document)


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
