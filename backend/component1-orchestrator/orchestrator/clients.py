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
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json
from pydantic import ValidationError

from orchestrator.models import AssuranceVerdict, Decision, OcrField, OcrResult, PipelineTrace

MOCK_DIR = Path(__file__).resolve().parent / "mock_data"
C2_DEFAULT_URL = "http://localhost:8000"
C3_DEFAULT_URL = "http://localhost:8003"
C4_DEFAULT_URL = "http://localhost:8004"


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


def mock_retrieval_file(day: date) -> Path:
    """C1's mock C3 answer for the YA (1 April – 31 March) that contains `day`."""
    start = day.year if day.month >= 4 else day.year - 1
    return MOCK_DIR / f"retrieval_ya{start}_{(start + 1) % 100:02d}.json"


class MockRetrievalClient:
    """Answers from `mock_data/retrieval_yaYYYY_YY.json` for the YA containing the query's
    as_of_date (C1 queries with the YA start), echoing the caller's request_id. With `path`,
    every query gets that one file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        self.requests.append(request)
        path = self.path or mock_retrieval_file(request.as_of_date)
        if not path.is_file():
            raise ComponentUnavailable(
                f"C1 mock C3 has no rules for {request.as_of_date} ({path.name}); "
                "turn on live C3 for this year"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
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

    def add(self, filename: str, data: bytes) -> str:
        """Register an uploaded file; returns the document ref to put in the TaxRequest."""
        ref = Path(filename).stem or "upload"
        self.uploads[ref] = (filename, data)
        return ref

    def _post(self, path: str, document: str) -> bytes:
        if document not in self.uploads:
            raise ComponentUnavailable(f"no uploaded file for document {document!r}")
        filename, data = self.uploads[document]
        body, content_type = _multipart("file", filename, data)
        query = urllib.parse.urlencode({"lang": "auto", "doc_id": document})
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


C4_DECISIONS = ("PASS", "REGENERATE", "HUMAN_REVIEW")


def _money(value: Decimal) -> str:
    """C4 takes money as decimal strings, never in exponent form."""
    return f"{value:f}"


def c4_payload(trace: PipelineTrace) -> tuple[str, dict]:
    """The C4 endpoint and body for one trace.

    C4 has no pipeline_trace endpoint yet, so the trace is flattened into what it checks:
      - line-item sum: S4's total against the band taxes it adds up (its inputs),
      - citations: one C4 claim per (claim, cited rule), the rule as the source and its C3
        evidence as both source text and excerpt. That makes the excerpt check pass by
        construction, so this checks only that every cited rule is a known source.
    Claims without a citation (S5, S6) and citations of rules without evidence (seed rules
    when C3 is off) are left out: C4 would read them as missing or unknown sources. With no
    claim left, only the line-item check is sent.
    """
    total = next(s for s in trace.calc_steps if s.step_id == "S4_TOTAL_TAX")
    # No band reached means the total is 0; C4 needs at least one line item.
    line_items = [_money(v) for v in total.inputs.values() if v is not None] or ["0"]
    evidence = {
        ref.rule_id: ref.evidence
        for ref in (trace.rules.personal_relief.ref, trace.rules.tax_bands.ref)
        if ref.evidence and ref.evidence.strip()
    }
    claims = [
        {"claim_id": f"{c.claim_id}/{rule_id}", "source_id": rule_id, "excerpt": evidence[rule_id]}
        for c in trace.claims
        for rule_id in c.citation_refs
        if rule_id in evidence
    ]
    body = {
        "reported_total": _money(total.result),
        "line_items": line_items,
        # C1 alone decides whether to regenerate (once at most, see stages.after_assure);
        # C4 only learns how many times C1 already has.
        "regeneration_attempts": trace.attempt - 1,
    }
    if not claims:
        return "/assurance/line-item-check", body
    cited = sorted({c["source_id"] for c in claims})
    sources = [{"source_id": rule_id, "text": evidence[rule_id]} for rule_id in cited]
    return "/assurance/combined-check", body | {"claims": claims, "sources": sources}


class HttpAssuranceClient:
    """C4's real assurance service (combined-check / line-item-check, see `c4_payload`).

    C4's answer has no request_id or attempt, so the verdict takes them from the trace.
    Its failed findings become the verdict's reasons. Keeps the last endpoint, body and
    answer (`last`) so the UI can show what C4 checked. Standard library only.
    """

    def __init__(self, base_url: str = C4_DEFAULT_URL, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last: dict | None = None

    def assure(self, trace: PipelineTrace) -> AssuranceVerdict:
        path, body = c4_payload(trace)
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        raw = _http("C4", req, self.timeout_s)
        try:
            answer = json.loads(raw)
            decision = answer["decision"]["action"]
            findings = answer["findings"]
            reasons = tuple(str(f["message"]) for f in findings if not f["passed"])
            if decision != "PASS" and not reasons:
                reasons = (str(answer["decision"]["reason"]),)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ComponentUnavailable(f"C4's answer is not an assurance response: {e}") from e
        if decision not in C4_DECISIONS:
            raise ComponentUnavailable(f"C4 answered an unknown decision {decision!r}")
        self.last = {"endpoint": path, "request": body, "response": answer}
        return AssuranceVerdict(
            request_id=trace.request_id,
            attempt=trace.attempt,
            decision=decision,
            reasons=reasons if decision != "PASS" else (),
        )


def _on(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "on", "yes")


def assurance_client_from_env(env: Mapping[str, str]) -> MockAssuranceClient | HttpAssuranceClient:
    """C4_LIVE=1 -> the real C4 service at C4_API_URL (default localhost:8004).

    Off by default (unset or anything else) -> the mock, which PASSes.
    """
    if _on(env.get("C4_LIVE")):
        return HttpAssuranceClient(env.get("C4_API_URL", C4_DEFAULT_URL))
    return MockAssuranceClient()
