"""C1 -> C2 over HTTP: the real client against a local server that speaks ocr_output v0.1.

The answer is what C2's service returned for its sample payslip (SYN_payslip_clean.png):
gross 150,000 and APIT 3,500 for one month.
"""

import json
import threading
from decimal import Decimal as D
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orchestrator.clients import HttpOcrClient, MockAssuranceClient, MockRetrievalClient
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps


def field(name, value, confidence=0.988, review=False):
    return {"field_name": name, "value": value, "raw_text": str(value), "confidence": confidence,
            "bbox": [10, 20, 100, 20], "needs_review": review}  # fmt: skip


PAYSLIP = {
    "schema_version": "0.1",
    "doc_id": "SYN_payslip_clean",
    "document_type": "payslip",
    "languages": ["en"],
    "extracted_fields": [
        field("gross_income", 150000.0),
        field("epf_employee", 12000.0),
        field("apit", 3500.0),
        field("net_amount", 134500.0, 0.985),
        field("pay_period", "2026-09", 0.938, review=True),
    ],
    "overall_confidence": 0.975,
    "needs_review": True,
}


class FakeC2:
    """GET /health and POST /extract(/visual) on a free local port."""

    def __init__(self, answer=PAYSLIP, languages=("eng", "osd"), status=200):
        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self._send(200, json.dumps({"status": "ok", "languages": list(languages)}).encode())

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append((self.path, self.headers["Content-Type"], body))
                if self.path.startswith("/extract/visual"):
                    return self._send(200, b"\x89PNG fake", "image/png")
                self._send(status, json.dumps(answer).encode())

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_c2():
    servers = []

    def start(**kw):
        servers.append(FakeC2(**kw))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def _client(c2, months=1):
    client = HttpOcrClient(c2.url, months_covered=months)
    ref = client.add("SYN_payslip_clean.png", b"image bytes")
    return client, ref


def test_monthly_payslip_is_mapped_and_annualised(fake_c2):
    c2 = fake_c2()
    client, ref = _client(c2)
    out = client.extract("req-1", ref)

    assert ref == "SYN_payslip_clean" and out.document_type == "payslip"
    assert {f.name: (f.value, f.confidence) for f in out.fields} == {
        "employment_income": (D("1800000.00"), D("0.988")),
        "apit_paid": (D("42000.00"), D("0.988")),
    }
    assert len(client.notes) == 2 and "x 12" in client.notes[0]

    ((path, ctype, body),) = c2.calls
    assert path.startswith("/extract?") and "lang=eng" in path  # only installed languages
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'filename="SYN_payslip_clean.png"' in body and b"image bytes" in body


def test_annual_summary_is_not_scaled(fake_c2):
    client, ref = _client(fake_c2(), months=12)
    out = client.extract("req-1", ref)
    assert [f.value for f in out.fields] == [D("150000.00"), D("3500.00")]
    assert client.notes == []


def test_full_pipeline_from_an_uploaded_payslip(fake_c2):
    client, ref = _client(fake_c2())
    request = TaxRequest(request_id="req-0007", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))

    assert result.status == "delivered"
    assert result.calc.employment_income == D("1800000.00")
    # 1,800,000 - relief 1,800,000 = 0 taxable, so the 42,000 APIT is a refund
    assert result.calc.total_tax == 0 and result.calc.refund == D("42000.00")


def test_document_that_is_not_a_payslip_fails_cleanly(fake_c2):
    client, ref = _client(fake_c2(answer=PAYSLIP | {"document_type": "bank_statement"}))
    request = TaxRequest(request_id="req-0008", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed"
    assert "'bank_statement', not a payslip" in result.review_reasons[0]


def test_c2_error_and_c2_down_fail_cleanly(fake_c2):
    client, ref = _client(fake_c2(answer={"detail": "file is not a readable image"}, status=415))
    request = TaxRequest(request_id="req-0009", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed" and "HTTP 415" in result.review_reasons[0]

    down = HttpOcrClient("http://127.0.0.1:9", timeout_s=2)
    request = request.model_copy(update={"documents": (down.add("p.png", b"x"),)})
    result = run(request, Deps(down, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed" and "C2 is not reachable" in result.review_reasons[0]


def test_no_supported_ocr_language_is_reported(fake_c2):
    client, ref = _client(fake_c2(languages=("osd",)))
    request = TaxRequest(request_id="req-0010", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed" and "none of the OCR languages" in result.review_reasons[0]


def test_visual_returns_the_annotated_image(fake_c2):
    client, ref = _client(fake_c2())
    assert client.visual(ref).startswith(b"\x89PNG")


def test_months_covered_is_checked():
    with pytest.raises(ValueError):
        HttpOcrClient(months_covered=0)
