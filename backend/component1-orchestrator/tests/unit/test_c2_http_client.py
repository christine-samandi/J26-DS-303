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
    """POST /extract(/visual) on a free local port."""

    def __init__(self, answer=PAYSLIP, status=200):
        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.end_headers()
                self.wfile.write(body)

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

    assert ref == "doc-1" and out.document_type == "payslip"
    assert {f.name: (f.value, f.confidence) for f in out.fields} == {
        "employment_income": (D("1800000.00"), D("0.988")),
        "apit_paid": (D("42000.00"), D("0.988")),
    }
    assert len(client.notes) == 2 and "x 12" in client.notes[0]

    ((path, ctype, body),) = c2.calls
    assert path.startswith("/extract?") and "lang=auto" in path  # C2 detects the scripts
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'filename="doc-1.png"' in body and b"image bytes" in body
    assert b"SYN_payslip" not in body and "SYN_payslip" not in path  # C2 never sees the name
    assert "doc_id=doc-1" in path and client.reads == {"doc-1": client.last}


def test_uploads_get_neutral_ids_in_the_order_they_are_added(fake_c2):
    client = HttpOcrClient(fake_c2().url)
    refs = [client.add(n, b"x") for n in ("Payslip Sept.PNG", "solar invoice.pdf", "x")]
    assert refs == ["doc-1", "doc-2", "doc-3"]
    assert [name for name, _ in client.uploads.values()] == ["doc-1.png", "doc-2.pdf", "doc-3"]


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


def test_bank_statement_alone_is_not_analysed_and_gives_no_income(fake_c2):
    client, ref = _client(fake_c2(answer=PAYSLIP | {"document_type": "bank_statement"}))
    request = TaxRequest(request_id="req-0008", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed"
    assert "no employment income" in result.review_reasons[0]
    assert client.notes == []  # nothing was scaled: its amounts were not read as a payslip


def test_invoice_fields_are_read_unscaled_for_the_user_to_confirm(fake_c2):
    invoice = {
        "schema_version": "0.1",
        "doc_id": "SYN_solar_invoice_2025",
        "document_type": "invoice",
        "languages": ["en"],
        "extracted_fields": [
            field("issuer", "Sunshine Solar Energy (Pvt) Ltd"),
            field("document_date", "2025-08-15"),
            field("total_amount", 800000.0),
        ],
        "overall_confidence": 0.97,
        "needs_review": False,
    }
    client, ref = _client(fake_c2(answer=invoice))  # months_covered=1 would scale a payslip x12
    out = client.extract("req-0010", ref)
    assert (out.document_type, out.fields) == ("invoice", ())
    assert out.issuer == "Sunshine Solar Energy (Pvt) Ltd"
    assert str(out.document_date) == "2025-08-15"
    assert out.total_amount == D("800000.00")
    assert client.notes == []


def test_unreadable_invoice_fields_are_left_out_not_guessed(fake_c2):
    invoice = {
        "schema_version": "0.1",
        "doc_id": "x",
        "document_type": "receipt",
        "languages": ["en"],
        "extracted_fields": [field("document_date", "15/08/2025"), field("total_amount", "n/a")],
        "overall_confidence": 0.5,
        "needs_review": True,
    }
    client, ref = _client(fake_c2(answer=invoice))
    out = client.extract("req-0011", ref)
    assert (out.issuer, out.document_date, out.total_amount) == (None, None, None)


def test_c2_error_and_c2_down_fail_cleanly(fake_c2):
    client, ref = _client(fake_c2(answer={"detail": "file is not a readable image"}, status=415))
    request = TaxRequest(request_id="req-0009", year_of_assessment="2025/26", documents=(ref,))
    result = run(request, Deps(client, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed" and "HTTP 415" in result.review_reasons[0]

    down = HttpOcrClient("http://127.0.0.1:9", timeout_s=2)
    request = request.model_copy(update={"documents": (down.add("p.png", b"x"),)})
    result = run(request, Deps(down, MockRetrievalClient(), MockAssuranceClient()))
    assert result.status == "failed" and "C2 is not reachable" in result.review_reasons[0]


def test_visual_returns_the_annotated_image(fake_c2):
    client, ref = _client(fake_c2())
    assert client.visual(ref).startswith(b"\x89PNG")


def test_months_covered_is_checked():
    with pytest.raises(ValueError):
        HttpOcrClient(months_covered=0)
