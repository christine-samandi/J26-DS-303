"""C1 -> C3 over HTTP: the real client against a local server that speaks the retrieval contract.

The server answers with the mock C3 output, so the run must match the mock run exactly; what is
tested here is the HTTP path (request body, validation, errors), not C3's knowledge graph.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from j26_contracts.retrieval import RetrievalRequest

from orchestrator.clients import (
    MOCK_DIR,
    HttpRetrievalClient,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
    retrieval_client_from_env,
)
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps

ANSWER = json.loads((MOCK_DIR / "retrieval_ya2025_26.json").read_text(encoding="utf-8"))


class FakeC3:
    """A local HTTP server on a free port. `reply(body) -> (status, bytes)` decides the answer."""

    def __init__(self, reply):
        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server's name)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                status, payload = reply(body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_c3():
    servers = []

    def start(reply):
        servers.append(FakeC3(reply))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def echo(body):
    return 200, json.dumps(ANSWER | {"request_id": body["request_id"]}).encode()


def _run(retrieval):
    request = TaxRequest(request_id="req-0042", year_of_assessment="2025/26",
                         documents=("payslip_clean",))  # fmt: skip
    return run(request, Deps(MockOcrClient(), retrieval, MockAssuranceClient()))


def test_http_run_matches_the_mock_run(fake_c3):
    c3 = fake_c3(echo)
    client = HttpRetrievalClient(c3.url)
    over_http, mocked = _run(client), _run(MockRetrievalClient())

    assert over_http.status == mocked.status == "delivered"
    assert over_http.calc.total_tax == mocked.calc.total_tax
    assert client.last is not None and client.last.request_id == "req-0042"

    ((path, body),) = c3.received
    assert path == "/retrieve"
    sent = RetrievalRequest.model_validate(body)  # C1 sends a valid contract request
    assert (sent.as_of_date.isoformat(), sent.caller) == ("2025-04-01", "c1-orchestrator")
    assert sorted(sent.rule_types) == ["relief", "tax_slab"]


def test_c3_not_running_fails_the_request_cleanly():
    result = _run(HttpRetrievalClient("http://127.0.0.1:9", timeout_s=2))
    assert result.status == "failed" and result.calc is None
    assert "C3 is not reachable" in result.review_reasons[0]


def test_c3_error_status_fails_the_request(fake_c3):
    c3 = fake_c3(lambda body: (503, b'{"detail": "knowledge graph unavailable"}'))
    result = _run(HttpRetrievalClient(c3.url))
    assert result.status == "failed"
    assert "HTTP 503" in result.review_reasons[0]
    assert "knowledge graph unavailable" in result.review_reasons[0]


def test_answer_breaking_the_contract_is_rejected(fake_c3):
    c3 = fake_c3(lambda body: (200, json.dumps({"request_id": body["request_id"]}).encode()))
    result = _run(HttpRetrievalClient(c3.url))
    assert result.status == "failed"
    assert "breaks the retrieval contract" in result.review_reasons[0]


def test_client_is_chosen_from_the_environment():
    assert isinstance(retrieval_client_from_env({}), MockRetrievalClient)
    assert isinstance(retrieval_client_from_env({"C1_RETRIEVAL": "mock"}), MockRetrievalClient)
    live = retrieval_client_from_env({"C1_RETRIEVAL": "http", "C3_API_URL": "http://c3:8003/"})
    assert isinstance(live, HttpRetrievalClient) and live.base_url == "http://c3:8003"


def live_c3_answer(body):
    """ANSWER reshaped like C3's graph backend (graph_backend._source): C3's document title and
    citation format, no C1 mock markers. Citation and excerpt wording are test fixtures."""
    answer = json.loads(json.dumps(ANSWER))
    where = {
        "personal_relief": "s.5 (amending Fifth Schedule para 2(a) of Act No. 24 of 2017)",
        "individual_income_tax_slabs": "s.3(1) "
        "(amending First Schedule para 1 of Act No. 24 of 2017)",
    }
    for r in answer["results"]:
        title = "Inland Revenue (Amendment) Act, No. 2 of 2025"
        r["source"] |= {"document_title": title, "citation": f"{title}, {where[r['rule_key']]}"}
        r["evidence"]["matched_text"] = f"test excerpt for {r['rule_key']}"
    return 200, json.dumps(answer | {"request_id": body["request_id"]}).encode()


def test_live_c3_citations_are_shown_as_c3_gave_them_with_no_mock_marker(fake_c3):
    """[C1 MOCK] marks rules from C1's stand-in for C3. C1 never adds it: with live C3, the
    steps, evidence and explanation show C3's citations unchanged."""
    result = _run(HttpRetrievalClient(fake_c3(live_c3_answer).url))
    refs = (result.rules.personal_relief.ref, result.rules.tax_bands.ref)
    assert [r.citation for r in refs] == [
        "Inland Revenue (Amendment) Act, No. 2 of 2025, s.5 "
        "(amending Fifth Schedule para 2(a) of Act No. 24 of 2017)",
        "Inland Revenue (Amendment) Act, No. 2 of 2025, s.3(1) "
        "(amending First Schedule para 1 of Act No. 24 of 2017)",
    ]
    shown = [*(r.citation for r in refs), *(r.evidence for r in refs), *result.explanation.lines]
    assert not [s for s in shown if "MOCK" in s]

    mock = _run(MockRetrievalClient())
    assert all(
        "[C1 MOCK]" in r.citation
        for r in (mock.rules.personal_relief.ref, mock.rules.tax_bands.ref)
    )
