"""C1 -> C4 over HTTP: the real client against a local server that answers like C4's service.

C4's answer shape ({"decision": {"action", "reason"}, "findings": [...]}) is written out in
`c4_answer`; findings' wording is a test fixture. What is tested here is the mapping of C4's
answer to a verdict, the regenerate loop staying under C1's control, and that C4 being down
never costs the user their figures or, by itself, a review flag.
"""

import json
import threading
from decimal import Decimal as D
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orchestrator.clients import (
    HttpAssuranceClient,
    MockAssuranceClient,
    MockOcrClient,
    MockRetrievalClient,
    assurance_client_from_env,
)
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps

LINEAR = ["extract", "normalise", "retrieve", "calculate", "explain", "score", "assure"]
UNREACHABLE = "http://127.0.0.1:9"


class FakeC4:
    """A local HTTP server on a free port. `reply(path, body) -> (status, bytes)` decides."""

    def __init__(self, reply):
        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server's name)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                status, payload = reply(self.path, body)
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
def fake_c4():
    servers = []

    def start(reply):
        servers.append(FakeC4(reply))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def finding(check_name, passed, message):
    """One finding in C4's FindingResponse shape (component4-assurance/api/main.py)."""
    return {"check_name": check_name, "passed": passed, "message": message,
            "severity": "info" if passed else "error", "error_code": "OK" if passed else "E_TEST",
            "expected_value": None, "actual_value": None}  # fmt: skip


def c4_answer(action, reason="", failed=()):
    """C4's answer: one passed finding plus one failed finding per message in `failed`."""
    findings = [finding("line_item_sum", True, "line items add up")]
    findings += [finding("citation", False, m) for m in failed]
    return json.dumps({"decision": {"action": action, "reason": reason}, "findings": findings})


def answers(*replies):
    """Reply with each (status, body) in turn; the last one repeats."""
    queue = list(replies)

    def reply(path, body):
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return status, payload.encode()

    return reply


def typed(**kw) -> TaxRequest:  # GS-001: 3,000,000 income, 50,000 APIT -> 96,000 tax
    return TaxRequest(request_id="req-0001", year_of_assessment="2025/26",
                      employment_income=D(3000000), apit_paid=D(50000), **kw)  # fmt: skip


def _run(assurance, request=None):
    return run(request or typed(), Deps(MockOcrClient(), MockRetrievalClient(), assurance))


def assert_engine_figures(result):
    """GS-001's figures, as the engine computes them whether or not C4 checked them."""
    assert (result.calc.taxable_income, result.calc.total_tax) == (D(1200000), D(96000))
    assert (result.calc.tax_payable, result.calc.refund) == (D(46000), D(0))
    assert result.explanation is not None and result.rules is not None


# --- C4 answers: PASS / REGENERATE / HUMAN_REVIEW -------------------------------------------


def test_pass_is_delivered_as_live_assurance(fake_c4):
    c4 = fake_c4(answers((200, c4_answer("PASS", "all checks passed"))))
    client = HttpAssuranceClient(c4.url)
    result = _run(client)

    assert result.status == "delivered" and result.review_reasons == ()
    assert (result.assurance, result.assurance_note) == ("live", "Checked by C4.")
    assert [(v.decision, v.reasons) for v in result.verdicts] == [("PASS", ())]
    assert_engine_figures(result)

    ((path, body),) = c4.received
    assert path == "/assurance/combined-check"
    assert (body["reported_total"], body["regeneration_attempts"]) == ("96000.00", 0)
    assert client.last["endpoint"] == path and client.last["request"] == body


def test_regenerate_maps_failed_findings_to_reasons_then_reruns(fake_c4):
    c4 = fake_c4(answers(
        (200, c4_answer("REGENERATE", "citation check failed", failed=["CL3 excerpt not found"])),
        (200, c4_answer("PASS", "all checks passed")),
    ))  # fmt: skip
    result = _run(HttpAssuranceClient(c4.url))

    assert result.status == "delivered" and result.attempt == 2
    first, second = result.verdicts
    assert (first.request_id, first.attempt, first.decision) == ("req-0001", 1, "REGENERATE")
    assert first.reasons == ("CL3 excerpt not found",)  # failed findings, not the summary
    assert (second.attempt, second.decision) == (2, "PASS")
    assert [b["regeneration_attempts"] for _, b in c4.received] == [0, 1]
    assert list(result.stages_run) == [*LINEAR, "regenerate", *LINEAR[3:], "deliver"]
    assert result.assurance == "live"
    assert_engine_figures(result)


def test_human_review_maps_to_needs_review_with_c4s_reasons(fake_c4):
    c4 = fake_c4(answers((200, c4_answer("HUMAN_REVIEW", "unknown source", failed=["no S9"]))))
    result = _run(HttpAssuranceClient(c4.url))

    assert result.status == "needs_review" and result.attempt == 1
    assert result.review_reasons == ("C4 asked for human review", "no S9")
    assert len(c4.received) == 1  # no regenerate: C4 asked for a person, not a rerun
    assert result.assurance == "live"
    assert_engine_figures(result)


def test_without_failed_findings_c4s_summary_reason_is_used(fake_c4):
    c4 = fake_c4(answers((200, c4_answer("HUMAN_REVIEW", "drift detected on this rule set"))))
    result = _run(HttpAssuranceClient(c4.url))
    assert result.review_reasons == ("C4 asked for human review", "drift detected on this rule set")


def test_c4_asking_to_regenerate_twice_gets_only_one_regeneration(fake_c4):
    c4 = fake_c4(answers((200, c4_answer("REGENERATE", "", failed=["total does not add up"]))))
    result = _run(HttpAssuranceClient(c4.url))

    assert len(c4.received) == 2  # attempt 1 and attempt 2, then C1 stops
    assert result.attempt == 2 and list(result.stages_run).count("regenerate") == 1
    assert [b["regeneration_attempts"] for _, b in c4.received] == [0, 1]
    assert result.status == "needs_review"
    assert result.review_reasons == (
        "C4 still found problems after regenerating once",
        "total does not add up",
    )
    assert_engine_figures(result)


# --- C4 down or unusable: delivered on C1's own checks, marked unchecked --------------------


def test_c4_not_running_delivers_the_engine_figures_unchecked():
    result = _run(HttpAssuranceClient(UNREACHABLE, timeout_s=2))

    assert result.status == "delivered" and result.review_reasons == ()
    assert result.assurance == "unavailable" and result.verdicts == ()
    assert "C4 is not reachable" in result.assurance_note
    assert "C1's own checks only" in result.assurance_note
    assert list(result.stages_run) == [*LINEAR, "deliver"]
    assert_engine_figures(result)
    assert result == _run(MockAssuranceClient()).model_copy(
        update={
            "assurance": result.assurance,
            "assurance_note": result.assurance_note,
            "verdicts": (),
        }  # fmt: skip
    )  # everything else is exactly the mock run's


@pytest.mark.parametrize(
    ("status", "payload", "in_note"),
    [
        (503, '{"detail": "verifier restarting"}', "HTTP 503"),
        (200, "not json", "not an assurance response"),
        (200, '{"decision": {"action": "PASS"}}', "not an assurance response"),  # no findings
        (200, '{"decision": {"action": "HUMAN_REVIEW"}, "findings": []}', "not an assurance"),
        (200, c4_answer("MAYBE"), "unknown decision 'MAYBE'"),
    ],
)
def test_unusable_c4_answer_is_treated_as_c4_unavailable(fake_c4, status, payload, in_note):
    c4 = fake_c4(answers((status, payload)))
    result = _run(HttpAssuranceClient(c4.url))
    assert result.status == "delivered" and result.assurance == "unavailable"
    assert in_note in result.assurance_note
    assert_engine_figures(result)


def test_c4_down_does_not_hide_c1s_own_review_reasons():
    request = TaxRequest(request_id="req-0001", year_of_assessment="2025/26",
                         documents=("payslip_noisy_low_confidence",))  # fmt: skip
    result = _run(HttpAssuranceClient(UNREACHABLE, timeout_s=2), request)
    assert result.status == "needs_review" and result.assurance == "unavailable"
    assert result.review_reasons == ("C1 confidence 0.55 is below the review threshold 0.7",)


def test_c4_going_down_after_a_regenerate_sends_the_unverified_retry_for_review(fake_c4):
    c4 = fake_c4(answers(
        (200, c4_answer("REGENERATE", "", failed=["CL3 excerpt not found"])),
        (503, '{"detail": "down"}'),
    ))  # fmt: skip
    result = _run(HttpAssuranceClient(c4.url))

    assert result.attempt == 2 and len(c4.received) == 2
    # C4 flagged attempt 1 and nothing checked the retry, so it is not delivered as clean
    assert result.status == "needs_review"
    assert result.review_reasons == (
        "C4 flagged attempt 1 and the retry could not be verified (C4 unavailable)",
        "CL3 excerpt not found",
    )
    assert [v.attempt for v in result.verdicts] == [1]
    assert result.assurance == "unavailable" and "HTTP 503" in result.assurance_note
    assert_engine_figures(result)


# --- selection ------------------------------------------------------------------------------


def test_mock_c4_is_reported_as_mock():
    result = _run(MockAssuranceClient())
    assert result.assurance == "mock" and "scripted" in result.assurance_note


def test_client_is_chosen_from_the_environment():
    assert isinstance(assurance_client_from_env({}), MockAssuranceClient)
    assert isinstance(assurance_client_from_env({"C4_LIVE": "0"}), MockAssuranceClient)
    live = assurance_client_from_env({"C4_LIVE": "1", "C4_API_URL": "http://c4:8004/"})
    assert isinstance(live, HttpAssuranceClient) and live.base_url == "http://c4:8004"
