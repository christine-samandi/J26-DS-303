"""C3's law-change notifications in C1: which ones affect the last calculation, and the page."""

import json
import threading
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from orchestrator.clients import ComponentUnavailable
from orchestrator.law_updates import affects, fetch_notifications, impacts

CHANGE_2025 = {
    "id": "N-20261021-090000-1",
    "created_at": "2026-10-21T09:00:00+00:00",
    "kind": "law_change",
    "audience": "users",
    "title": "Tax law updated: Inland Revenue (Amendment) Act, No. 2 of 2025",
    "message": "...",
    "changes": [
        {"rule_id": "RULE-PERSONAL_RELIEF-V3", "rule_key": "personal_relief",
         "title": "Personal relief", "change": "replaced",
         "summary": "Personal relief: Rs. 1,200,000 now ends on 2025-03-31, replaced by "
                    "RULE-PERSONAL_RELIEF-V4",
         "valid_from": "2023-01-01", "valid_to": "2025-03-31"},
        {"rule_id": "RULE-PERSONAL_RELIEF-V4", "rule_key": "personal_relief",
         "title": "Personal relief", "change": "added",
         "summary": "Personal relief: new version, Rs. 1,800,000, from 2025-04-01",
         "valid_from": "2025-04-01", "valid_to": None},
    ],
}  # fmt: skip
USED_V3 = {"RULE-PERSONAL_RELIEF-V3", "RULE-INDIVIDUAL_SLABS-V3"}


def test_a_calculation_made_with_a_replaced_version_is_affected():
    hits = impacts([CHANGE_2025], USED_V3, date(2025, 4, 1))
    assert [h.rule_id for h in hits] == ["RULE-PERSONAL_RELIEF-V3", "RULE-PERSONAL_RELIEF-V4"]
    assert impacts([CHANGE_2025], USED_V3, date(2024, 4, 1)) == []  # V3 is still right then


def test_new_relief_matters_only_while_in_force_and_not_already_used():
    solar = {"rule_id": "RULE-SOLAR_RELIEF-V1", "change": "added", "valid_from": "2021-04-01"}
    assert affects(solar, USED_V3, date(2025, 4, 1))
    assert not affects(solar, USED_V3, date(2020, 4, 1))
    assert not affects(solar, {"RULE-SOLAR_RELIEF-V1"}, date(2025, 4, 1))
    ended = {"rule_id": "RULE-SENIOR_CITIZEN_RELIEF-V1", "change": "ended",
             "valid_to": "2019-12-31"}  # fmt: skip
    assert not affects(ended, USED_V3, date(2025, 4, 1))  # not used: nothing to redo


class FakeC3:
    def __init__(self, items, status=200):
        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                calls.append(self.path)
                body = json.dumps({"notifications": items, "count": len(items)}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def c3():
    servers = []

    def start(items, status=200):
        servers.append(FakeC3(items, status))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def test_notifications_are_fetched_and_failures_are_reported(c3):
    server = c3([CHANGE_2025])
    since = datetime(2026, 10, 21, tzinfo=UTC)
    assert fetch_notifications(server.url, since=since)[0]["id"] == CHANGE_2025["id"]
    assert "audience=users" in server.calls[0] and "since=2026-10-21" in server.calls[0]
    with pytest.raises(ComponentUnavailable, match="HTTP 500"):
        fetch_notifications(c3([], status=500).url)
    with pytest.raises(ComponentUnavailable, match="not reachable"):
        fetch_notifications("http://127.0.0.1:9", timeout_s=2)


def test_page_warns_when_the_law_changed_after_the_last_calculation(c3, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    server = c3([CHANGE_2025])
    monkeypatch.setenv("C1_RETRIEVAL", "http")
    monkeypatch.setenv("C3_API_URL", server.url)
    app = Path(__file__).resolve().parents[2] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.session_state["last_calc"] = {
        "at": datetime(2026, 10, 21, 8, 0, tzinfo=UTC),  # before the 09:00 law change
        "ya": "2025/26", "as_of": date(2025, 4, 1), "rule_ids": USED_V3,
    }  # fmt: skip
    at.run()
    assert not at.exception
    (warning,) = at.warning
    assert "The law changed after your last calculation" in warning.value
    assert "Rs. 1,800,000, from 2025-04-01" in warning.value
    assert any(e.label == "Law updates (1)" for e in at.expander)
