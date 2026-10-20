"""Scheduled law monitor and the law-change log (no network, no Neo4j)."""

from datetime import UTC, datetime

import pytest

from c3_common import notifications
from data_collection import monitor
from data_collection.sources import load_registry

KNOWN = load_registry().get("DOC-IRA-AMEND-02-2025").url
PAGE = "https://www.ird.gov.lk/en/publications/sitepages/acts.aspx"
NEW = "https://www.ird.gov.lk/en/publications/Acts_Income%20Tax_2017/IR_Act_No._5_2027_E.pdf"
HTML = f"""<html><body>
<a href="{KNOWN}">Inland Revenue (Amendment) Act, No. 2 of 2025</a>
<a href="{NEW}">Inland Revenue (Amendment) Act, No. 5 of 2027</a>
<a href="/en/publications/VAT_Act_No_14_2002.pdf">Value Added Tax Act, No. 14 of 2002</a>
<a href="/en/about.html">About us</a>
</body></html>"""


@pytest.fixture(autouse=True)
def tmp_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv("C3_NOTIFICATIONS_FILE", str(tmp_path / "notifications.json"))
    return tmp_path


def _fetch(url):
    return HTML, url


def test_only_new_income_tax_acts_are_reported_once():
    state = monitor.State()
    new = monitor.check(_fetch, [PAGE], monitor.DEFAULT_KEYWORDS, state)
    assert [f.url for f in new] == [NEW]  # the known Act and the VAT Act are not reported
    assert monitor.check(_fetch, [PAGE], monitor.DEFAULT_KEYWORDS, state) == []  # not twice
    assert [f.url for f in state.pending] == [NEW] and state.last_checked


def test_dismissed_documents_and_unreachable_pages_are_skipped():
    state = monitor.State(dismissed=[NEW])
    assert monitor.check(_fetch, [PAGE], monitor.DEFAULT_KEYWORDS, state) == []

    def down(url):
        raise ConnectionError("no route to host")

    state = monitor.State()
    assert monitor.check(down, [PAGE], monitor.DEFAULT_KEYWORDS, state) == []
    assert state.errors == [PAGE] and state.last_checked is None  # not counted as checked


def test_report_only_mode_notifies_the_maintainer_and_keeps_its_state(tmp_path):
    path = tmp_path / "state.json"
    assert [f.url for f in monitor.run_once(_fetch, path, auto=False)] == [NEW]
    assert monitor.State.load(path).pending[0].status == "new"
    (n,) = notifications.load()
    assert (n.kind, n.audience, n.url) == ("new_document_found", "admin", NEW)
    assert "not added (--no-auto)" in n.message
    assert monitor.run_once(_fetch, path, auto=False) == [] and len(notifications.load()) == 1


def test_new_documents_go_through_the_automatic_update(tmp_path):
    from knowledge_graph.auto_update import Outcome

    path = tmp_path / "state.json"
    asked = []

    def held(url):
        asked.append(url)
        return Outcome("held", "the graph could not be updated (ServiceUnavailable)",
                       "DOC-IRA-AMEND-05-2027", ["1 is a PDF"])  # fmt: skip

    monitor.run_once(_fetch, path, ingest=held)
    (f,) = monitor.State.load(path).pending
    assert asked == [NEW] and (f.status, f.document_id) == ("held", "DOC-IRA-AMEND-05-2027")
    assert "HELD, not added" in notifications.load()[-1].message
    monitor.run_once(_fetch, path, ingest=held)
    assert asked == [NEW]  # a held document is not retried by itself every month...

    def added(url):
        return Outcome("added", "3 rule version(s) changed", "DOC-IRA-AMEND-05-2027")

    assert monitor.retry(NEW, path, ingest=added).status == "added"  # ...only with --retry
    assert monitor.State.load(path).pending[0].status == "added"


@pytest.mark.parametrize("text, seconds", [("30d", 2592000), ("7d", 604800), ("12h", 43200)])
def test_schedule_interval(text, seconds):
    assert monitor.parse_every(text) == seconds


def test_adding_an_act_tells_users_what_changed():
    pytest.importorskip("spacy")
    from test_reliefs import CHUNKS

    from knowledge_graph.update import notify_law_change, version_changes
    from knowledge_graph.versions import build_versions

    def as_rows(versions):
        rows = []
        for vs in versions.values():
            for v in vs:
                d = v.to_dict()
                rows.append({"rule_id": v.rule_id, "rule_key": v.rule_key, "status": v.status,
                             "valid_from": d["valid_from"], "valid_to": d["valid_to"],
                             "source": v.source_document_id,
                             "amount": v.value if v.value_kind == "amount" else None,
                             "cap": v.value if v.value_kind == "cap" else None,
                             "rate_percent": v.value if v.value_kind == "rate" else None,
                             "bands": v.value if v.value_kind == "bands" else None})  # fmt: skip
        return rows

    before = build_versions([c for c in CHUNKS if c.document_id == "DOC-IRA-2017"])
    after = build_versions([c for c in CHUNKS if c.document_id in
                            {"DOC-IRA-2017", "DOC-IRA-AMEND-10-2021"}])  # fmt: skip
    act = load_registry().get("DOC-IRA-AMEND-10-2021")
    n = notify_law_change([act], version_changes(as_rows(before), after), after)
    by_rule = {c.rule_id: c for c in n.changes}
    assert (n.kind, n.audience) == ("law_change", "users")
    solar = by_rule["RULE-SOLAR_RELIEF-V1"]
    assert solar.change == "added" and solar.valid_from == "2021-04-01"
    assert "new relief, up to Rs. 600,000" in solar.summary and "solar panels" in solar.evidence
    senior = by_rule["RULE-SENIOR_CITIZEN_RELIEF-V1"]
    assert senior.change == "ended" and senior.valid_to == "2019-12-31"
    assert by_rule["RULE-PERSONAL_RELIEF-V1"].change == "replaced"
    assert notifications.load()[0].id == n.id


def test_api_lists_notifications_newest_first():
    from fastapi.testclient import TestClient

    from retrieval_api.app import app

    first = notifications.add(notifications.make("new_document_found", "admin", "A", "a"))
    notifications.add(notifications.make("law_change", "users", "B", "b"))
    client = TestClient(app)
    body = client.get("/notifications").json()
    assert body["count"] == 2 and body["notifications"][0]["title"] == "B"
    users = client.get("/notifications", params={"audience": "users"}).json()
    assert [n["title"] for n in users["notifications"]] == ["B"]
    after = datetime.fromisoformat(first.created_at).astimezone(UTC).isoformat()
    newer = client.get("/notifications", params={"since": after}).json()
    assert all(n["created_at"] > first.created_at or n["title"] == "B"
               for n in newer["notifications"])  # fmt: skip
    assert client.get("/notifications", params={"since": "2026-01-01T00:00:00"}).status_code == 422
