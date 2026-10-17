"""Adding a newly published Act: the graph gets new versions and the old ones are closed."""

from dataclasses import asdict
from datetime import date

import pytest

pytest.importorskip("spacy")

from graph_fixtures import RELIEF_CHUNKS  # noqa: E402

from data_collection.sources import load_registry  # noqa: E402
from entity_recognition.extract import extract_entities  # noqa: E402
from knowledge_graph.build import graph_rows, known_documents  # noqa: E402
from knowledge_graph.update import delta_rows, report, version_changes  # noqa: E402
from knowledge_graph.versions import build_versions  # noqa: E402

NEW = "DOC-IRA-AMEND-02-2025"


def _as_graph(versions):
    """The versions the way CURRENT_VERSIONS reads them back from Neo4j."""
    out = []
    for key, vs in versions.items():
        for v in vs:
            d = v.to_dict()
            out.append(
                {"rule_id": v.rule_id, "rule_key": key, "valid_from": d["valid_from"],
                 "valid_to": d["valid_to"], "status": v.status, "amount": v.value,
                 "bands": [], "source": v.source_document_id}
            )  # fmt: skip
    return out


def test_new_act_adds_a_version_and_closes_the_old_one():
    before = build_versions([c for c in RELIEF_CHUNKS if c.document_id != NEW])
    after = build_versions(RELIEF_CHUNKS)
    changes = {(c.kind, c.rule_id): c.detail for c in version_changes(_as_graph(before), after)}
    assert changes[("~", "RULE-PERSONAL_RELIEF-V3")] == "active -> replaced (valid to 2025-03-31)"
    assert changes[("+", "RULE-PERSONAL_RELIEF-V4")].startswith("Rs. 1,800,000  2025-04-01 .. now")
    assert len([k for k in changes if k[1].startswith("RULE-PERSONAL_RELIEF")]) == 2


def test_an_act_that_changes_nothing_reports_no_change():
    versions = build_versions(RELIEF_CHUNKS)
    changes = version_changes(_as_graph(versions), versions)
    assert [c for c in changes if c.rule_id.startswith("RULE-PERSONAL_RELIEF")] == []


def test_only_the_new_acts_rows_are_written():
    extracted = [
        c.to_dict() | {"entities": [asdict(e) for e in extract_entities(c.text)]}
        for c in RELIEF_CHUNKS
    ]
    registry = load_registry()
    rows = graph_rows(extracted, build_versions(RELIEF_CHUNKS), registry.documents)
    delta = delta_rows(rows, {NEW})
    assert {d["document_id"] for d in delta["documents"]} == {NEW}
    assert {c["document_id"] for c in delta["chunks"]} == {NEW}
    assert delta["versions"] == rows["versions"]  # rule rows always rewritten
    text = report([registry.get(NEW)], delta, [])
    assert "No. 2 of 2025" in text and "no change to the tracked rules" in text


def test_corpus_as_known_on_a_date():
    docs = known_documents(load_registry().documents, date(2025, 1, 1))
    ids = {d.document_id for d in docs}
    assert "DOC-IRA-2017" in ids and NEW not in ids and "DOC-IRA-AMEND-45-2022" in ids
