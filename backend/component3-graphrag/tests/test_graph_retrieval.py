"""Phase 4: GraphRAG retrieval answers in contract form, with the version in force on the date.

The graph is built from Act wording (graph_fixtures) as plain rows, so no Neo4j is needed.
"""

from dataclasses import asdict
from datetime import date

import pytest

pytest.importorskip("spacy")

from fastapi.testclient import TestClient  # noqa: E402
from graph_fixtures import OTHER_CHUNKS, RELIEF_CHUNKS, SLAB_CHUNKS  # noqa: E402

from c3_common.contracts import OUTPUT_SCHEMA, contract_errors  # noqa: E402
from data_collection.sources import load_registry  # noqa: E402
from entity_recognition.extract import extract_entities  # noqa: E402
from j26_contracts.retrieval import RetrievalRequest, to_contract_json  # noqa: E402
from knowledge_graph.build import graph_rows  # noqa: E402
from knowledge_graph.versions import build_versions  # noqa: E402
from retrieval_api import graph_backend  # noqa: E402
from retrieval_api.lexical import LexicalIndex, query_terms  # noqa: E402


@pytest.fixture(scope="module")
def data():
    chunks = RELIEF_CHUNKS + SLAB_CHUNKS + OTHER_CHUNKS
    extracted = [
        c.to_dict() | {"entities": [asdict(e) for e in extract_entities(c.text)]} for c in chunks
    ]
    rows = graph_rows(extracted, build_versions(chunks), load_registry().documents)
    return graph_backend.from_rows(rows)


def _ask(data, text=None, day="2024-06-01", **kw):
    req = RetrievalRequest(schema_version="1.0.0", request_id="t", query_text=text,
                           as_of_date=date.fromisoformat(day), **kw)  # fmt: skip
    out = graph_backend.retrieve(req, data)
    assert contract_errors(to_contract_json(out), OUTPUT_SCHEMA) == []
    return out


@pytest.mark.parametrize(
    "day, rule_id, amount",
    [
        ("2019-06-01", "RULE-PERSONAL_RELIEF-V1", 500_000),
        ("2021-06-01", "RULE-PERSONAL_RELIEF-V2", 3_000_000),
        ("2024-06-01", "RULE-PERSONAL_RELIEF-V3", 1_200_000),
        ("2026-10-15", "RULE-PERSONAL_RELIEF-V4", 1_800_000),
    ],
)
def test_the_version_in_force_on_the_date_is_returned(data, day, rule_id, amount):
    out = _ask(data, "What personal relief can I claim?", day)
    (top,) = out.results
    assert (top.rule_id, top.values.amount.amount) == (rule_id, amount)
    assert top.validity.in_force_on_as_of_date and top.confidence.level == "high"
    assert out.warnings == []


def test_result_carries_citation_and_verbatim_evidence(data):
    (top,) = _ask(data, "personal relief", "2024-06-01").results
    source_text = {c.chunk_id: c.text for c in RELIEF_CHUNKS}[top.evidence.chunk_ids[0]]
    assert top.evidence.matched_text in source_text
    assert "No. 45 of 2022" in top.source.citation and "Fifth Schedule" in top.source.citation
    edges = {(e.from_, e.relation, e.to) for e in top.evidence.graph_path}
    assert ("RULE-PERSONAL_RELIEF-V3", "SUPERSEDES", "RULE-PERSONAL_RELIEF-V2") in edges
    assert ("CHUNK-IRA-AMEND-45-2022-S036-2", "AMENDS", "SEG-IRA-2017-SCH5-P002") in edges
    assert top.validity.status == "replaced"  # replaced today, but in force on the date asked
    assert top.validity.superseded_by == "RULE-PERSONAL_RELIEF-V4"


def test_history_lists_the_other_versions_after_the_one_in_force(data):
    out = _ask(data, "personal relief", "2021-06-01", include_history=True)
    assert [r.rule_id[-2:] for r in out.results] == ["V2", "V1", "V3", "V4"]
    assert [r.validity.in_force_on_as_of_date for r in out.results] == [True, False, False, False]
    assert out.results[1].confidence.components.temporal_match == 0.0


def test_tax_bands_come_back_as_structured_values(data):
    (top,) = _ask(data, "tax slabs for individuals", "2021-06-01").results
    assert top.rule_type == "tax_slab"
    assert [(b.lower, b.upper, b.rate_percent) for b in top.values.bands] == [
        (0, 3_000_000, 6.0),
        (3_000_000, None, 12.0),
    ]


def test_unrelated_question_is_no_match(data):
    out = _ask(data, "customs duty on imported cars")
    assert out.results == [] and "NO_MATCH" in out.warnings


def test_date_before_the_act_warns(data):
    out = _ask(data, "personal relief", "2017-01-01")
    assert out.results == []
    assert {"NO_RULE_IN_FORCE_ON_DATE", "AS_OF_DATE_BEFORE_CORPUS"} <= set(out.warnings)


def test_rule_ids_and_rule_types_without_text(data):
    out = _ask(data, rule_ids=["RULE-PERSONAL_RELIEF-V2"], day="2021-06-01")
    assert [r.rule_id for r in out.results] == ["RULE-PERSONAL_RELIEF-V2"]
    out = _ask(data, rule_types=["tax_rate"], day="2021-06-01")
    assert [r.rule_key for r in out.results] == ["individual_income_tax_slabs"]
    out = _ask(data, rule_types=["deduction"], day="2021-06-01")
    assert out.results == [] and "PARTIAL_CORPUS_COVERAGE" in out.warnings


def test_taxpayer_category_filter(data):
    out = _ask(data, "personal relief", taxpayer_category="non_resident_individual")
    assert out.results == []


def test_lexical_coverage_ignores_unknown_words():
    index = LexicalIndex({"a": "the personal relief of Rs. 500,000", "b": "a penalty is payable"})
    assert index.coverage("what personal relief applies to me?", "a") == 1.0
    assert index.coverage("personal relief", "b") == 0.0
    assert "rate" in query_terms("tax slabs")  # people's words mapped to the Acts' words


def test_api_serves_the_graph_in_graphrag_mode(data, monkeypatch):
    from types import SimpleNamespace

    import retrieval_api.app as app_module

    monkeypatch.setattr(app_module, "get_settings", lambda: SimpleNamespace(mode="graphrag"))
    monkeypatch.setattr(graph_backend, "graph", lambda: data)
    body = TestClient(app_module.app).post("/retrieve", json={
        "schema_version": "1.0.0", "request_id": "api-1", "query_text": "personal relief",
        "as_of_date": "2021-06-01",
    }).json()  # fmt: skip
    assert contract_errors(body, OUTPUT_SCHEMA) == []
    assert body["results"][0]["values"]["amount"]["amount"] == 3_000_000
    assert body["metadata"]["retrieval_mode"] == "graphrag"


class _Date:
    def __init__(self, iso):
        self.iso = iso

    def to_native(self):
        return date.fromisoformat(self.iso)


class _FakeDriver:
    """Answers the three retrieval queries the way Neo4j would (records as dicts)."""

    def execute_query(self, query):
        if "collect(DISTINCT b" in query:
            return [{
                "v": {"rule_id": "RULE-PERSONAL_RELIEF-V1", "rule_key": "personal_relief",
                      "valid_from": _Date("2018-04-01"), "valid_to": None, "status": "active",
                      "amount": 500000, "matched_text": "Rs. 500,000 for each year of assessment",
                      "source_document_id": "DOC-IRA-2017",
                      "income_types": ["employment"]},
                "rule": {"rule_key": "personal_relief", "rule_type": "relief",
                         "title": "Personal relief", "provision": "Fifth Schedule, paragraph 2(a)"},
                "doc": {"document_id": "DOC-IRA-2017",
                        "title": "Inland Revenue Act, No. 24 of 2017",
                        "document_type": "act", "act_number": "24 of 2017",
                        "certified_date": "2017-10-24", "url": "https://x"},
                "chunk": {"chunk_id": "CHUNK-IRA-2017-SCH5-P002", "schedule": "FIFTH SCHEDULE",
                          "paragraph": "2", "section": None, "subsection": None,
                          "page_start": 230, "text": "(a) Rs. 500,000 for each year of assessment"},
                "bands": [], "categories": ["resident_individual"],
                "evidence": ["CHUNK-IRA-2017-SCH5-P002"], "supersedes": None,
                "superseded_by": None, "amends": [],
            }], None, None  # fmt: skip
        if "AS direct" in query:
            return [{"rule_key": "personal_relief", "direct": ["CHUNK-IRA-2017-SCH5-P002"],
                     "near": []}], None, None  # fmt: skip
        return [{"chunk_id": "CHUNK-IRA-2017-SCH5-P002",
                 "text": "the personal relief of Rs. 500,000"}], None, None  # fmt: skip


def test_loading_from_neo4j_records():
    data = graph_backend.from_neo4j(_FakeDriver())
    assert data.versions[0]["valid_from"] == date(2018, 4, 1)
    (top,) = _ask(data, "personal relief", "2020-01-01").results
    assert top.source.citation.startswith("Inland Revenue Act, No. 24 of 2017, Fifth Schedule")
