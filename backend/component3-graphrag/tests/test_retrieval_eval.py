"""Phase 5: the retrieval evaluation scores against the ground truth, not the system."""

from dataclasses import asdict
from datetime import date

import pytest

pytest.importorskip("spacy")

from graph_fixtures import OTHER_CHUNKS, RELIEF_CHUNKS  # noqa: E402

from data_collection.sources import load_registry  # noqa: E402
from entity_recognition.extract import extract_entities  # noqa: E402
from evaluation.retrieval import (  # noqa: E402
    GoldVersion,
    load_questions,
    provision_chunks,
    report,
    run_flat,
    run_graphrag,
    summarise,
)
from knowledge_graph.build import graph_rows  # noqa: E402
from knowledge_graph.versions import build_versions  # noqa: E402
from retrieval_api.graph_backend import from_rows  # noqa: E402

RELIEF_CSV = (
    "version,amount_lkr,valid_from,valid_to,source_document_id,source_provision,verified,note\n"
    "V1,500000,2018-04-01,2019-12-31,DOC-IRA-2017,Fifth Schedule para 2(a),yes,\n"
    "V2,3000000,2020-01-01,2022-12-31,DOC-IRA-AMEND-10-2021,s.55(2)(a) amending,yes,\n"
    "V3,1200000,2023-01-01,2025-03-31,DOC-IRA-AMEND-45-2022,s.36(2)(a) amending,yes,\n"
    "V4,1800000,2025-04-01,,DOC-IRA-AMEND-02-2025,s.5 inserting item (v),yes,\n"
)
BANDS_CSV = (
    "version,valid_from,valid_to,band_order,lower_lkr,upper_lkr,rate_percent,"
    "source_document_id,source_provision,verified\n"
    "B1,2018-04-01,,1,0,,4,DOC-IRA-2017,First Schedule para 1(1),yes\n"
)
QUESTIONS_CSV = (
    "qid,question,as_of_date,expected_rule_key,why\n"
    "Q1,What personal relief could I claim in June 2019?,2019-06-01,personal_relief,V1\n"
    "Q2,Personal relief amount in 2021,2021-06-01,personal_relief,V2\n"
    "Q3,What personal relief can I claim now?,2026-10-01,personal_relief,V4\n"
    "N1,What is the penalty for failing to furnish a return?,2025-06-01,none,not modelled\n"
)


@pytest.fixture()
def setup(tmp_path):
    (tmp_path / "personal_relief_versions.csv").write_text(RELIEF_CSV)
    (tmp_path / "individual_tax_bands.csv").write_text(BANDS_CSV)
    (tmp_path / "retrieval_questions.csv").write_text(QUESTIONS_CSV)
    chunks = RELIEF_CHUNKS + OTHER_CHUNKS
    extracted = [
        c.to_dict() | {"entities": [asdict(e) for e in extract_entities(c.text)]} for c in chunks
    ]
    data = from_rows(graph_rows(extracted, build_versions(chunks), load_registry().documents))
    return load_questions(tmp_path), extracted, data


def test_expected_version_comes_from_the_ground_truth_dates(setup):
    questions, _, _ = setup
    assert [q.gold.version if q.gold else None for q in questions] == ["V1", "V2", "V4", None]


def test_provision_chunks_from_the_ground_truth_provision(setup):
    _, chunks, _ = setup
    g = GoldVersion("personal_relief", "V2", 0, date(2020, 1, 1), None,
                    "DOC-IRA-AMEND-10-2021", "s.55(2)(a) amending")  # fmt: skip
    assert provision_chunks(g, chunks) == {"CHUNK-IRA-AMEND-10-2021-S055-2"}
    g.document_id, g.provision = "DOC-IRA-2017", "Fifth Schedule para 2(a)"
    assert provision_chunks(g, chunks) == {"CHUNK-IRA-2017-SCH5-P002"}


def test_graphrag_and_flat_rag_scored_on_the_same_questions(setup):
    questions, chunks, data = setup
    golds = {"personal_relief": [q.gold for q in questions if q.gold]}
    golds["personal_relief"].append(
        GoldVersion("personal_relief", "V3", 1_200_000, date(2023, 1, 1), date(2025, 3, 31),
                    "DOC-IRA-AMEND-45-2022", "s.36(2)(a)")
    )  # fmt: skip
    graph = summarise(questions, run_graphrag(questions, chunks, golds, data))
    flat = summarise(questions, run_flat(questions, chunks, golds))
    assert graph["correct_answer"] == 1.0 and graph["correct_evidence_at_1"] == 1.0
    assert graph["refused_out_of_scope"] == 1.0
    assert flat["refused_out_of_scope"] == 0.0  # BM25 returns the penalty section anyway
    assert "Flat RAG (BM25)" in report(flat, graph)
