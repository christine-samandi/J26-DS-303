"""Phase 2: chunking and rule-based entity extraction, on wording taken from the Acts."""

import json
import re
from typing import get_args

import pytest

pytest.importorskip("spacy")

from entity_recognition.chunking import MAX_CHARS, chunks_from_document  # noqa: E402
from entity_recognition.extract import (  # noqa: E402
    date_qualifier,
    extract_corpus,
    extract_entities,
    words_to_number,
)
from j26_contracts.retrieval import ChunkId, EntityLabel  # noqa: E402


def _found(text):
    return [(e.label, e.text, e.value) for e in extract_entities(text)]


# --- values -------------------------------------------------------------------------------


def test_amounts_in_figures_words_and_millions():
    found = _found(
        "Rs. 1,800,000 for each year; not exceeding one hundred and eighty thousand rupees; "
        "Rs.0; a fine not exceeding ten million rupees; Rs. 3 million"
    )
    amounts = [v for label, _, v in found if label == "AMOUNT"]
    assert amounts == [1_800_000, 180_000, 0, 10_000_000, 3_000_000]


def test_number_words():
    assert words_to_number("one hundred and eighty thousand") == 180_000
    assert words_to_number("forty five") == 45
    assert words_to_number("two million five hundred thousand") == 2_500_000
    assert words_to_number("eighty banana") is None


def test_rates_in_figures_and_words():
    found = _found("taxed at 24%; 25 percent of rent; 15 per centum; six per centum; Zero percent")
    assert [v for label, _, v in found if label == "RATE"] == [24.0, 25.0, 15.0, 6.0, 0.0]


def test_rate_glued_to_a_word_in_the_pdf():
    # "betting and gaming-40%" has no space before the figure in the Act
    assert ("RATE", "40%", 40.0) in _found(
        "gains and profits from conducting betting and gaming-40%"
    )


def test_dates_in_every_printed_format():
    found = _found("April 1, 2025; 24th of October, 2017; 1 April 2018; 01.04.2025 and 31.03.2023")
    dates = [v for label, _, v in found if label == "DATE"]
    assert dates == ["2025-04-01", "2017-10-24", "2018-04-01", "2025-04-01", "2023-03-31"]


def test_date_direction_is_kept_for_versioning():
    entities = extract_entities(
        "for a year of assessment prior to April 1, 2025 or for a year of assessment "
        "commencing on or after April 1, 2025, with effect from January 1, 2020"
    )
    assert [e.qualifier for e in entities if e.label == "DATE"] == [
        "prior_to",
        "on_or_after",
        "on_or_after",
    ]
    assert date_qualifier("the thirtieth day of ") is None


def test_recurring_deadlines_without_a_year_are_not_dates():
    assert not [
        e for e in extract_entities("ending on the thirty first day of March") if e.label == "DATE"
    ]


# --- tax bands ----------------------------------------------------------------------------


def test_tax_band_rows_keep_their_limits_as_amounts():
    found = _found(
        "Not exceeding Rs. 1,000,000 | 6% of the amount in excess of Rs.0 "
        "Exceeding Rs. 1,000,000 but not exceeding Rs. 1,500,000 | Rs. 60,000 plus 18% "
        "Exceeding Rs. 3 million but not exceeding Rs. 6 million | Exceeding Rs. 2,500,000 |"
    )
    slabs = [v for label, _, v in found if label == "TAX_SLAB"]
    assert slabs == ["0-1000000", "1000000-1500000", "3000000-6000000", "2500000-"]
    assert ("AMOUNT", "Rs. 1,500,000", 1_500_000) in found  # band limit also an AMOUNT


def test_lower_case_cap_is_an_amount_not_a_band():
    found = _found("a refund not exceeding Rs. 50,000 for a quarter")
    assert [label for label, _, _ in found] == ["AMOUNT"]


# --- legal references ---------------------------------------------------------------------


def test_chained_legal_references_are_one_entity():
    found = _found(
        "Section 52 of the principal enactment is hereby amended; in item (iii) of that "
        "subparagraph; paragraph 2(a) of the Fifth Schedule to the principal enactment; "
        "section 150 of the Inland Revenue Act, No. 24 of 2017"
    )
    refs = [t for label, t, _ in found if label == "SECTION_REF"]
    assert refs == [
        "Section 52 of the principal enactment",
        "item (iii) of that subparagraph",
        "paragraph 2(a) of the Fifth Schedule to the principal enactment",
        "section 150 of the Inland Revenue Act, No. 24 of 2017",
    ]


# --- terms --------------------------------------------------------------------------------


def test_longest_term_wins():
    found = _found(
        "this personal relief; the taxable income of a resident or non-resident individual; "
        "income from employment; qualifying payments; exempt amounts"
    )
    assert found == [
        ("RELIEF", "personal relief", "personal_relief"),
        ("TAXPAYER_CATEGORY", "resident or non-resident individual",
         "resident_or_non_resident_individual"),
        ("INCOME_TYPE", "income from employment", "employment"),
        ("DEDUCTION", "qualifying payments", "qualifying_payment"),
        ("EXEMPTION", "exempt amounts", "exempt_amount"),
    ]  # fmt: skip


def test_resident_individual_written_as_a_clause():
    found = _found("the taxable income of an individual who is resident in Sri Lanka")
    assert (
        "TAXPAYER_CATEGORY",
        "individual who is resident in Sri Lanka",
        "resident_individual",
    ) in found


def test_only_contract_labels_and_verbatim_offsets():
    text = (
        "(v) Rs. 1,800,000, for each year of assessment commencing on or after April 1, 2025, "
        "under paragraph 2 of the Fifth Schedule, the personal relief of a senior citizen at 6%"
    )
    entities = extract_entities(text)
    assert {e.label for e in entities} <= set(get_args(EntityLabel))
    assert all(text[e.start : e.end] == e.text for e in entities)
    assert all(0 < e.confidence <= 1 for e in entities)


# --- chunking -----------------------------------------------------------------------------


def _parsed():
    long_items = "; ".join(f"({chr(97 + i)}) item {i} " + "word " * 60 for i in range(12))
    return {
        "document_id": "DOC-IRA-2017",
        "segments": [
            {"segment_id": "SEG-IRA-2017-PRE", "kind": "preamble", "number": None,
             "schedule": None, "heading": None, "marginal_note": None, "page_start": 13,
             "page_end": 13, "text": "BE it enacted by the Parliament", "subsections": []},
            {"segment_id": "SEG-IRA-2017-S052", "kind": "section", "number": "52",
             "schedule": None, "heading": "PART I", "marginal_note": "Reliefs.",
             "page_start": 55, "page_end": 55, "text": "",
             "subsections": [{"label": "1", "text": "first text."},
                             {"label": "2", "text": "the aggregate reliefs shall be deducted."}]},
            {"segment_id": "SEG-IRA-2017-S195", "kind": "section", "number": "195",
             "schedule": None, "heading": None, "marginal_note": "Interpretation.",
             "page_start": 180, "page_end": 190, "text": "In this Act -",
             "subsections": [{"label": "1", "text": long_items}]},
            {"segment_id": "SEG-IRA-2017-SCH5-P002", "kind": "schedule_paragraph", "number": "2",
             "schedule": "FIFTH SCHEDULE", "heading": None, "marginal_note": None,
             "page_start": 230, "page_end": 230, "text": "The reliefs ... (a) Rs. 500,000",
             "subsections": []},
        ],
    }  # fmt: skip


def test_chunks_have_contract_ids_and_citation_fields():
    chunks = chunks_from_document(_parsed())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(get_args(ChunkId)[1].pattern, i) for i in ids)
    s52 = next(c for c in chunks if c.chunk_id == "CHUNK-IRA-2017-S052-2")
    assert (s52.section, s52.subsection, s52.page_start) == ("52", "2", 55)
    assert s52.text == "(2) the aggregate reliefs shall be deducted."
    sch = next(c for c in chunks if c.chunk_id == "CHUNK-IRA-2017-SCH5-P002")
    assert (sch.schedule, sch.paragraph, sch.section) == ("FIFTH SCHEDULE", "2", None)


def test_short_lead_text_joins_the_first_subsection():
    s195 = [c for c in chunks_from_document(_parsed()) if c.section == "195"]
    assert s195[0].text.startswith("In this Act - (1) (a) item 0")


def test_long_text_is_split_between_items_without_losing_words():
    parts = [c for c in chunks_from_document(_parsed()) if c.section == "195"]
    assert len(parts) > 1 and all(len(c.text) <= MAX_CHARS for c in parts)
    assert [c.part for c in parts] == list(range(1, len(parts) + 1))
    assert parts[1].text.startswith("(")
    joined = " ".join(c.text for c in parts)
    assert joined.count("word") == 12 * 60


def test_extract_corpus_writes_json(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "DOC-IRA-2017.json").write_text(json.dumps(_parsed()), encoding="utf-8")
    stats = extract_corpus(processed, tmp_path / "extracted")
    assert stats["documents"] == 1 and stats["chunks"] >= 4
    data = json.loads((tmp_path / "extracted" / "DOC-IRA-2017.json").read_text(encoding="utf-8"))
    sch = next(c for c in data["chunks"] if c["chunk_id"] == "CHUNK-IRA-2017-SCH5-P002")
    assert {"label": "AMOUNT", "value": 500000}.items() <= sch["entities"][-1].items()


def test_income_type_is_named_by_its_head_words():
    # found in the evaluation: "income from an investment" was tagged inside this phrase
    found = _found("an individual with rental income from an investment asset")
    assert ("INCOME_TYPE", "rental income", "investment") in found
    assert not [t for label, t, _ in found if t == "income from an investment"]
