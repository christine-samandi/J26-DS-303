"""General relief engine: every Fifth Schedule para 2 relief rebuilt from the Acts' wording.

The chunks below are the official text (lists of items shortened with "..."), so the tests
exercise the three kinds of amendment the corpus uses: a subparagraph replaced, words
substituted inside a subparagraph, and new subparagraphs added.
"""

from dataclasses import asdict
from datetime import date

import pytest

pytest.importorskip("spacy")

from graph_fixtures import FIFTH, RELIEF_CHUNKS, SLAB_CHUNKS, _c  # noqa: E402

from c3_common.contracts import OUTPUT_SCHEMA, contract_errors  # noqa: E402
from data_collection.sources import load_registry  # noqa: E402
from entity_recognition.extract import extract_entities  # noqa: E402
from j26_contracts.retrieval import RetrievalRequest, to_contract_json  # noqa: E402
from knowledge_graph.build import graph_rows  # noqa: E402
from knowledge_graph.reliefs import (  # noqa: E402
    consolidate,
    operation_date,
    read_wording,
    split_subparagraphs,
)
from knowledge_graph.versions import build_versions, document_order  # noqa: E402
from retrieval_api import graph_backend  # noqa: E402

PARA2_2017 = _c(
    "CHUNK-IRA-2017-SCH5-P002", "DOC-IRA-2017",
    "The reliefs referred to in section 52 shall be as follows: (a) Rs. 500,000 for each year of "
    "assessment, except that an individual who is a trustee, receiver, executor or liquidator "
    "shall not be entitled to deduct this personal relief as such trustee, receiver, executor or "
    "liquidator, and the relief is not available to be deducted against gains from the "
    "realisation of investment assets; (b) in the case of an individual with income from "
    "employment, Rs. 700,000 for each year of assessment, up to the total of the individual’s "
    "income from employment for the year; (c) in the case of an individual with rental income "
    "from an investment asset, an amount equal to 25 percent of the total rental income for the "
    "year of assessment, being a relief for the repair, maintenance, and depreciation relating "
    "to the investment asset; (d) in the case of an individual who is a senior citizen in a year "
    "with interest income derived from a financial institution, Rs. 1,500,000 for each year of "
    "assessment, up to the total of the individual’s interest income for the year; (e) in the "
    "case of a resident individual or partner of a partnership with income earned in foreign "
    "currency in Sri Lanka, from any service rendered in or outside Sri Lanka to any person to "
    "be utilized outside Sri Lanka, Rs. 15,000,000 for each year of assessment, up to the total "
    "of such income for the year.",
    schedule="FIFTH SCHEDULE", paragraph="2",
)  # fmt: skip
ACT10 = "DOC-IRA-AMEND-10-2021"
ACT10_CHUNKS = [
    _c("CHUNK-IRA-AMEND-10-2021-S001-3", ACT10,
       "(3) The provisions of sections referred to in Table ‘A’ set out in this Amendment Act "
       "shall be deemed to have come into operation on April 1, 2021.", section="1",
       note="Short title and the date of operation"),
    _c("CHUNK-IRA-AMEND-10-2021-S055-0", ACT10,
       "The Fifth Schedule to the principal enactment is hereby amended as follows: -",
       section="55", note=FIFTH),
    _c("CHUNK-IRA-AMEND-10-2021-S055-2-P1", ACT10,
       "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) of that "
       "paragraph and the substitution therefor, of the following subparagraph: - “(a) (i) Rs. "
       "500,000, for each year of assessment prior to January 1, 2020; and (ii) Rs. 3,000,000, "
       "for each year of assessment commencing on or after January 1, 2020, except that an "
       "individual who is a trustee ... shall not be entitled to deduct this personal relief;”; "
       "(b) in subparagraph (b) of that paragraph, by the substitution for the words “year of "
       "assessment,” of the words and figures “year of assessment, but prior to January 1, "
       "2020,”; (c) in subparagraph (d) of that paragraph, by the substitution for the words "
       "“for the year;”, of the words and figures “for the year up to December 31, 2019;”; (d) "
       "in subparagraph (e) of that paragraph, by the substitution for the words “for the "
       "year.”, of the words and figures “for the year up to December 31, 2019;”; (e) by the "
       "addition immediately after subparagraph (e) of that paragraph, of the following new "
       "subparagraphs:- “(f) in the case of a resident individual, following expenditure up to "
       "a total sum of Rs. 1,200,000, incurred for a year of assessment on or after January 1, "
       "2020: - (i) health expenditure including contributions to medical insurance; (ii) "
       "vocational education or other educational expenditure ...; (iii) interest paid on "
       "housing loans; (iv) contributions made to any local pension scheme ...;",
       section="55", note=FIFTH),
    _c("CHUNK-IRA-AMEND-10-2021-S055-2-P2", ACT10,
       "(v) expenditure incurred for the purchase of shares ...; (g) in the case of a resident "
       "individual who has acquired solar panels to fix on his premises and connected to the "
       "national grid, Rs. 600,000 for each year of assessment, upto the total expenditure on "
       "such solar panels or upto the amounts paid to a bank in respect of any loan obtained to "
       "acquire such solar panels.”.", section="55", note=FIFTH),
    _c("CHUNK-IRA-AMEND-10-2021-S061", ACT10,
       "In the event of any inconsistency ... Table ‘A’ (section 1) Column I | Column II 6 | 11 "
       "54(3) | paragraph 4 of the Fourth Schedule 55(1)(b) | subparagraphs (d), (e) and (f) of "
       "paragraph 1 of the Fifth Schedule 55(2)(e) | subparagraph (g) of paragraph 2 of the "
       "Fifth Schedule 56(4) | paragraphs 10 and 11 of the Sixth Schedule", section="61",
       note="Sinhala text to prevail"),
]  # fmt: skip
ACT45_CHUNK = _c(
    "CHUNK-IRA-AMEND-45-2022-S036-2", "DOC-IRA-AMEND-45-2022",
    "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) of that "
    "paragraph and the substitution therefor, of the following subparagraph: - “(a) (i) Rs. "
    "500,000, for each year of assessment prior to January 1, 2020; (ii) Rs. 3,000,000, for each "
    "year of assessment commencing on or after January 1, 2020, but prior to April 1, 2022; "
    "(iii) Rs. 2,250,000, for first nine months and Rs. 300,000 for second three months of the "
    "year of assessment commencing on April 1, 2022; and (iv) Rs. 1,200,000, for each year of "
    "assessment commencing on or after April 1, 2023, except that ...;”; and (b) in "
    "subparagraph (f) of that paragraph, by the substitution for the words and figures “on or "
    "after January 1, 2020: -”, of the words and figures “on or after January 1, 2020, but "
    "prior to April 1, 2022 and sum of Rs. 900,000, incurred for the first nine months of the "
    "year of assessment commencing on April 1, 2022: -”.",
    section="36", note=FIFTH,
)  # fmt: skip
ACT2025 = [c for c in RELIEF_CHUNKS if c.document_id == "DOC-IRA-AMEND-02-2025"]
CHUNKS = [PARA2_2017, *ACT10_CHUNKS, ACT45_CHUNK, *ACT2025]


def test_subparagraphs_split_on_letters_not_roman_numerals():
    parts = split_subparagraphs("(f) expenditure: - (i) health; (v) shares; (g) solar panels.", "f")
    assert parts == {"f": "expenditure: - (i) health; (v) shares;", "g": "solar panels."}


def test_all_three_kinds_of_amendment_are_applied_in_order():
    history = consolidate(CHUNKS, document_order())
    assert [w.how for w in history["b"]] == ["enacted", "words substituted"]
    assert "but prior to January 1, 2020, up to the total" in history["b"][-1].text
    assert history["d"][-1].text.endswith("for the year up to December 31, 2019;")
    assert [w.how for w in history["f"]] == ["added", "words substituted"]
    assert history["f"][0].document_id == ACT10
    assert history["f"][-1].document_id == "DOC-IRA-AMEND-45-2022"
    assert history["g"][0].chunk_id == "CHUNK-IRA-AMEND-10-2021-S055-2-P2"  # where (g) is printed
    assert len(history["c"]) == 1  # rent relief never amended


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Rs. 700,000 for each year of assessment, but prior to January 1, 2020, up to the total",
         [("cap", 700_000, None, date(2020, 1, 1))]),
        ("Rs. 1,500,000 for each year of assessment, up to the total of the interest income for "
         "the year up to December 31, 2019;", [("cap", 1_500_000, None, date(2020, 1, 1))]),
        ("an amount equal to 25 percent of the total rental income", [("rate", 25.0, None, None)]),
        ("expenditure up to a total sum of Rs. 1,200,000, incurred for a year of assessment on or "
         "after January 1, 2020, but prior to April 1, 2022 and sum of Rs. 900,000, incurred for "
         "the first nine months of the year of assessment commencing on April 1, 2022: -",
         [("cap", 1_200_000, date(2020, 1, 1), date(2022, 4, 1)),
          ("cap", 1_200_000, date(2022, 4, 1), date(2023, 1, 1))]),  # 900,000 for 9 months
    ],
)  # fmt: skip
def test_values_and_periods_are_read_from_the_wording(text, expected):
    got = [(r.kind, r.value, r.start, r.end) for r in read_wording(text)]
    assert got == expected


def test_a_new_subparagraph_without_a_date_starts_on_its_table_a_date():
    assert operation_date(CHUNKS, ACT10, "g") == date(2021, 4, 1)
    assert operation_date(CHUNKS, ACT10, "b") is None  # not in Table A: certified date applies


def test_relief_timelines_with_withdrawn_and_new_reliefs():
    versions = build_versions(CHUNKS + SLAB_CHUNKS)
    got = {
        key: [(v.value_kind, v.value, v.valid_from, v.valid_to, v.status, v.source_document_id)
              for v in versions[key]]
        for key in ("employment_relief", "rental_relief", "senior_citizen_interest_relief",
                    "qualifying_expenditure_relief", "solar_panel_relief")
    }  # fmt: skip
    start = date(2018, 4, 1)
    assert got == {
        "employment_relief": [("cap", 700_000, start, date(2019, 12, 31), "outdated",
                               "DOC-IRA-2017")],
        "rental_relief": [("rate", 25.0, start, None, "active", "DOC-IRA-2017")],
        "senior_citizen_interest_relief": [("cap", 1_500_000, start, date(2019, 12, 31),
                                            "outdated", "DOC-IRA-2017")],
        "qualifying_expenditure_relief": [("cap", 1_200_000, date(2020, 1, 1),
                                           date(2022, 12, 31), "outdated", ACT10)],
        "solar_panel_relief": [("cap", 600_000, date(2021, 4, 1), None, "active", ACT10)],
    }  # fmt: skip
    solar = versions["solar_panel_relief"][0]
    assert "solar panels to fix on his premises" in solar.matched_text  # verbatim evidence
    assert versions["personal_relief"][-1].value == 1_800_000  # personal relief unchanged


@pytest.fixture(scope="module")
def data():
    chunks = CHUNKS + SLAB_CHUNKS
    extracted = [
        c.to_dict() | {"entities": [asdict(e) for e in extract_entities(c.text)]} for c in chunks
    ]
    rows = graph_rows(extracted, build_versions(chunks), load_registry().documents)
    return graph_backend.from_rows(rows)


def _ask(data, text=None, day="2025-06-01", **kw):
    req = RetrievalRequest(schema_version="1.0.0", request_id="t", query_text=text,
                           as_of_date=date.fromisoformat(day), **kw)  # fmt: skip
    out = graph_backend.retrieve(req, data)
    assert contract_errors(to_contract_json(out), OUTPUT_SCHEMA) == []
    return out


def test_a_named_relief_returns_only_that_relief_with_a_cap(data):
    out = _ask(data, "Can I claim relief for solar panels?")
    assert [r.rule_id for r in out.results] == ["RULE-SOLAR_RELIEF-V1"]
    r = out.results[0]
    assert r.values.cap.amount == 600_000 and r.values.amount is None
    assert "connected to the national grid" in r.text
    assert "paragraph 2(g)" in r.source.citation


def test_rent_relief_is_a_rate(data):
    (r,) = _ask(data, "rent relief").results
    assert r.values.rate_percent == 25.0
    assert "para 2(c)" in r.source.citation


def test_a_relief_not_yet_in_force_is_refused_with_the_reason(data):
    out = _ask(data, "solar panel relief", day="2020-06-01")
    assert out.results == []
    assert out.warnings == ["NO_RULE_IN_FORCE_ON_DATE"]  # not "before the corpus": 2020 is in it
    hist = _ask(data, "solar panel relief", day="2020-06-01", include_history=True)
    assert hist.results[0].validity.valid_from == date(2021, 4, 1)


def test_a_withdrawn_relief_is_not_answered_with_personal_relief(data):
    out = _ask(data, "What relief does a senior citizen get on interest income?")
    assert out.results == [] and "NO_RULE_IN_FORCE_ON_DATE" in out.warnings
    (r,) = _ask(data, "senior citizen interest relief", day="2019-06-01").results
    assert r.values.cap.amount == 1_500_000 and r.validity.status == "outdated"


def test_bare_relief_lists_every_relief_in_force_personal_first(data):
    out = _ask(data, "What reliefs can I claim?", day="2021-06-01")
    keys = [r.rule_key for r in out.results]
    assert keys[0] == "personal_relief"
    assert {"qualifying_expenditure_relief", "solar_panel_relief", "rental_relief"} <= set(keys)


def test_component1_query_still_gets_exactly_personal_relief_and_bands(data):
    question = "Personal relief and income tax rates for a resident individual's employment income"
    out = _ask(
        data,
        question,
        day="2025-04-01",
        income_types=["employment"],
        taxpayer_category="resident_individual",
        rule_types=["relief", "tax_slab"],
    )
    assert {r.rule_key for r in out.results} == {"personal_relief", "individual_income_tax_slabs"}


@pytest.mark.parametrize(
    "question, day, rule_id",
    [  # everyday wording (found with the plain-language stress test)
        ("How much can I earn without paying any tax this year?", "2025-06-01",
         "RULE-PERSONAL_RELIEF-V4"),
        ("My father is 65 and earns FD interest. Was there a relief for him in 2019?",
         "2019-06-01", "RULE-SENIOR_CITIZEN_RELIEF-V1"),
        ("I work freelance for a US company and get paid in dollars. Any relief in 2019?",
         "2019-06-01", "RULE-FOREIGN_SERVICE_RELIEF-V1"),
    ],
)  # fmt: skip
def test_plain_words_find_the_relief(data, question, day, rule_id):
    assert _ask(data, question, day).results[0].rule_id == rule_id


def test_working_for_a_company_is_still_a_question_about_an_individual(data):
    assert _ask(data, "I am employed at a software company. What are my tax bands?").results
    for question in ("What is the income tax rate for companies?", "How much tax does my "
                     "company pay?", "Tax rate for a partnership?"):  # fmt: skip
        assert _ask(data, question).results == [], question


def test_a_relief_is_returned_only_when_the_question_asks_about_one(data):
    out = _ask(data, "What is the tax on lottery winnings?")
    assert not any(r.rule_type == "relief" for r in out.results)
