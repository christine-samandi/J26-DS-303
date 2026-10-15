"""Phase 3: rule versioning and the knowledge-graph rows, on wording taken from the Acts.

No Neo4j needed: the graph is built as plain rows first, and `load` is checked with a fake
driver that records the Cypher it is given.
"""

from datetime import date

import pytest

pytest.importorskip("spacy")

from entity_recognition.chunking import Chunk  # noqa: E402
from entity_recognition.extract import extract_entities  # noqa: E402
from knowledge_graph.build import (  # noqa: E402
    WRITES,
    amended_provision,
    graph_rows,
    load,
    resolve_reference,
)
from knowledge_graph.versions import (  # noqa: E402
    RULES,
    Statement,
    build_timeline,
    build_versions,
    relief_statements,
    score,
    slab_statements,
)


def _c(cid, doc, text, section=None, schedule=None, paragraph=None, note=None):
    seg = (
        cid.replace("CHUNK-", "SEG-").rsplit("-", 1)[0]
        if section
        else cid.replace("CHUNK-", "SEG-")
    )
    return Chunk(cid, doc, seg, "section" if section else "schedule_paragraph", section, None,
                 schedule, paragraph, 1, None, note, 1, 1, text)  # fmt: skip


FIFTH = "Amendment of the Fifth Schedule to the principal enactment"
FIRST = "Amendment of the First Schedule to the principal enactment"

# Fifth Schedule para 2(a) in each Act (abridged from the official text)
RELIEF_CHUNKS = [
    _c("CHUNK-IRA-2017-SCH5-P002", "DOC-IRA-2017",
       "The reliefs referred to in section 52 shall be as follows: (a) Rs. 500,000 for each year "
       "of assessment, except that an individual who is a trustee ...; (b) in the case of an "
       "individual with income from employment, Rs. 700,000 for each year of assessment;",
       schedule="FIFTH SCHEDULE", paragraph="2"),
    _c("CHUNK-IRA-AMEND-10-2021-S055-2", "DOC-IRA-AMEND-10-2021",
       "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) of that "
       "paragraph and the substitution therefor, of the following subparagraph: - “(a) (i) "
       "Rs. 500,000, for each year of assessment prior to January 1, 2020; and (ii) Rs. "
       "3,000,000, for each year of assessment commencing on or after January 1, 2020, except "
       "that an individual ...;”; (e) “(f) in the case of a resident individual, following "
       "expenditure up to a total sum of Rs. 1,200,000, incurred for a year of assessment”",
       section="55", note=FIFTH),
    _c("CHUNK-IRA-AMEND-45-2022-S036-2", "DOC-IRA-AMEND-45-2022",
       "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) ...: - “(a) "
       "(i) Rs. 500,000, for each year of assessment prior to January 1, 2020; (ii) Rs. "
       "3,000,000, for each year of assessment commencing on or after January 1, 2020, but "
       "prior to April 1, 2022; (iii) Rs. 2,250,000, for first nine months and Rs. 300,000 for "
       "second three months of the year of assessment commencing on April 1, 2022; and (iv) "
       "Rs. 1,200,000, for each year of assessment commencing on or after April 1, 2023, "
       "except that ...;”", section="36", note=FIFTH),
    _c("CHUNK-IRA-AMEND-02-2025-S005-0", "DOC-IRA-AMEND-02-2025",
       "The Fifth Schedule to the principal enactment is hereby amended in subparagraph (a) of "
       "paragraph 2 as follows: -", section="5", note=FIFTH),
    _c("CHUNK-IRA-AMEND-02-2025-S005-3", "DOC-IRA-AMEND-02-2025",
       "(3) by the addition immediately after item (iv) of that subparagraph, of the following "
       "new item: - “(v) Rs. 1,800,000, for each year of assessment commencing on or after "
       "April 1, 2025,”.", section="5", note=FIFTH),
]  # fmt: skip

T17 = ("Not Exceeding Rs. 600,000 | 4% of the amount in excess of Rs. 0 Exceeding Rs. 600,000 "
       "but not exceeding Rs. 1,200,000 | Rs. 24,000 plus 8% of the amount in excess of "
       "Rs. 600,000 Exceeding Rs. 1,200,000 | Rs. 72,000 plus 12% of the amount")  # fmt: skip

SLAB_CHUNKS = [
    _c("CHUNK-IRA-2017-SCH1-P001-1", "DOC-IRA-2017",
       "(1) Subject to the provisions of subparagraph (2), the taxable income of a resident or "
       "non-resident individual for a year of assessment shall be taxed at the following "
       f"rates: Taxable Income | Tax Payable {T17}:", schedule="FIRST SCHEDULE", paragraph="1"),
    _c("CHUNK-IRA-AMEND-10-2021-S051-1", "DOC-IRA-AMEND-10-2021",
       "(1) in paragraph 1 of that Schedule- (b) ... “(1A) subject to the provisions of "
       "subparagraph (2), the taxable income of a resident or non-resident individual commencing "
       "from January 1, 2020 shall be taxed at the following rates: - Taxable income | Tax "
       "payable Not exceeding Rs. 3 million | 6% of the amount in excess of Rs.0 Exceeding "
       "Rs. 3 million | Rs. 180,000 plus 12% of the amount in excess of Rs. 3 million”; "
       "(iii) “(iii) on or after January 1, 2020: Total income from employment | Tax payable "
       "Not exceeding Rs. 10,000,000 | 0% of the amount in excess of Rs.0”",
       section="51", note=FIRST),
    _c("CHUNK-IRA-AMEND-45-2022-S033-1", "DOC-IRA-AMEND-45-2022",
       "“(1B) Subject to the provisions of subparagraph (2), the taxable income of a resident or "
       "nonresident individual for the year of assessment commencing from April 1, 2022 shall "
       "be taxed at the following rates: - (a) Taxable income for the first nine months period "
       "of the year of assessment commencing from April 1, 2022:- Taxable Income | Tax payable "
       "Not exceeding Rs. 2,250,000 | 6% of the amount in excess of Rs. 0 Exceeding Rs. "
       "2,250,000 | Rs. 135,000 plus 12% of the amount; (b) Taxable income for the second three "
       "months period of the year of assessment commencing from April 1, 2022:- Taxable Income "
       "| Tax payable Not exceeding Rs. 125,000 | 6% of the amount in excess of Rs.0 Exceeding "
       "Rs. 125,000 | Rs. 7,500 plus 12% of the amount”", section="33", note=FIRST),
]  # fmt: skip


def test_relief_history_with_a_split_year_and_later_acts_overriding():
    order = {"DOC-IRA-2017": 0, "DOC-IRA-AMEND-10-2021": 1, "DOC-IRA-AMEND-45-2022": 2,
             "DOC-IRA-AMEND-02-2025": 3}  # fmt: skip
    statements = relief_statements(RELIEF_CHUNKS)
    assert 700_000 not in {s.value for s in statements}  # para 2(b), not 2(a)
    assert 1_200_000 in {s.value for s in statements if s.document_id.endswith("45-2022")}
    versions = build_timeline(statements, RULES["personal_relief"], order, date(2018, 4, 1))
    got = [(v.value, str(v.valid_from), str(v.valid_to), v.source_document_id) for v in versions]
    assert got == [
        (500_000, "2018-04-01", "2019-12-31", "DOC-IRA-2017"),
        (3_000_000, "2020-01-01", "2022-12-31", "DOC-IRA-AMEND-10-2021"),
        # Rs. 300,000 for the last three months of 2022/23 = Rs. 1,200,000 a year from 1 Jan
        (1_200_000, "2023-01-01", "2025-03-31", "DOC-IRA-AMEND-45-2022"),
        (1_800_000, "2025-04-01", "None", "DOC-IRA-AMEND-02-2025"),
    ]
    assert [v.status for v in versions] == ["replaced"] * 3 + ["active"]
    assert versions[1].supersedes == versions[0].rule_id == "RULE-PERSONAL_RELIEF-V1"
    assert versions[0].superseded_by == "RULE-PERSONAL_RELIEF-V2"


def test_evidence_text_is_verbatim_from_the_chunk():
    by_id = {c.chunk_id: c.text for c in RELIEF_CHUNKS + SLAB_CHUNKS}
    for s in relief_statements(RELIEF_CHUNKS) + slab_statements(SLAB_CHUNKS):
        assert s.matched_text in by_id[s.chunk_id], s.matched_text


def test_tax_tables_including_an_annualised_split_year():
    statements = slab_statements(SLAB_CHUNKS)
    tables = [(s.document_id[-7:], s.start, s.end, s.value) for s in statements]
    assert tables[0] == ("RA-2017", None, None, ((0, 600_000, 4.0), (600_000, 1_200_000, 8.0),
                                                  (1_200_000, None, 12.0)))  # fmt: skip
    assert tables[1][1:] == (date(2020, 1, 1), None, ((0, 3_000_000, 6.0), (3_000_000, None, 12.0)))
    # the employment-income table in the same chunk is a different rule: not read
    assert len([t for t in tables if t[0] == "10-2021"]) == 1
    # 9 months at Rs. 2,250,000 = Rs. 3,000,000 a year; 3 months at Rs. 125,000 = 500,000
    assert tables[2][1:] == (date(2022, 4, 1), date(2023, 1, 1),
                             ((0, 3_000_000, 6.0), (3_000_000, None, 12.0)))  # fmt: skip
    assert tables[3][1:] == (date(2023, 1, 1), date(2023, 4, 1),
                             ((0, 500_000, 6.0), (500_000, None, 12.0)))  # fmt: skip


def test_same_value_restated_by_a_later_act_is_one_version():
    s = [
        Statement("personal_relief", 5, None, None, "DOC-A", "CHUNK-A", "x"),
        Statement("personal_relief", 5, date(2020, 1, 1), None, "DOC-B", "CHUNK-B", "y"),
    ]
    (v,) = build_timeline(s, RULES["personal_relief"], {"DOC-A": 0, "DOC-B": 1}, date(2018, 4, 1))
    assert (v.valid_from, v.valid_to, v.source_document_id) == (date(2018, 4, 1), None, "DOC-A")
    assert v.evidence_chunk_ids == ["CHUNK-A", "CHUNK-B"]


def test_versioning_score_against_ground_truth(tmp_path):
    (tmp_path / "personal_relief_versions.csv").write_text(
        "version,amount_lkr,valid_from,valid_to,source_document_id\n"
        "V1,500000,2018-04-01,2019-12-31,DOC-IRA-2017\n"
        "V2,3000000,2020-01-01,,DOC-IRA-AMEND-10-2021\n"
    )
    (tmp_path / "individual_tax_bands.csv").write_text(
        "version,valid_from,valid_to,band_order,lower_lkr,upper_lkr,rate_percent,"
        "source_document_id\nB1,2018-04-01,,1,0,,4,DOC-IRA-2017\n"
    )
    order = {"DOC-IRA-2017": 0, "DOC-IRA-AMEND-10-2021": 1}
    relief = build_timeline(relief_statements(RELIEF_CHUNKS[:2]), RULES["personal_relief"], order,
                            date(2018, 4, 1))  # fmt: skip
    result = score({"personal_relief": relief, "individual_income_tax_slabs": []}, tmp_path)
    assert result["personal_relief"]["exact"] == 2
    assert result["individual_income_tax_slabs"]["checks"][0]["wrong"] == ["missing"]
    assert result["ALL"]["accuracy"] == round(2 / 3, 4)


# --- graph --------------------------------------------------------------------------------


def _as_extracted(chunk: Chunk) -> dict:
    from dataclasses import asdict

    return chunk.to_dict() | {"entities": [asdict(e) for e in extract_entities(chunk.text)]}


def test_references_resolve_to_the_right_act():
    amend = {"document_id": "DOC-IRA-AMEND-10-2021"}
    own = {"document_id": "DOC-IRA-2017"}
    assert resolve_reference("Section 52 of the principal enactment", amend) == "SEG-IRA-2017-S052"
    assert resolve_reference("section 52", own) == "SEG-IRA-2017-S052"
    assert resolve_reference("section 84A", own) == "SEG-IRA-2017-S084A"
    assert (
        resolve_reference("paragraph 2(a) of the Fifth Schedule to the principal enactment", amend)
        == "SEG-IRA-2017-SCH5-P002"
    )
    assert resolve_reference("subsection (2) of section 140", own) == "SEG-IRA-2017-S140"
    assert resolve_reference("item (iii) of that subparagraph", own) is None


def test_amended_provision_from_marginal_note_and_lead():
    leads = {("DOC-IRA-AMEND-02-2025", "5"): RELIEF_CHUNKS[3].text}
    c = _as_extracted(RELIEF_CHUNKS[4])
    assert amended_provision(c, leads) == "SEG-IRA-2017-SCH5-P002"
    c = {"marginal_note": "Amendment of section 52 of the principal enactment", "text": "",
         "document_id": "DOC-X", "section": "5"}  # fmt: skip
    assert amended_provision(c, {}) == "SEG-IRA-2017-S052"


def test_graph_rows_link_versions_to_their_evidence(monkeypatch):
    import knowledge_graph.versions as v

    monkeypatch.setattr(v, "corpus_start", lambda: date(2018, 4, 1))
    chunks = RELIEF_CHUNKS + SLAB_CHUNKS
    # a 2017 chunk for section 52, so "section 52" references have a target
    s52 = _c("CHUNK-IRA-2017-S052-2", "DOC-IRA-2017", "(2) ... the aggregate reliefs ...",
             section="52")  # fmt: skip
    versions = build_versions(chunks)
    rows = graph_rows([_as_extracted(c) for c in chunks + [s52]], versions, [])
    assert {r["rule_id"] for r in rows["versions"]} >= {"RULE-PERSONAL_RELIEF-V4"}
    chunk_ids = {r["chunk_id"] for r in rows["chunks"]}
    assert all(r["chunk_id"] in chunk_ids for r in rows["defined_in"])
    assert {"chunk_id": "CHUNK-IRA-2017-SCH5-P002", "segment_id": "SEG-IRA-2017-S052",
            "text": "section 52"} in rows["refers_to"]  # fmt: skip
    amends = {(r["chunk_id"], r["segment_id"]) for r in rows["amends"]}
    assert ("CHUNK-IRA-AMEND-02-2025-S005-3", "SEG-IRA-2017-SCH5-P002") in amends
    assert any(b["rule_id"].startswith("RULE-INDIVIDUAL_SLABS") for b in rows["bands"])
    assert {"rule_id": "RULE-PERSONAL_RELIEF-V1", "category": "resident_individual"} in rows[
        "applies_to"
    ]


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, **params):
        self.calls.append((query, params))


def test_load_writes_every_row_group_in_batches():
    rows = {key: [{"n": i} for i in range(3)] for key, _ in WRITES}
    driver = _FakeDriver()
    load(driver, rows, batch=2)
    assert driver.calls[0][0] == "MATCH (n) DETACH DELETE n"
    unwinds = [p["rows"] for q, p in driver.calls if q.startswith("UNWIND")]
    assert len(unwinds) == 2 * len(WRITES)  # 3 rows in batches of 2
    assert sum(len(r) for r in unwinds) == 3 * len(WRITES)
