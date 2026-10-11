"""Phase 1: PDF layout + segmentation on generated PDFs laid out like Sri Lankan Acts."""

import json

import pytest

pytest.importorskip("pymupdf")

from act_pdf_fixtures import build_amendment_act, build_principal_act_with_schedule  # noqa: E402

from data_collection.download import register_local  # noqa: E402
from data_collection.layout import classify, read_pages  # noqa: E402
from data_collection.parse import parse_corpus, parse_pdf  # noqa: E402
from data_collection.segment import join_rows, split_subsections  # noqa: E402


@pytest.fixture(scope="module")
def amendment(tmp_path_factory):
    path = build_amendment_act(tmp_path_factory.mktemp("pdf") / "amend.pdf")
    return parse_pdf(path, "DOC-IRA-AMEND-02-2025")


@pytest.fixture(scope="module")
def principal(tmp_path_factory):
    path = build_principal_act_with_schedule(tmp_path_factory.mktemp("pdf") / "principal.pdf")
    return parse_pdf(path, "DOC-IRA-2017")


def _seg(parsed, segment_id):
    return next(s for s in parsed.segments if s.segment_id == segment_id)


# --- layout -------------------------------------------------------------------------------


def test_running_title_and_page_numbers_are_removed(tmp_path):
    pages = read_pages(build_amendment_act(tmp_path / "a.pdf"))
    classify(pages)
    kinds = {(ln.kind, ln.text) for p in pages for ln in p.lines}
    assert ("header", "Inland Revenue (Amendment) Act, No. 2 of 2025") in kinds
    assert ("footer", "1") in kinds
    body = " ".join(ln.text for p in pages for ln in p.lines if ln.kind == "body")
    assert "Inland Revenue (Amendment) Act, No. 2 of 2025 1" not in body


def test_margin_notes_are_separated_from_body(tmp_path):
    pages = read_pages(build_amendment_act(tmp_path / "a.pdf"))
    classify(pages)
    margin = [ln.text for p in pages for ln in p.lines if ln.kind == "margin"]
    assert "Short title and" in margin and "Amendment of" in margin


# --- metadata -----------------------------------------------------------------------------


def test_title_and_certified_date_detected(amendment):
    assert amendment.title_detected == "INLAND REVENUE (AMENDMENT) ACT, No. 2 OF 2025"
    assert amendment.certified_date_detected == "2025-03-20"
    assert amendment.page_count == 3


# --- sections -----------------------------------------------------------------------------


def test_sections_found_in_order(amendment):
    sections = [s.number for s in amendment.segments if s.kind == "section"]
    assert sections == ["1", "2", "3", "4"]


def test_preamble_holds_title_page_and_enacting_words(amendment):
    pre = amendment.segments[0]
    assert pre.kind == "preamble" and pre.segment_id == "SEG-IRA-AMEND-02-2025-PRE"
    assert "BE it enacted" in pre.text


def test_out_of_order_number_is_not_a_new_section(amendment):
    s3 = _seg(amendment, "SEG-IRA-AMEND-02-2025-S003")
    assert "2025. This line looks like a section number" in s3.text


def test_quoted_new_subparagraph_stays_inside_its_section(amendment):
    s3 = _seg(amendment, "SEG-IRA-AMEND-02-2025-S003")
    assert '"(1D) the test rate' in s3.text and s3.subsections == []


def test_hyphenated_line_break_is_rejoined(amendment):
    s2 = _seg(amendment, "SEG-IRA-AMEND-02-2025-S002")
    assert "amended as follows:" in s2.text


def test_subsections_split(amendment):
    s1 = _seg(amendment, "SEG-IRA-AMEND-02-2025-S001")
    assert [sub.label for sub in s1.subsections] == ["1", "2"]
    assert s1.subsections[1].text.endswith("on April 1, 2025.")


def test_margin_notes_attached_to_sections(amendment):
    assert _seg(amendment, "SEG-IRA-AMEND-02-2025-S001").marginal_note == (
        "Short title and date of operation."
    )
    assert _seg(amendment, "SEG-IRA-AMEND-02-2025-S004").marginal_note == "Amendment of the Act."


def test_page_numbers_recorded_for_citation(amendment):
    s4 = _seg(amendment, "SEG-IRA-AMEND-02-2025-S004")
    assert (s4.page_start, s4.page_end) == (3, 3)


def test_schedule_paragraphs(principal):
    paras = [s for s in principal.segments if s.kind == "schedule_paragraph"]
    assert [s.segment_id for s in paras] == ["SEG-IRA-2017-SCH5-P001", "SEG-IRA-2017-SCH5-P002"]
    assert all(s.schedule == "FIFTH SCHEDULE" for s in paras)
    assert "personal relief" in paras[1].text


def test_three_subsections_in_principal_act(principal):
    s2 = _seg(principal, "SEG-IRA-2017-S002")
    assert [sub.label for sub in s2.subsections] == ["1", "2", "3"]


# --- helpers ------------------------------------------------------------------------------


def test_join_rows_keeps_hyphens_before_capitals():
    assert join_rows(["the Inland-", "Revenue Act"]) == "the Inland- Revenue Act"
    assert join_rows(["amend-", "ment"]) == "amendment"


def test_split_subsections_ignores_out_of_order_markers():
    lead, subs = split_subsections("(1) first part\n(3) skipped two\n(2) second part")
    assert [s.label for s in subs] == ["1", "2"]


# --- end to end: manifest -> JSON ---------------------------------------------------------


def test_parse_corpus_writes_json_and_checks_registry(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    pdf = build_amendment_act(tmp_path / "amend.pdf")
    register_local("DOC-IRA-AMEND-02-2025", pdf, raw)
    [parsed] = parse_corpus(raw, out)
    assert parsed.warnings == []  # certified date and Act number match the registry
    data = json.loads((out / "DOC-IRA-AMEND-02-2025.json").read_text(encoding="utf-8"))
    assert data["sha256"] == parsed.sha256
    assert data["segments"][1]["segment_id"] == "SEG-IRA-AMEND-02-2025-S001"


def test_registry_mismatch_is_warned(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    pdf = build_amendment_act(tmp_path / "amend.pdf")
    register_local("DOC-IRA-AMEND-45-2022", pdf, raw)  # wrong document on purpose
    [parsed] = parse_corpus(raw, out)
    assert any("certified date" in w for w in parsed.warnings)
    assert any("does not match Act No. 45 of 2022" in w for w in parsed.warnings)
