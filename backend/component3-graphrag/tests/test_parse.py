"""Phase 1: PDF layout + segmentation on generated PDFs laid out like Sri Lankan Acts."""

import json

import pytest

pytest.importorskip("pymupdf")

from act_pdf_fixtures import (  # noqa: E402
    build_amendment_act,
    build_principal_act_with_schedule,
    build_real_layout_act,
    build_ruled_rate_table,
)

from data_collection.download import register_local  # noqa: E402
from data_collection.layout import _dedup, classify, read_pages  # noqa: E402
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


# --- the real printed layout (mirrored pages, table of contents, 8pt schedules) -----------


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    path = build_real_layout_act(tmp_path_factory.mktemp("pdf") / "real.pdf")
    return parse_pdf(path, "DOC-TEST-7-2030")


def test_table_of_contents_is_not_cut_into_sections(real):
    s1 = _seg(real, "SEG-TEST-7-2030-S001")
    assert s1.text.startswith("This Act may be cited") and s1.page_start == 3
    assert [s.number for s in real.segments if s.kind == "section"] == ["1", "2", "3", "4"]
    assert "ARRANGEMENT OF SECTIONS" not in " ".join(s.text for s in real.segments)


def test_two_line_title_in_different_font_sizes(real):
    assert real.title_detected == "TEST REVENUE (AMENDMENT) ACT, No. 7 OF 2030"
    assert real.certified_date_detected == "2030-05-05"


def test_notes_found_on_both_sides_of_mirrored_pages(real):
    notes = {s.number: s.marginal_note for s in real.segments if s.kind == "section"}
    assert notes == {
        "1": "Short title.",  # odd page, right margin
        "2": "Charging provision.",
        "3": "Non-Governmental test bodies.",  # even page, left margin; hyphen kept before capital
        "4": "Interpretation.",
    }


def test_running_header_and_printer_code_are_not_text(tmp_path):
    pages = read_pages(build_real_layout_act(tmp_path / "real.pdf"))
    classify(pages)
    kinds = {}
    for p in pages:
        for ln in p.lines:
            kinds.setdefault(ln.text, set()).add(ln.kind)
    assert kinds["Test Revenue Act, No. 7 of 2030"] == {"header"}  # at y=167 on every page
    assert kinds["1 PL 001234-1,000 (01/2030)"] == {"footer"}


def test_part_chapter_division_become_the_heading(real):
    s1 = _seg(real, "SEG-TEST-7-2030-S001")
    assert (
        s1.heading
        == "PART I > CHAPTER I IMPOSITION OF TEST TAX > Division I: Test Income and Test Gains"
    )
    text = " ".join(s.text for s in real.segments)
    assert "CHAPTER I" not in text and "Test Gains" not in text


def test_schedule_heading_with_section_reference_and_rate_table(real):
    intro = _seg(real, "SEG-TEST-7-2030-SCH1")
    assert intro.kind == "schedule_text" and intro.text == "FIRST SCHEDULE (Section 2) TAX RATES"
    p1 = _seg(real, "SEG-TEST-7-2030-SCH1-P001")
    rates = p1.subsections[0].text
    assert "Not exceeding Rs. 100 6% of the amount" in rates  # 8pt table cells are not notes
    assert _seg(real, "SEG-TEST-7-2030-SCH1-P002").page_start == 5


def test_schedule_without_numbered_paragraphs_is_kept(real):
    third = _seg(real, "SEG-TEST-7-2030-SCH3")
    assert third.kind == "schedule_text"
    assert third.text.startswith("(Section 9) THIRD SCHEDULE The following shall be exempt")
    assert "(b) amounts derived by a test fund." in third.text
    assert "(Section 9)" not in _seg(real, "SEG-TEST-7-2030-SCH1-P002").text


def test_section_number_drawn_twice_is_read_once():
    spans = [
        {"text": "5.", "bbox": (126.5, 0, 134, 10)},
        {"text": "5. Section 52 of the principal enactment", "bbox": (126.5, 0, 300, 10)},
    ]
    assert "".join(s["text"] for s in _dedup(spans)) == "5. Section 52 of the principal enactment"
    kept = [
        {"text": "4.", "bbox": (126.5, 0, 134, 10)},
        {"text": " Section 46", "bbox": (134, 0, 200, 10)},
    ]
    assert _dedup(kept) == kept


def test_ruled_table_is_read_row_by_row(tmp_path):
    pages = read_pages(build_ruled_rate_table(tmp_path / "table.pdf"))
    rows = [ln.text for ln in pages[0].lines if "|" in ln.text]
    assert rows == [
        "Taxable Income | Tax Payable",
        "Not exceeding Rs. 100 | 6% of the amount in excess of Rs. 0",
        "Exceeding Rs. 100 but not exceeding Rs. 200 | "
        "Rs. 6 plus 12% of the amount in excess of Rs. 100",
    ]


def test_table_reading_can_be_switched_off(tmp_path):
    pages = read_pages(build_ruled_rate_table(tmp_path / "table.pdf"), tables=False)
    assert not any("|" in ln.text for ln in pages[0].lines)


def test_reference_at_line_start_is_not_a_subsection():
    # Act No. 11 of 2026, s.5: "... in terms of subsection" / "(1) of this section, is not possible"
    text = "Where the deduction in terms of subsection\n(1) of this section, is not possible"
    assert split_subsections(text) == (" ".join(text.split()), [])
    lead, subs = split_subsections("(1) First.\n(2) of that paragraph applies.\n(2) Second.")
    assert [s.label for s in subs] == ["1", "2"] and subs[1].text == "Second."
