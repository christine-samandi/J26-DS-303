"""Generate small PDFs laid out like Sri Lankan Acts, for parser tests.

Layout copied from the printed Acts: a title page with the "[Certified on ...]" line, then
pages with a running title at the top, a page number at the bottom, numbered sections in one
body column, marginal notes in the right margin, and schedules at the end. The text is
invented test content, not quoted law.
"""

from __future__ import annotations

from pathlib import Path

W, H = 595, 842  # A4 in points
BODY_X = 100
NOTE_X = 470
LINE = 14


def _page(doc, running_title: str | None, number: int | None):
    page = doc.new_page(width=W, height=H)
    if running_title:
        page.insert_text((BODY_X, 40), running_title, fontsize=9)
    if number is not None:
        page.insert_text((W / 2, 815), str(number), fontsize=9)
    return page


def build_amendment_act(path: Path) -> Path:
    """A 3-page amendment Act: title page + 2 body pages with sections 1-4."""
    import pymupdf

    title = "Inland Revenue (Amendment) Act, No. 2 of 2025"
    doc = pymupdf.open()

    p = _page(doc, None, None)
    for i, text in enumerate(
        [
            "PARLIAMENT OF THE DEMOCRATIC SOCIALIST",
            "REPUBLIC OF SRI LANKA",
            "INLAND REVENUE (AMENDMENT) ACT, No. 2 OF 2025",
            "[Certified on 20th of March, 2025]",
            "Printed on the Order of Government",
        ]
    ):
        p.insert_text((BODY_X, 200 + i * 30), text, fontsize=12)

    body = [
        # (text, margin note or None)
        ("AN ACT TO AMEND THE INLAND REVENUE ACT, No. 24 OF 2017", None),
        ("BE it enacted by the Parliament of the Democratic Socialist Republic of", None),
        ("Sri Lanka as follows:-", None),
        ("1. (1) This Act may be cited as the Inland Revenue (Amendment) Act,", "Short title"),
        ("No. 2 of 2025.", None),
        ("(2) The provisions of this Act shall come into operation on April 1,", None),
        ("2025.", None),
        ("2. Section 150 of the Inland Revenue Act, No. 24 of 2017 (hereinafter", "Amendment of"),
        ("referred to as the principal enactment) is hereby amended as fol-", None),
        ("lows: the test amount shall be increased.", None),
        ("3. The First Schedule to the principal enactment is hereby amended", "Amendment of"),
        ("by the insertion of the following new subparagraph:-", None),
        ('"(1D) the test rate shall be six per centum on the first', None),
        ("1,000,000 rupees of taxable income;", None),
        ("2025. This line looks like a section number but is out of order.", None),
    ]
    p = _page(doc, title, 1)
    _write_body(p, body, [("Short title and", 0), ("date of operation.", 1)], start_y=90)
    p = _page(doc, title, 2)
    body2 = [
        ("4. The Fifth Schedule to the principal enactment is hereby amended in", "Amendment of"),
        ("paragraph 2 by substituting a new test amount.", None),
    ]
    _write_body(p, body2, [], start_y=90)
    doc.save(path)
    doc.close()
    return path


def _write_body(page, rows, extra_notes, start_y):
    y = start_y
    for text, note in rows:
        page.insert_text((BODY_X, y), text, fontsize=10)
        if note == "Amendment of":
            page.insert_text((NOTE_X, y), "Amendment of", fontsize=7)
            page.insert_text((NOTE_X, y + 9), "the Act.", fontsize=7)
        elif note == "Short title":
            for label, offset in extra_notes:
                page.insert_text((NOTE_X, y + offset * 9), label, fontsize=7)
        y += LINE


def build_principal_act_with_schedule(path: Path) -> Path:
    """A 2-page principal-Act extract: sections 1-2, then FIFTH SCHEDULE paragraphs 1-2."""
    import pymupdf

    title = "Inland Revenue Act, No. 24 of 2017"
    doc = pymupdf.open()
    p = _page(doc, title, 1)
    rows = [
        ("1. This Act may be cited as the Inland Revenue Act, No. 24 of 2017.", None),
        ("2. (1) Income tax shall be charged for each year of assessment on the", None),
        ("taxable income of every person.", None),
        ("(2) The test rates are set out in the First Schedule.", None),
        ("(3) This subsection is a test of the third subsection.", None),
    ]
    _write_body(p, rows, [], start_y=90)
    p = _page(doc, title, 2)
    rows = [
        ("FIFTH SCHEDULE", None),
        ("1. Test reliefs for charitable institutions.", None),
        ("2. (a) a personal relief of a test amount for each year of", None),
        ("assessment;", None),
        ("(b) a test relief for employment income.", None),
    ]
    _write_body(p, rows, [], start_y=90)
    doc.save(path)
    doc.close()
    return path
