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


# --- the real printed layout ----------------------------------------------------------------
# Measured on the seven Inland Revenue Acts in the corpus (A4, points):
#   running header at y~167 (page number + short title), not in the top 8% of the paper;
#   odd pages: text column x=144..384, marginal notes on the RIGHT at x=396 (8pt);
#   even pages: text column x=211..451, marginal notes on the LEFT at x=139 (8pt);
#   a table of contents before "BE it enacted"; schedules printed in 8pt; printer codes.

REAL_TOP = 167
ODD = {"left": 144, "indent": 156, "note": 396}
EVEN = {"left": 211, "indent": 223, "note": 139}


def _real_page(doc, number: int | None):
    page = doc.new_page(width=W, height=H)
    if number is not None:
        side = ODD if number % 2 else EVEN
        page.insert_text((side["left"] + 230, REAL_TOP), str(number), fontsize=10)
        page.insert_text(
            (side["left"] + 40, REAL_TOP), "Test Revenue Act, No. 7 of 2030", fontsize=10
        )
    return page


def _lines(page, x: float, y: float, rows: list[str], size: float = 10, step: float = 12) -> float:
    for row in rows:
        page.insert_text((x, y), row, fontsize=size)
        y += step
    return y


def build_real_layout_act(path: Path) -> Path:
    """A 6-page Act laid out like the real ones: cover, table of contents, mirrored body pages
    with notes on alternating sides, chapter headings, schedules with a rate table."""
    import pymupdf

    doc = pymupdf.open()

    # page 1: cover, title over two lines in different sizes
    p = _real_page(doc, None)
    p.insert_text((166, 282), "PARLIAMENT OF THE DEMOCRATIC", fontsize=15)
    p.insert_text((260, 300), "SRI LANKA", fontsize=15)
    p.insert_text((180, 360), "TEST REVENUE (AMENDMENT)", fontsize=12)
    p.insert_text((220, 378), "ACT, No. 7 OF 2030", fontsize=13)
    p.insert_text((226, 420), "[Certified on 5th of May, 2030]", fontsize=10)

    # page 2: table of contents; every line looks like a section start
    p = _real_page(doc, None)
    _lines(p, 235, 190, ["ARRANGEMENT OF SECTIONS"])
    y = 214
    for number, title, at in [
        (1, "Short title.", 1),
        (2, "Charging provision.", 1),
        (3, "Test bodies.", 2),
        (4, "Interpretation.", 2),
    ]:
        p.insert_text((156, y), f"{number}.", fontsize=10)
        p.insert_text((181, y), title, fontsize=10)
        p.insert_text((443, y), str(at), fontsize=10)
        y += 12
    _lines(p, 186, y + 12, ["FIRST SCHEDULE", "THIRD SCHEDULE"])

    # page 3 (odd): enacting words, Part / Chapter / Division headings, sections 1-2
    p = _real_page(doc, 1)
    x = ODD["left"]
    _lines(p, 190, 191, ["[Certified on 5th of May, 2030]"])
    _lines(p, 165, 215, ["AN ACT TO PROVIDE FOR A TEST TAX"])
    _lines(
        p,
        x,
        239,
        [
            "BE it enacted by the Parliament of the Democratic",
            "Socialist Republic of Sri Lanka as follows:-",
        ],
    )
    _lines(p, 255, 275, ["PART I"])
    _lines(p, 245, 299, ["CHAPTER I", "IMPOSITION OF TEST TAX"])
    _lines(p, 200, 335, ["Division I: Test Income and", "Test Gains"])
    p.insert_text((ODD["indent"], 371), "1.", fontsize=10)
    p.insert_text((x + 30, 371), "This Act may be cited as the Test Revenue", fontsize=10)
    _lines(p, x, 383, ["Act, No. 7 of 2030 and comes into operation."])
    _lines(p, ODD["note"], 371, ["Short title."], size=8, step=9)
    p.insert_text((ODD["indent"], 407), "2.", fontsize=10)
    p.insert_text((x + 30, 407), "(1) Test tax shall be payable for each year of", fontsize=10)
    _lines(
        p,
        x,
        419,
        [
            "assessment by a person who has taxable income.",
            "(2) The rates are set out in the First Schedule.",
        ],
    )
    _lines(p, ODD["note"], 407, ["Charging", "provision."], size=8, step=9)
    _lines(p, x, 683, ["1 PL 001234-1,000 (01/2030)"], size=8)

    # page 4 (even): notes on the LEFT, column shifted right, mostly indented text
    p = _real_page(doc, 2)
    x = EVEN["left"]
    p.insert_text((EVEN["indent"], 191), "3.", fontsize=10)
    p.insert_text((x + 30, 191), "A test body shall be exempt where -", fontsize=10)
    _lines(
        p,
        x + 61,
        215,
        [
            "(a) it is registered with the Commissioner-",
            "General; and",
            "(b) it files a test return for the year.",
        ],
    )
    _lines(p, EVEN["note"], 191, ["Non-", "Governmental", "test bodies."], size=8, step=9)
    p.insert_text((EVEN["indent"], 263), "4.", fontsize=10)
    p.insert_text((x + 30, 263), "In this Act, “test” means a test.", fontsize=10)
    _lines(p, EVEN["note"], 263, ["Interpretation."], size=8, step=9)

    # page 5 (odd): FIRST SCHEDULE in 8pt, "(Section 2)" on the heading row, a rate table
    p = _real_page(doc, 3)
    p.insert_text((301, 194), "FIRST SCHEDULE", fontsize=8)
    p.insert_text((416, 194), "(Section 2)", fontsize=8)
    _lines(p, 318, 213, ["TAX RATES"], size=8)
    _lines(p, 228, 232, ["1. Tax rates for test individuals."], size=8)
    _lines(
        p,
        216,
        251,
        ["(1) The taxable income of an individual shall be taxed at the", "following rates:"],
        size=8,
        step=10,
    )
    for i, (band, rate) in enumerate(
        [
            ("Not exceeding Rs. 100", "6% of the amount"),
            ("Exceeding Rs. 100", "Rs. 6 plus 12% of the"),
        ]
    ):
        p.insert_text((223, 290 + 22 * i), band, fontsize=8)
        p.insert_text((363, 290 + 22 * i), rate, fontsize=8)
    _lines(p, 216, 350, ["2. Tax rates for test companies are set out in this paragraph."], size=8)

    # page 6 (even): "(Section 9)" printed ABOVE the heading; a schedule with no numbered paragraphs
    p = _real_page(doc, 4)
    p.insert_text((418, 192), "(Section 9)", fontsize=8)
    p.insert_text((300, 205), "THIRD SCHEDULE", fontsize=8)
    _lines(
        p,
        216,
        242,
        [
            "The following shall be exempt amounts:-",
            "(a) amounts derived by a test body;",
            "(b) amounts derived by a test fund.",
        ],
        size=8,
        step=10,
    )
    doc.save(path)
    doc.close()
    return path
