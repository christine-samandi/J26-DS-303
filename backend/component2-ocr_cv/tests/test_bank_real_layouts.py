"""Why: three real statements (Commercial Bank photos, a Bank of Ceylon scan) showed: a misread
account-holder name leaking through raw_text of the date, the last transaction row split into an
'amounts' line and a 'date + description' line (closing balance taken from the row above), and a
damaged bank name returned as raw OCR text. Made-up numbers, no personal data."""
from dataclasses import dataclass

import numpy as np

from entity_extraction.payslip import extract_document
from preprocessing import preprocess


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows, conf=0.92):
    toks = []
    for i, cells in enumerate(rows):
        y = 100 + 40 * i
        for x, text in cells:
            for w in text.split():
                toks.append(Tok(w, conf, [x, y, 12 * len(w) + 4, 20]))
                x += 12 * len(w) + 14
    return toks


def boc_statement(split_last_row=True):
    rows = [[(40, "BANK OF CEYLON")], [(40, "MR J T SAMPLENAME"), (500, "Date : 10/21/25")],
            [(40, "310725"), (140, "Balance B/F"), (900, "115.17")],
            [(40, "110825"), (140, "BOOKED SAVINGS DEP"), (700, "5,000.00"), (900, "5,115.17")],
            [(40, "200825"), (140, "ATMCASHWDL"), (560, "700.00"), (900, "31.93")]]
    if split_last_row:
        rows += [[(560, "5.00"), (900, "26.93")], [(40, "200825"), (140, "ATM SERVICE CHARGE")]]
    else:
        rows += [[(40, "200825"), (140, "ATM SERVICE CHARGE"), (560, "5.00"), (900, "26.93")]]
    rows += [[(200, "*** END OF STATEMENT ***")]]
    return page(rows)


def test_split_last_row_still_gives_the_closing_balance():
    out, report = extract_document(boc_statement(), "T")
    f = {x.field_name: x for x in out.extracted_fields}
    assert out.document_type == "bank_statement"
    assert f["total_amount"].value == 26.93                     # not 31.93 from the row above
    assert any(c["rule"].startswith("previous balance") and c["passed"] for c in report["checks"])


def test_date_raw_text_never_carries_the_rest_of_the_line():
    out, _ = extract_document(boc_statement(), "T")
    f = {x.field_name: x for x in out.extracted_fields}
    assert f["document_date"].value == "2025-10-21"
    assert "SAMPLENAME" not in f["document_date"].raw_text and "MR" not in f["document_date"].raw_text


def test_bank_from_registration_number_and_damaged_name():
    rows = [[(40, "STATEMENT OF ACCOUNT")], [(600, "Soe COMMERCIAL Ban OF CeVLON Plc")],
            [(600, "Reg.No.: PQ116")], [(40, "DATE 29/07/24")],
            [(40, "BALANCE AS OF 29FEB24"), (900, "26,215.08")],
            [(40, "29JUL24"), (140, "DEPOSIT"), (700, "200,000.00"), (900, "226,215.08")],
            [(40, "29JUL24"), (140, "CHEQUE"), (500, "200,000.00"), (900, "26,215.08")],
            [(40, "29JUL24 CLOSING BALANCE"), (900, "26,215.08")]]
    out, _ = extract_document(page(rows), "T")
    f = {x.field_name: x for x in out.extracted_fields}
    assert f["issuer"].value == "Commercial Bank of Ceylon PLC"          # official name, not the OCR line


def test_large_photo_is_shrunk_before_denoising():
    big = np.full((3072, 4080), 255, np.uint8)
    res = preprocess(big)
    assert res.steps_applied[:2] == ["resize", "denoise"] and max(res.image.shape) <= 2100
