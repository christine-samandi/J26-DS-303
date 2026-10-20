"""Why: personal identifiers must be masked right after OCR (NFR4) WITHOUT damaging the tax fields.
All values here are made up; the patterns mirror real Sri Lankan payslips and bank statements."""
from dataclasses import dataclass

from privacy import mask_tokens


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows):
    toks = []
    for i, row in enumerate(rows):
        for j, w in enumerate(row.split()):
            toks.append(Tok(w, 0.9, [20 + 80 * j, 100 + 40 * i, 70, 20], f"1_1_{i}"))
    return toks


def visible(tokens):
    return " ".join(t.text for t in tokens)


def test_nic_old_and_new_are_masked():
    masked, rep = mask_tokens(page(["NIC No : 851234567V", "ID 199012345678"]))
    assert "851234567V" not in visible(masked) and "199012345678" not in visible(masked)
    assert rep.counts["nic"] >= 2


def test_labelled_name_and_account_are_masked_amounts_kept():
    rows = ["Name : Kamala Perera", "Account No : 1234567890", "Net Salary 121,180.82", "Total Deductions 339,605.42"]
    masked, rep = mask_tokens(page(rows))
    text = visible(masked)
    assert "Kamala" not in text and "Perera" not in text and "1234567890" not in text
    assert "121,180.82" in text and "339,605.42" in text and "Net" in text      # tax data untouched
    assert "name" in rep.counts and ({"account_no", "long_id"} & set(rep.counts))   # caught by either rule


def test_sri_lankan_initials_name_even_with_broken_title():
    masked, rep = mask_tokens(page(["- BR 9 T SAMPLENAME Date : 10/21/25"]))       # 'MR J T' misread by OCR
    text = visible(masked)
    assert "SAMPLENAME" not in text and "10/21/25" in text and "Date" in text


def test_address_block_masked_bank_name_kept():
    rows = ["COMMERCIAL BANK OF CEYLON PLC", "STATEMENT OF ACCOUNT", "SAMPLE MOTORS", "NO 12/3/4",
            "GREEN CITY", "LAKE ROAD", "TOWNVILLE", "DATE 29/07/24"]
    masked, rep = mask_tokens(page(rows))
    text = visible(masked)
    assert "COMMERCIAL" in text and "CEYLON" in text and "29/07/24" in text
    for secret in ("SAMPLE", "MOTORS", "GREEN", "LAKE", "TOWNVILLE"):
        assert secret not in text
    assert rep.counts["address"] >= 3


def test_phone_and_repeated_place_name():
    rows = ["Mrs W.A.D. Sample", "No. 5 Temple Road", "Kandyville", "Tel : 077 123 4567", "Branch Kandyville ATM 1,000.00"]
    masked, rep = mask_tokens(page(rows))
    text = visible(masked)
    assert "077" not in text and "4567" not in text and "Kandyville" not in text
    assert "1,000.00" in text


def test_amounts_dates_and_short_numbers_are_never_masked():
    rows = ["29JUL24 CLOSING BALANCE 26,215.08", "120825 POS 330.00 9,130.17", "Gross 460,786.24", "PAGE 1"]
    masked, rep = mask_tokens(page(rows))
    assert visible(masked) == visible(page(rows)) and not rep.counts


def test_report_has_kinds_and_counts_only():
    masked, rep = mask_tokens(page(["NIC 851234567V"]))
    d = rep.to_dict()
    assert d["masked"] == {"nic": 1} and "851234567V" not in str(d)
