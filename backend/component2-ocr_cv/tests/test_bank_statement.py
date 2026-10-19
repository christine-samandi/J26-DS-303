"""Why: bank statements now return issuer, statement date, statement month and closing balance,
and the closing balance is verified with the statement's own arithmetic. These tests use a FAKE
statement (made-up numbers) with the noise seen on real phone photos of Sri Lankan statements."""
from dataclasses import dataclass
from datetime import date

from entity_extraction.bank_statement import parse_date
from entity_extraction.extractor import Line, split_amount
from entity_extraction.payslip import extract_document


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows, conf=0.9):
    toks = []
    for i, (label, amount) in enumerate(rows):
        y = 100 + 40 * i
        for j, w in enumerate(label.split()):
            toks.append(Tok(w, conf, [20 + 90 * j, y, 80, 20]))
        for k, w in enumerate(amount.split()):
            toks.append(Tok(w, conf, [700 + 110 * k, y, 100, 20]))
    return toks


def statement(closing="13,000.00", withdrawals="2,000.00"):
    return [("HATTON NATIONAL BANK PLC", ""), ("STATEMENT OF ACCOUNT", ""), ("DATE 31/03/2025", ""),
            ("BALANCE AS OF 28FEB25", "10,000.00"), ("01MAR25 DEPOSIT", "5,000.00 15,000.00"),
            ("15MAR25 ATM WITHDRAWAL", "2,000.00 13,000.00"), ("31MAR25 CLOSING BALANCE", closing),
            ("TOTAL DEPOSITS 1 ITEMS", "5,000.00"), ("TOTAL WITHDRAWALS 1 ITEMS", withdrawals)]


def fields(out):
    return {f.field_name: f for f in out.extracted_fields}


def test_statement_fields_and_verified_closing_balance():
    out, report = extract_document(page(statement()), doc_id="BS_FAKE")
    by = fields(out)
    assert out.document_type == "bank_statement"
    assert by["issuer"].value == "Hatton National Bank PLC"
    assert by["document_date"].value == "2025-03-31" and by["pay_period"].value == "2025-03"
    assert by["total_amount"].value == 13000.0 and by["total_amount"].tax_entity == "DOCUMENT_TOTAL"
    hard = [c for c in report["checks"] if c["hard"]]
    assert hard and all(c["passed"] for c in hard)


def test_misread_closing_balance_is_corrected_and_flagged():
    out, report = extract_document(page(statement(closing="18,000.00")), doc_id="BS_FAKE")   # 3 read as 8
    assert fields(out)["total_amount"].value == 13000.0 and fields(out)["total_amount"].needs_review
    assert report["corrections"]["total_amount"] == {"ocr_value": 18000.0, "corrected_value": 13000.0}


def test_amounts_without_cents_are_not_used_as_totals():
    out, report = extract_document(page(statement(withdrawals="2,000")), doc_id="BS_FAKE")
    assert not any(c["hard"] for c in report["checks"])          # flow rule needs all four totals
    assert fields(out)["total_amount"].value == 13000.0


def test_dates_as_printed_on_statements():
    assert parse_date("DATE 29/07/24") == date(2024, 7, 29)
    assert parse_date("BALANCE AS OF 29FEB24") == date(2024, 2, 29)
    assert parse_date("17 June 2025") == date(2025, 6, 17)
    assert parse_date("2024-07-29") == date(2024, 7, 29)


def test_table_rule_noise_after_amount_is_ignored():
    toks = [Tok(t, 0.9, [i * 60, 0, 50, 20]) for i, t in enumerate(["CLOSING", "BALANCE", "26", "215", "08", "if", "'"])]
    label, amount = split_amount(Line(toks))
    assert [t.text for t in amount] == ["26", "215", "08"]


# --- Layout without a 'closing balance' line (real Bank of Ceylon statement, made-up numbers) -----
NO_CLOSING_LINE = [("Date : 10/21/25", ""), ("310725 Balance B/F", "115.17"),
                   ("110825 BOOKED SAVINGS DEP", "5,000 00 5,115 17"), ("120825 POS CARGILLS - KOTHALAWA", "90 00 5,025 17"),
                   ("200825 ATM SERVICE CHARGE", "5 00 5,020 17"), ("*** END OF STATEMENT ***", "")]


def test_closing_balance_from_last_row_and_month_from_transactions():
    out, report = extract_document(page(NO_CLOSING_LINE + [("SAVINGS STATEMENT DEPOSIT WITHDRAWAL", "")]), doc_id="BS_F2")
    by = fields(out)
    assert out.document_type == "bank_statement"
    assert by["total_amount"].value == 5020.17 and by["total_amount"].needs_review
    assert by["document_date"].value == "2025-10-21"          # printed month-first
    assert by["pay_period"].value == "2025-08"                # transactions, not the print date
    assert "issuer" not in by                                  # a Cargills SHOP is not Cargills Bank
    assert any(c["rule"].startswith("previous balance") and c["passed"] for c in report["checks"])


def test_amounts_split_at_the_decimal_point():
    from entity_extraction.bank_statement import row_amounts
    toks = [Tok(t, 0.9, [0, 0, 1, 1]) for t in "1,000 00 4,118 17".split()]
    assert row_amounts(toks) == [1000.0, 4118.17]
