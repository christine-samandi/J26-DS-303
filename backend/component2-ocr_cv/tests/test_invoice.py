"""Why: invoices, receipts and quotations carry the evidence for tax reliefs (a solar installation, a
donation). These tests rebuild FOUR real layouts as OCR tokens (a UK-style invoice, an Indian GST
solar quotation, a US solar contractor estimate) plus Sri Lankan English/Sinhala documents, and check
that the reading is faithful, that the document's own arithmetic is checked, and that one misread
digit is repaired only when the arithmetic points to exactly one value. All numbers are made up or
come from public templates; no personal data."""
from dataclasses import dataclass

import jsonschema
import json
from pathlib import Path

from document_classifier import classify_tokens
from entity_extraction.invoice import parse_date, words_to_number
from entity_extraction.invoice_checks import digit_close
from entity_extraction.payslip import extract_document

SCHEMA = json.loads((Path(__file__).resolve().parents[3] / "contracts/schemas/ocr_output_schema.json")
                    .read_text(encoding="utf-8"))


@dataclass
class Tok:
    text: str
    confidence: float
    bbox: list
    line_id: str = ""


def page(rows, conf=0.93):
    """rows: list of rows; a row is a list of (x, text) cells, or ('BIG', x, text) for a heading."""
    toks, y = [], 40
    for row in rows:
        for cell in row:
            big = cell[0] == "BIG"
            x, text = (cell[1], cell[2]) if big else cell
            h, cw = (40, 20) if big else (20, 9)
            for w in text.split():
                width = cw * len(w) + 4
                toks.append(Tok(w, conf, [x, y, width, h]))
                x += width + 8
        y += 60 if any(c[0] == "BIG" for c in row) else 34
    return toks


def run(rows, conf=0.93):
    out, report = extract_document(page(rows, conf), doc_id="T")
    jsonschema.validate(json.loads(out.model_dump_json(exclude_none=True)), SCHEMA)
    fields = {f.field_name: f for f in out.extracted_fields}
    return out, report, fields


def check(report, rule):
    return next(c for c in report["checks"] if c["rule"] == rule)


# --- 1. UK-style invoice template: Rate before Quantity, '£' amounts, terms in days ---------------
UK_INVOICE = [
    [("BIG", 60, "Invoice")],
    [(60, "To:"), (560, "Your Company Name")],
    [(60, "Client Name"), (560, "Company Address")],
    [(560, "Company Registration No:"), (860, "If Applicable")],
    [(200, "Invoice Number"), (480, "0"), (560, "Tax Date"), (900, "1/1/2011")],
    [(200, "PO Number"), (480, "0"), (560, "Terms"), (900, "28 Days")],
    [(200, "Reference"), (480, "0"), (560, "Payment Due By"), (900, "29/1/2011")],
    [(60, "Description"), (240, "Rate"), (520, "Quantity"), (760, "Gross")],
    [(60, "Sample Item 1"), (240, "£10.00"), (520, "1.00"), (760, "£10.00")],
    [(60, "Sample Item 2"), (240, "£15.00"), (520, "2.00"), (760, "£30.00")],
    [(60, "Sample Item 3"), (240, "£22.00"), (520, "3.00"), (760, "£66.00")],
    [(560, "Gross Total"), (880, "£106.00")],
]


def test_uk_invoice_items_total_and_dates():
    out, report, f = run(UK_INVOICE)
    assert out.document_type == "invoice" and out.schema_version == "0.2"
    assert [i.description for i in out.line_items] == ["Sample Item 1", "Sample Item 2", "Sample Item 3"]
    assert [(i.quantity, i.unit_price, i.amount) for i in out.line_items] == [(1, 10, 10), (2, 15, 30), (3, 22, 66)]
    assert f["total_amount"].value == 106.0
    assert check(report, "items_sum")["passed"]
    assert f["document_date"].value == "2011-01-01" and f["due_date"].value == "2011-01-29"
    assert check(report, "due date = issue date + terms")["passed"]
    assert "seller_reg_no" not in f                     # 'If Applicable' is not a number


# --- 2. Indian GST solar quotation: amounts INCLUDE 5% GST, a column-total row, SGST + CGST -------
GST_QUOTATION = [
    [(40, "Company Name:")], [(40, "Address:")], [(40, "Phone No.:")], [(40, "GSTIN:")],
    [("BIG", 320, "Quotation")],
    [(40, "Bill To:")], [(40, "Name:")],
    [(560, "Invoice No.:"), (740, "ABC-2022-0001")],
    [(560, "Invoice Date:"), (740, "DD/MM/YYYY")],
    [(20, "#"), (60, "Items"), (220, "Type"), (400, "Qnty"), (470, "Unit"), (540, "Rate"), (640, "Dis"),
     (700, "GST"), (800, "Amount")],
    [(20, "1"), (60, "Solar Panels"), (220, "Mono-crystalline"), (400, "2.5"), (470, "W"), (540, "200.00"),
     (640, "0"), (700, "5%"), (800, "525")],
    [(20, "2"), (60, "Circuit Breakers"), (220, "Miscellaneous"), (400, "3"), (470, "Peic."), (540, "250.00"),
     (640, "0"), (700, "5%"), (800, "787.5")],
    [(20, "3"), (60, "Frames"), (220, "Mounting Structure"), (400, "1"), (470, "Panel"), (540, "340.00"),
     (640, "0"), (700, "5%"), (800, "357")],
    [(20, "4"), (60, "Invertor"), (220, "Off grid"), (400, "5"), (470, "KW"), (540, "30000.00"),
     (640, "0"), (700, "5%"), (800, "157500")],
    [(20, "5"), (60, "Batteries"), (220, "Lead Acid"), (400, "240"), (470, "AH"), (540, "300.00"),
     (640, "0"), (700, "5%"), (800, "75600")],
    [(20, "6"), (60, "Installation"), (220, "Labor Cost"), (400, "1"), (470, "Person"), (540, "300.00"),
     (640, "0"), (700, "5%"), (800, "315")],
    [(20, "7"), (60, "Wires"), (220, "Miscellaneous"), (400, "1.5"), (470, "Peic."), (540, "200.00"),
     (640, "0"), (700, "5%"), (800, "315")],
    [(60, "Total"), (400, "254"), (640, "0"), (700, "11210"), (800, "235400")],
    [(40, "Amount in words:"), (500, "Sub Total:"), (800, "224190.00")],
    [(500, "Discount:"), (800, "0")],
    [(500, "SGST"), (800, "5604.75")],
    [(500, "CGST"), (800, "5604.75")],
    [(500, "Total"), (800, "235399.5")],
    [(500, "Advance"), (800, "1000")],
    [(500, "Balance"), (800, "234399.50")],
]


def test_gst_quotation_is_a_quotation_not_an_invoice():
    cls = classify_tokens(page(GST_QUOTATION))
    assert cls.document_type == "quotation"           # title wins over 'Invoice No.' printed on the form


def test_gst_quotation_tax_inclusive_lines_and_totals():
    out, report, f = run(GST_QUOTATION)
    assert len(out.line_items) == 7
    solar = out.line_items[0]
    assert solar.description == "Solar Panels Mono-crystalline" and solar.unit == "W"
    assert (solar.quantity, solar.unit_price, solar.tax_rate, solar.amount) == (2.5, 200.0, 0.05, 525.0)
    assert all(i.amount_includes_tax for i in out.line_items)
    assert f["subtotal"].value == 224190.0
    assert f["total_amount"].value == 235399.5         # NOT the rounded 235400 of the column-total row
    assert f["amount_paid"].value == 1000.0 and f["balance_due"].value == 234399.5
    assert {(t.name, t.amount) for t in out.tax_components} == {("SGST", 5604.75), ("CGST", 5604.75)}
    for rule in ("items_sum", "items_sum_before_tax", "totals_chain", "balance", "implied tax rate"):
        assert check(report, rule)["passed"], rule
    assert f["document_number"].value == "ABC-2022-0001"
    assert "column_total_row" in report["table"]


# --- 3. US solar contractor estimate: Unit Cost | Quantity | Total Cost, printed project facts -----
ESTIMATE = [
    [(60, "SunCraft Installations"), (600, "Solar Panel Installation")],
    [(60, "123 Street, Los Angeles"), (600, "Contractor Estimate")],
    [(60, "Client Information")],
    [(60, "Client Name"), (300, "Cityville Corporation"), (600, "Contact Email"), (880, "info@mail.com")],
    [(60, "Project Location"), (300, "4174 Columbia Mine Road"), (600, "Contact Phone"), (880, "222-555-7777")],
    [(60, "Contact Person"), (300, "John Smith"), (600, "Date"), (880, "January 15, 2025")],
    [(60, "Project Details")],
    [(60, "Project Type"), (300, "Commercial Solar Panel"), (600, "Inverter Type"), (880, "String Inverters")],
    [(60, "Roof Type"), (300, "Flat Roof"), (600, "Mounting System"), (880, "Ballasted Flat Roof")],
    [(60, "System Size"), (300, "200 kw"), (600, "Expected Start Date"), (880, "March 1, 2025")],
    [(60, "Number of Panels"), (300, "600"), (600, "Expected Completion"), (880, "June 30, 2025")],
    [(60, "Cost Estimate")],
    [(60, "Description"), (560, "Unit Cost"), (690, "Quantity"), (830, "Total Cost")],
] + [[(60, d), (560, f"$ {u}"), (690, q), (830, f"$ {t}")] for d, u, q, t in [
    ("Solar Panels", "400.00", "600.00", "240,000.00"), ("Inverters", "1,200.00", "20", "24,000.00"),
    ("Mounting System", "800.00", "1", "800.00"), ("Wiring and Connectors", "5,000.00", "1", "5,000.00"),
    ("Labor", "30,000.00", "1", "30,000.00"), ("Permit Fees", "2,500.00", "1", "2,500.00"),
    ("Engineering Design", "10,000.00", "1", "10,000.00"), ("Project Management", "8,000.00", "1", "8,000.00"),
    ("Contingency", "7,500.00", "2", "15,000.00")]] + [
    [(60, "Total Estimate"), (560, "$ 65,400.00"), (690, "628.00"), (830, "$ 335,300.00")],
    [(60, "Payment Schedule")],
    [(60, "Phase"), (260, "Percentage"), (400, "Amount Due"), (600, "Due Date")],
    [(60, "Signing Contract"), (260, "20%"), (400, "$70,286.00"), (600, "February 1, 2025")],
]


def test_solar_estimate_items_attributes_and_privacy():
    out, report, f = run(ESTIMATE)
    assert out.document_type == "quotation"
    assert len(out.line_items) == 9 and out.line_items[0].description == "Solar Panels"
    assert f["total_amount"].value == 335300.0 and check(report, "items_sum")["passed"]
    attrs = {a.key: a.value for a in out.attributes}
    assert attrs["System Size"] == "200 kw" and attrs["Number of Panels"] == "600"
    assert attrs["Inverter Type"] == "String Inverters" and attrs["Expected Start Date"] == "March 1, 2025"
    assert not any(k.startswith(("Client", "Contact", "Project Location")) for k in attrs)   # personal data
    assert f["document_date"].value == "2025-01-15"
    assert f["issuer"].value == "SunCraft Installations"          # not the client's company


# --- 4. Sri Lankan solar invoice (English + Sinhala labels, 18% VAT) with one misread digit -----
def sl_solar_invoice(total="1,003,000.00"):
    return [
        [("BIG", 60, "SUNLANKA SOLAR (PVT) LTD")],
        [("BIG", 60, "TAX INVOICE / ඉන්වොයිසිය")],
        [(60, "VAT Reg No:"), (300, "114567890-7000"), (560, "Invoice No:"), (760, "SL-2026-0142")],
        [(560, "Date / දිනය:"), (760, "14/08/2026")],
        [(60, "System Capacity:"), (300, "5 kW")],
        [(60, "Description / විස්තරය"), (420, "Qty"), (520, "Unit Price"), (760, "Amount")],
        [(60, "Solar PV Panel 550W Mono"), (420, "10"), (520, "45,000.00"), (760, "450,000.00")],
        [(60, "Hybrid Inverter 5kW"), (420, "1"), (520, "285,000.00"), (760, "285,000.00")],
        [(60, "Mounting Structure & Wiring"), (420, "1"), (520, "65,000.00"), (760, "65,000.00")],
        [(60, "Installation / ස්ථාපනය"), (420, "1"), (520, "50,000.00"), (760, "50,000.00")],
        [(400, "Sub Total / උප එකතුව"), (760, "850,000.00")],
        [(400, "VAT 18%"), (760, "153,000.00")],
        [(400, "Total / මුළු මුදල"), (760, total)],
        [(400, "Amount Paid / ගෙවූ මුදල"), (760, "1,003,000.00")],
        [(400, "Balance / ශේෂය"), (760, "0.00")],
    ]


def test_sl_solar_invoice_clean():
    out, report, f = run(sl_solar_invoice())
    assert out.document_type == "invoice" and "si" in out.languages
    assert out.line_items[0].description == "Solar PV Panel 550W Mono"      # '550W' stays in the text
    assert (out.line_items[0].quantity, out.line_items[0].amount) == (10, 450000.0)
    assert f["subtotal"].value == 850000.0 and f["vat"].value == 153000.0 and f["total_amount"].value == 1003000.0
    assert f["seller_reg_no"].value == "114567890-7000" and f["document_number"].value == "SL-2026-0142"
    assert f["document_date"].value == "2026-08-14"
    assert {a.key: a.value for a in out.attributes}["System Capacity"] == "5 kW"
    vat = next(t for t in out.tax_components if t.name == "VAT")
    assert vat.rate == 0.18
    assert all(check(report, r)["passed"] for r in ("items_sum", "totals_chain", "balance", "implied tax rate"))
    assert f["total_amount"].confidence > f["total_amount"].confidence * 0 and not report["corrections"]


def test_sl_solar_invoice_misread_total_is_repaired():
    # OCR read 3 as 8: '1,008,000.00'. Totals AND balance break; changing the total by one digit
    # repairs both, while VAT or the paid amount would repair only one -> total is corrected.
    out, report, f = run(sl_solar_invoice(total="1,008,000.00"))
    assert report["corrections"]["total_amount"] == {"ocr_value": 1008000.0, "corrected_value": 1003000.0}
    assert f["total_amount"].value == 1003000.0 and f["total_amount"].needs_review   # user must confirm
    assert check(report, "totals_chain")["passed"] and check(report, "balance")["passed"]


# --- 5. Sri Lankan donation receipt, amount in words --------------------------------------------
def donation_receipt(words="Twenty Five Thousand Rupees Only"):
    return [
        [("BIG", 60, "LANKA CHILD CARE FOUNDATION")],
        [(60, "Charity Reg No:"), (300, "GA/2019/0456")],
        [("BIG", 60, "DONATION RECEIPT / පරිත්‍යාග ලදුපත")],
        [(60, "Receipt No:"), (300, "DR-2026-0318"), (560, "Date:"), (700, "05/09/2026")],
        [(60, "Purpose:"), (300, "School supplies programme")],
        [(60, "Amount / මුදල:"), (300, "Rs. 25,000.00")],
        [(60, "Amount in words:"), (300, words)],
        [(60, "Payment:"), (300, "Cash")],
    ]


def test_donation_receipt():
    out, report, f = run(donation_receipt())
    assert out.document_type == "receipt"
    assert f["total_amount"].value == 25000.0 and f["seller_reg_no"].value == "GA/2019/0456"
    assert f["document_number"].value == "DR-2026-0318"
    assert check(report, "amount in words = amount in figures")["passed"]
    assert {a.key: a.value for a in out.attributes}["Purpose"] == "School supplies programme"
    d = f["document_date"]
    assert d.value == "2026-09-05" and d.needs_review          # 05/09: day and month both <= 12


def test_donation_receipt_words_disagree():
    out, report, f = run(donation_receipt(words="Twenty Six Thousand Rupees Only"))
    assert not check(report, "amount in words = amount in figures")["passed"]
    assert f["total_amount"].needs_review


# --- helpers ------------------------------------------------------------------------------------
def test_words_to_number():
    assert words_to_number("Twenty Five Thousand Rupees Only") == 25000
    assert words_to_number("One Lakh Fifty Thousand Rupees") == 150000
    assert words_to_number("Two Hundred Thirty Five Thousand Three Hundred Ninety Nine Rupees and Fifty Cents") == 235399.5


def test_parse_date_orders():
    assert parse_date("10/14/2026 (MM/DD/YYYY)", "mdy")[0].isoformat() == "2026-10-14"
    assert parse_date("2026-10-11", None)[0].isoformat() == "2026-10-11"
    d, ambiguous, other = parse_date("05/09/2026", None)
    assert (d.isoformat(), ambiguous, other.isoformat()) == ("2026-09-05", True, "2026-05-09")
    assert parse_date("29/1/2011", None) == (parse_date("29/1/2011", "dmy")[0], False, None)


def test_digit_close():
    assert digit_close(1008000, 1003000) and digit_close(4150000, 150000)
    assert not digit_close(1008000, 1003500)
