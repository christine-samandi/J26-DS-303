"""Make SYNTHETIC purchase documents for testing the invoice/receipt/quotation extractor.

Why: no real solar-installation invoice or charity receipt was available for PP1, and real ones
carry personal data. These images follow real layouts (items table, totals block, VAT 18%, Sinhala +
English labels) with made-up companies and numbers, so the expected values are known exactly.
They are marked SYN_ and must be reported separately from real-document results.

    python scripts/make_synthetic_invoices.py            -> sample_documents/SYN_*.png + SYN_*.expected.json

Sinhala labels need a Sinhala font (Windows: Nirmala UI / Iskoola Pota) AND Pillow built with
libraqm for correct letter shaping; without them the labels are drawn in English only and the
script says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

OUT = Path(__file__).resolve().parents[1] / "sample_documents"
LATIN = ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
LATIN_BOLD = ["arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
SINHALA = ["C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/iskpota.ttf",
           "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf"]


def font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size, layout_engine=ImageFont.Layout.RAQM if features.check("raqm")
                                      else ImageFont.Layout.BASIC)
        except OSError:
            continue
    return ImageFont.load_default()


def has_sinhala() -> bool:
    return features.check("raqm") and any(Path(p).exists() for p in SINHALA)


SI = has_sinhala()


def label(en: str, si: str) -> str:
    return f"{en} / {si}" if SI and si else en


def render(name: str, rows, expected: dict):
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 70
    for row in rows:
        if row == "---":
            d.line([(60, y + 6), (W - 60, y + 6)], fill=(120, 120, 120), width=2)
            y += 24
            continue
        size, bold = (34, True) if row[0] == "BIG" else (22, False)
        cells = row[1:] if row[0] == "BIG" else row
        for x, text in cells:
            si = any("\u0d80" <= ch <= "\u0dff" for ch in text)
            f = font(SINHALA if si else (LATIN_BOLD if bold else LATIN), size)
            if si and not SI:
                continue
            d.text((x, y), text, fill="black", font=f)
        y += 64 if row[0] == "BIG" else 42
    OUT.mkdir(exist_ok=True)
    img.save(OUT / f"{name}.png")
    (OUT / f"{name}.expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(f"wrote {name}.png")


def solar_invoice(total="1,003,000.00"):
    items = [("Solar PV Panel 550W Mono", "10", "45,000.00", "450,000.00"),
             ("Hybrid Inverter 5kW", "1", "285,000.00", "285,000.00"),
             ("Mounting Structure & Wiring", "1", "65,000.00", "65,000.00"),
             (label("Installation", "ස්ථාපනය"), "1", "50,000.00", "50,000.00")]
    return [
        ("BIG", (80, "SUNLANKA SOLAR (PVT) LTD")),
        [(80, "No. 12, Galle Road, Colombo 03")],
        ("BIG", (80, label("TAX INVOICE", "ඉන්වොයිසිය"))),
        [(80, "VAT Reg No:"), (330, "114567890-7000"), (700, "Invoice No:"), (900, "SL-2026-0142")],
        [(700, label("Date", "දිනය") + ":"), (900, "14/08/2026")],
        [(80, "System Capacity:"), (330, "5 kW")],
        [(80, "Grid Connection:"), (330, "CEB Net Metering")],
        "---",
        [(80, label("Description", "විස්තරය")), (620, "Qty"), (720, "Unit Price"), (980, "Amount")],
        "---",
        *[[(80, a), (630, b), (720, c), (980, e)] for a, b, c, e in items],
        "---",
        [(560, label("Sub Total", "උප එකතුව")), (980, "850,000.00")],
        [(560, "VAT 18%"), (980, "153,000.00")],
        [(560, label("Total", "මුළු මුදල")), (960, total)],
        [(560, label("Amount Paid", "ගෙවූ මුදල")), (960, "1,003,000.00")],
        [(560, label("Balance", "ශේෂය")), (1010, "0.00")],
    ]


SOLAR_EXPECTED = {"document_type": "invoice", "issuer": "SUNLANKA SOLAR (PVT) LTD", "document_number": "SL-2026-0142",
                  "document_date": "2026-08-14", "seller_reg_no": "114567890-7000", "subtotal": 850000.0,
                  "vat": 153000.0, "total_amount": 1003000.0, "amount_paid": 1003000.0, "balance_due": 0.0,
                  "line_items": 4, "attributes": {"System Capacity": "5 kW", "Grid Connection": "CEB Net Metering"}}


def quotation():
    rows = [("Solar Panel 550W", "8", "46,000.00", "368,000.00"), ("On-grid Inverter 3kW", "1", "190,000.00", "190,000.00"),
            ("Installation and Wiring", "1", "60,000.00", "60,000.00")]
    return [
        ("BIG", (80, "GREENVOLT ENGINEERING (PVT) LTD")),
        ("BIG", (80, label("QUOTATION", "මිල ගණන් පත්‍රය"))),
        [(80, "Quotation No:"), (330, "GV-Q-2026-077"), (700, "Date:"), (900, "02/07/2026")],
        [(80, "System Size:"), (330, "4.4 kW")],
        [(80, "Valid Until:"), (330, "02/08/2026")],
        "---",
        [(80, "Description"), (620, "Qty"), (720, "Unit Price"), (980, "Amount")],
        "---",
        *[[(80, a), (630, b), (720, c), (980, e)] for a, b, c, e in rows],
        "---",
        [(560, "Sub Total"), (980, "618,000.00")],
        [(560, "VAT 18%"), (980, "111,240.00")],
        [(560, "Total"), (980, "729,240.00")],
    ]


QUOTATION_EXPECTED = {"document_type": "quotation", "subtotal": 618000.0, "vat": 111240.0, "total_amount": 729240.0,
                      "line_items": 3, "note": "a price offer, not proof of spending"}


def donation():
    return [
        ("BIG", (80, "LANKA CHILD CARE FOUNDATION")),
        [(80, "Charity Reg No:"), (380, "GA/2019/0456")],
        ("BIG", (80, label("DONATION RECEIPT", "පරිත්‍යාග ලදුපත"))),
        [(80, "Receipt No:"), (380, "DR-2026-0318"), (700, "Date:"), (850, "05/09/2026")],
        [(80, "Purpose:"), (380, "School supplies programme")],
        [(80, label("Amount", "මුදල") + ":"), (380, "Rs. 25,000.00")],
        [(80, "Amount in words:"), (380, "Twenty Five Thousand Rupees Only")],
        [(80, "Payment:"), (380, "Cash")],
    ]


DONATION_EXPECTED = {"document_type": "receipt", "total_amount": 25000.0, "seller_reg_no": "GA/2019/0456",
                     "document_number": "DR-2026-0318", "document_date": "2026-09-05 (ambiguous 05/09: flagged)",
                     "attributes": {"Purpose": "School supplies programme"}}


def main():
    if not SI:
        print("note: no Sinhala font with letter shaping (libraqm) found; Sinhala labels are left out")
    render("SYN_solar_invoice", solar_invoice(), SOLAR_EXPECTED)
    render("SYN_solar_invoice_total_error", solar_invoice(total="1,008,000.00"),
           {**SOLAR_EXPECTED, "printed_total": 1008000.0,
            "note": "total printed with one wrong digit; totals and balance checks must catch it"})
    render("SYN_solar_quotation", quotation(), QUOTATION_EXPECTED)
    render("SYN_donation_receipt", donation(), DONATION_EXPECTED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
