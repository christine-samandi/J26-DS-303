"""Why: a real Sinhala payslip photo exposed three linked problems.
1. The binarised reading lost every decimal point and the boxed header rows (so the NIC and the
   name were never read, never masked, and still visible in /extract/visual).
2. On the greyscale re-reading, 'ඊ.පී.එෆ්.' / 'ඊ.ටී.එෆ්.' were garbled, so both employer lines
   matched only the shared phrase 'සංස්ථා දායකත්වය' and EPF was reported as ETF.
3. With masking on, a masked personal line could be chosen as the issuer.
These tests use FAKE tokens only (no real document)."""
from dataclasses import dataclass

from api.pipeline import looks_broken, reading_quality
from entity_extraction.extractor import extract_fields, find_issuer, group_lines


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
            toks.append(Tok(w, conf, [20 + 120 * j, y, 110, 20]))
        for k, w in enumerate(amount.split()):
            toks.append(Tok(w, conf, [700 + 70 * k, y, 65, 20]))
    return toks


def test_shared_contribution_label_larger_amount_is_epf_employer():
    found = extract_fields(page([("6.8.03. සංස්ථා දායකන්වය", "28,775.29"),     # garbled ඊ.පී.එෆ්.
                                 ("රී.වි.එන්. සංස්ථා දායකන්වය", "5,755.06")]))    # garbled ඊ.ටී.එෆ්.
    assert found["epf_employer"].value == 28775.29
    assert found["etf"].value == 5755.06
    assert any("shared by EPF and ETF" in n for n in found["epf_employer"].notes)


def test_single_shared_contribution_label_stays_etf():
    found = extract_fields(page([("රී.වි.එන්. සංස්ථා දායකත්වය", "5,755.06")]))
    assert found["etf"].value == 5755.06 and "epf_employer" not in found


def test_clear_epf_label_is_not_overridden_by_twin_rule():
    found = extract_fields(page([("ඊ.පී.එෆ්. සංස්ථා දායකත්වය", "28,775.29"),
                                 ("ඊ.ටී.එෆ්. සංස්ථා දායකත්වය", "5,755.06")]))
    assert found["epf_employer"].value == 28775.29 and found["etf"].value == 5755.06


def test_issuer_never_taken_from_masked_or_personal_line():
    lines = group_lines(page([("අංකය : ●●●●● ●●●●●●", ""), ("නම : ●●●●● ●●●●●●", ""),
                              ("SAMPLE HOLDINGS (PVT) LTD", "")]))
    issuer = find_issuer(lines)
    assert issuer is not None and "●" not in issuer.value and "LTD" in issuer.value


def test_reading_without_decimal_points_looks_broken():
    lost = page([("Gross", "179,045 79"), ("Net", "121,180 82"), ("EPF", "28,775 29")])
    report = {"checks": [{"rule": "net = gross - deductions", "hard": True, "passed": True}]}
    q = reading_quality(lost, report, 3)
    assert q[2] == 0.0 and looks_broken(q)


def test_clean_reading_passes_self_check_and_better_reading_wins():
    clean = page([("Gross", "179,045.79"), ("Net", "121,180.82")])
    passed = {"checks": [{"rule": "net = gross - deductions", "hard": True, "passed": True}]}
    good = reading_quality(clean, passed, 5)
    assert not looks_broken(good)
    assert good > reading_quality(clean, {"checks": []}, 5)     # a passing hard check outranks
