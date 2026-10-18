"""Why: benchmark numbers go on PP1 slides, so the scoring rules themselves must be correct.
Covers field statuses, precision/recall/F1, calibration, the review flag and a full run on the
synthetic payslip in a temporary data folder (never the real data/)."""
import csv
import json
import shutil
from pathlib import Path

import pytest

from scripts.benchmark import main, tesseract_lang
from scripts.metrics import (FieldResult, cer_wer, expected_calibration_error, precision_recall_f1,
                             review_quality, score_fields, values_match)

COMPONENT = Path(__file__).resolve().parents[1]


def test_values_match_rules():
    assert values_match("gross_income", 150000.0, 150000.00)
    assert not values_match("gross_income", 150000.0, 15000.0)          # one digit off is wrong
    assert values_match("issuer", "LANKA TEXTILES PVT LTD", "LANKA TEXTILES (PVT) LTD")
    assert not values_match("issuer", "Commercial Bank", "LANKA TEXTILES (PVT) LTD")
    assert values_match("pay_period", "2026-09", "2026-09") and not values_match("pay_period", "2026-08", "2026-09")


def test_field_statuses_and_scores():
    expected = {"gross_income": 150000.0, "net_amount": 134500.0, "apit": 3500.0, "etf": None}
    predicted = {"gross_income": (150000.0, 0.98, False),   # correct
                 "net_amount": (184500.0, 0.90, True),      # wrong
                 "etf": (4500.0, 0.70, True)}                # extra (not on the document)
    res = {r.field: r.status for r in score_fields(predicted, expected)}
    assert res == {"gross_income": "correct", "net_amount": "wrong", "apit": "missed", "etf": "extra"}
    s = precision_recall_f1(score_fields(predicted, expected))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 2)
    assert s["precision"] == pytest.approx(1 / 3, abs=1e-3) and s["recall"] == pytest.approx(1 / 3, abs=1e-3)


def test_calibration_and_review_flag():
    assert expected_calibration_error([(0.99, True)] * 10) == pytest.approx(0.01, abs=1e-3)
    assert expected_calibration_error([(0.99, False)] * 10) == pytest.approx(0.99, abs=1e-3)
    assert expected_calibration_error([]) is None
    q = review_quality([FieldResult("a", "correct", 0.99, False), FieldResult("b", "wrong", 0.99, False),
                        FieldResult("c", "wrong", 0.6, True), FieldResult("d", "missed")])
    assert q == {"trusted_correct": 1, "trusted_wrong": 1, "flagged_correct": 0, "flagged_wrong": 1}


def test_cer_wer():
    assert cer_wer("", "anything") == (None, None)
    cer, wer = cer_wer("Net Salary 134,500.00", "Net Salary 134,500.00")
    assert cer == 0 and wer == 0
    cer, _ = cer_wer("Net Salary 134,500.00", "Net Sa1ary 184,500.00")
    assert 0 < cer < 0.2


def test_language_from_manifest():
    installed = ["eng", "sin", "tam", "osd"]
    assert tesseract_lang("si+en", installed) == "sin+eng"
    assert tesseract_lang("ta", installed) == "tam+eng"
    assert tesseract_lang("", installed) == "eng"
    assert tesseract_lang("si", ["eng"]) == "eng"                       # missing pack -> falls back


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_benchmark_end_to_end_on_synthetic_payslip(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "ground_truth").mkdir()
    shutil.copy(COMPONENT / "sample_documents" / "SYN_payslip_clean.png", data / "images" / "PS_001_digital.png")
    with open(data / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_file", "doc_id", "doc_type", "source", "language", "capture",
                                           "device", "split", "notes"])
        w.writeheader()
        w.writerow({"image_file": "PS_001_digital.png", "doc_id": "PS_001", "doc_type": "payslip",
                    "source": "synthetic", "language": "en", "capture": "digital", "device": "", "split": "test",
                    "notes": ""})
    gt = {"doc_id": "PS_001", "document_type": "payslip", "source": "synthetic", "languages": ["en"],
          "fields": {"issuer": "LANKA TEXTILES (PVT) LTD", "pay_period": "2026-09", "gross_income": 150000.0,
                     "epf_employee": 12000.0, "epf_employer": None, "etf": None, "apit": 3500.0,
                     "total_deductions": 15500.0, "net_amount": 134500.0},
          "full_text": "", "checked": True}
    (data / "ground_truth" / "PS_001.json").write_text(json.dumps(gt), encoding="utf-8")

    out = tmp_path / "out"
    assert main(["--data", str(data), "--out", str(out), "--modes", "pre"]) == 0
    rows = list(csv.DictReader(open(out / "per_image.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["type_ok"] == "True"
    assert int(rows[0]["fields_correct"]) >= 6
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("| engine")
