"""Why: the manifest and ground truth feed every accuracy number; these tests make sure file
names are parsed correctly, splits never leak a document, and bad ground truth is caught."""
import json

from scripts.dataset import (assign_splits, build_manifest, check_dataset, ground_truth_skeleton,
                             parse_image_name, payslip_arithmetic, write_manifest)


def touch(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")


def test_parse_image_name():
    p = parse_image_name("PS_001_photo_dark.jpg")
    assert (p.doc_id, p.doc_type, p.capture) == ("PS_001", "payslip", "photo_dark")
    assert parse_image_name("IN_002_digital.pdf").doc_type == "invoice"
    assert parse_image_name("PS_003_photo_good_2.jpg").capture == "photo_good"   # extra shot number
    assert parse_image_name("payslip.jpg") is None
    assert parse_image_name("PS_001_photo_dark.txt") is None


def test_split_is_per_document_and_stratified():
    docs = {f"PS_00{i}": "payslip" for i in range(1, 7)} | {"IN_001": "invoice", "IN_002": "invoice"}
    s = assign_splits(docs, {})
    assert sum(v == "test" for k, v in s.items() if k.startswith("PS")) == 2
    assert any(s[d] == "test" for d in ("IN_001", "IN_002"))          # every type has a test doc
    assert assign_splits(docs, {}) == s                                # deterministic


def test_existing_split_is_kept():
    s = assign_splits({"PS_001": "payslip", "PS_002": "payslip"}, {"PS_001": "test"})
    assert s["PS_001"] == "test"


def test_build_manifest_keeps_manual_columns(tmp_path):
    img = tmp_path / "images"
    touch(img, "PS_001_photo_good.jpg", "PS_001_photo_dark.jpg", "PS_002_scan.png", "random.jpg")
    rows, skipped, _ = build_manifest(img, [])
    assert skipped == ["random.jpg"] and len(rows) == 3
    assert len({r["split"] for r in rows if r["doc_id"] == "PS_001"}) == 1   # same split for one doc
    rows[0]["language"] = "si+en"
    rows2, _, _ = build_manifest(img, rows)
    assert next(r for r in rows2 if r["image_file"] == rows[0]["image_file"])["language"] == "si+en"


def test_skeleton_has_type_specific_fields():
    gt = ground_truth_skeleton("IN_001", "invoice")
    assert set(gt["fields"]) == {"issuer", "document_date", "vat", "sscl", "total_amount"}
    assert all(v is None for v in gt["fields"].values())


def test_payslip_arithmetic_catches_typo():
    assert payslip_arithmetic({"gross_income": 150000, "total_deductions": 15500, "net_amount": 134500}) == []
    assert payslip_arithmetic({"gross_income": 150000, "total_deductions": 15500, "net_amount": 143500})


def test_check_dataset_end_to_end(tmp_path):
    img, gt_dir, man = tmp_path / "images", tmp_path / "gt", tmp_path / "manifest.csv"
    touch(img, "PS_001_photo_good.jpg", "PS_002_photo_dark.jpg")
    rows, _, _ = build_manifest(img, [])
    for r in rows:
        r["language"] = "en"
    write_manifest(man, rows)
    gt_dir.mkdir()
    good = ground_truth_skeleton("PS_001", "payslip")
    good["fields"].update(gross_income=150000.0, total_deductions=15500.0, net_amount=134500.0)
    (gt_dir / "PS_001.json").write_text(json.dumps(good))
    rep = check_dataset(img, man, gt_dir)
    assert any("PS_002: ground truth missing" in e for e in rep.errors)

    bad = ground_truth_skeleton("PS_002", "payslip")
    bad["fields"]["gross_income"] = "150,000"                      # text instead of a number
    (gt_dir / "PS_002.json").write_text(json.dumps(bad))
    rep = check_dataset(img, man, gt_dir)
    assert any("PS_002.json: invalid" in e for e in rep.errors)

    bad["fields"]["gross_income"] = 150000.0
    (gt_dir / "PS_002.json").write_text(json.dumps(bad))
    rep = check_dataset(img, man, gt_dir)
    assert rep.ok, rep.errors
    assert rep.counts["documents by type"]["payslip"] == 2
