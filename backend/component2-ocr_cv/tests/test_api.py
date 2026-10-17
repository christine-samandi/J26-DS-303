"""Why: the API is the only thing other components touch; these tests make sure every endpoint
answers with the shared contract, and that bad uploads get a clear error instead of a crash."""
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[1] / "sample_documents"
REPO = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO / "contracts/schemas/ocr_output_schema.json").read_text(encoding="utf-8"))
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")


def upload(name="SYN_payslip_clean.png"):
    return {"file": (name, (SAMPLES / name).read_bytes(), "image/png")}


@needs_tesseract
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "eng" in r.json()["languages"]


@needs_tesseract
def test_extract_returns_contract_json():
    r = client.post("/extract", params={"lang": "eng", "doc_id": "SYN_001"}, files=upload())
    assert r.status_code == 200, r.text
    data = r.json()
    jsonschema.validate(data, SCHEMA)
    by = {f["field_name"]: f["value"] for f in data["extracted_fields"]}
    assert by["gross_income"] == 150000.0 and by["net_amount"] == 134500.0
    assert data["doc_id"] == "SYN_001" and data["document_type"] == "payslip"


@needs_tesseract
def test_extract_explain_includes_checks():
    r = client.post("/extract/explain", params={"lang": "eng"}, files=upload())
    assert r.status_code == 200
    exp = r.json()["explanation"]
    assert any(c["rule"].startswith("net") and c["passed"] for c in exp["checks"])
    assert "deskew" in exp["preprocessing"] or "resize" in exp["preprocessing"]


@needs_tesseract
def test_extract_visual_returns_png():
    r = client.post("/extract/visual", params={"lang": "eng"}, files=upload())
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


@needs_tesseract
def test_unreadable_file_is_rejected():
    r = client.post("/extract", params={"lang": "eng"}, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and "not a readable image" in r.json()["detail"]


@needs_tesseract
def test_empty_file_is_rejected():
    r = client.post("/extract", params={"lang": "eng"}, files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 415


@needs_tesseract
def test_missing_language_is_explained():
    r = client.post("/extract", params={"lang": "eng+xyz"}, files=upload())
    assert r.status_code == 400 and "xyz" in r.json()["detail"]


def test_openapi_lists_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/extract", "/extract/explain", "/extract/visual"} <= set(paths)
