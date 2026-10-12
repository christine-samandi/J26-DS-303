"""Phase 1: corpus registry, downloader, manifest and link discovery (no real network)."""

import csv

import pytest

from data_collection import download as dl
from data_collection.discover import find_pdf_links
from data_collection.manifest import read_manifest, sha256_of
from data_collection.sources import load_registry

FAKE_PDF = b"%PDF-1.7\n% fake test pdf\n%%EOF\n"


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, ctype: str = "application/pdf"):
        self.content = content
        self.status_code = status
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Answers each URL from a dict; records what was requested."""

    def __init__(self, answers: dict[str, FakeResponse]):
        self.answers = answers
        self.requested: list[str] = []

    def get(self, url, timeout=None):
        self.requested.append(url)
        return self.answers.get(url, FakeResponse(b"<html>404</html>", 404, "text/html"))


def _all_ok(content: bytes = FAKE_PDF) -> FakeSession:
    return FakeSession({d.url: FakeResponse(content) for d in load_registry().documents})


# --- registry -----------------------------------------------------------------------------


def test_registry_has_the_full_act_series_in_order():
    ids = [d.document_id for d in load_registry().documents]
    assert ids == [
        "DOC-IRA-2017",
        "DOC-IRA-AMEND-10-2021",
        "DOC-IRA-AMEND-45-2022",
        "DOC-IRA-AMEND-04-2023",
        "DOC-IRA-AMEND-14-2023",
        "DOC-IRA-AMEND-02-2025",
        "DOC-IRA-AMEND-11-2026",
    ]
    dates = [d.certified_date for d in load_registry().documents]
    assert dates == sorted(dates)


def test_all_documents_come_from_the_ird_website():
    assert all(d.url.startswith("https://www.ird.gov.lk/") for d in load_registry().documents)


def test_operative_date_may_be_unknown(tmp_path):
    doc = load_registry().get("DOC-IRA-AMEND-11-2026")
    assert doc.operative_from is None
    dl.download_all(tmp_path, only=doc.document_id, session=_all_ok())
    assert read_manifest(tmp_path)[doc.document_id].operative_from == ""


def test_registry_retrospective_act_is_recorded():
    """Act 10 of 2021 was certified in 2021 but operates from 2020: both dates are kept."""
    doc = load_registry().get("DOC-IRA-AMEND-10-2021")
    assert doc.operative_from.isoformat() == "2020-01-01"
    assert doc.certified_date.isoformat() == "2021-05-13"


# --- downloader ---------------------------------------------------------------------------


def test_download_saves_pdfs_and_writes_manifest(tmp_path):
    session = _all_ok()
    outcomes = dl.download_all(tmp_path, session=session)
    n = len(load_registry().documents)
    assert [o.status for o in outcomes] == ["new"] * n
    rows = read_manifest(tmp_path)
    assert len(rows) == n
    row = rows["DOC-IRA-2017"]
    assert (tmp_path / row.filename).read_bytes() == FAKE_PDF
    assert row.sha256 == sha256_of(tmp_path / row.filename)
    assert row.method == "download" and row.size_bytes == len(FAKE_PDF)


def test_manifest_csv_has_expected_columns(tmp_path):
    dl.download_all(tmp_path, session=_all_ok())
    with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[:3] == ["document_id", "title", "document_type"]
    assert {"source_url", "sha256", "retrieved_at", "method"} <= set(header)


def test_rerun_skips_existing_files(tmp_path):
    dl.download_all(tmp_path, session=_all_ok())
    session = _all_ok()
    outcomes = dl.download_all(tmp_path, session=session)
    assert {o.status for o in outcomes} == {"unchanged"}
    assert session.requested == []


def test_force_detects_upstream_change(tmp_path):
    dl.download_all(tmp_path, session=_all_ok())
    first = read_manifest(tmp_path)["DOC-IRA-2017"]
    outcomes = dl.download_all(tmp_path, force=True, session=_all_ok(FAKE_PDF + b"% revised\n"))
    assert {o.status for o in outcomes} == {"changed"}
    assert read_manifest(tmp_path)["DOC-IRA-2017"].sha256 != first.sha256


def test_force_with_same_content_keeps_first_retrieval_time(tmp_path):
    dl.download_all(tmp_path, session=_all_ok())
    first = read_manifest(tmp_path)["DOC-IRA-2017"].retrieved_at
    dl.download_all(tmp_path, force=True, session=_all_ok())
    assert read_manifest(tmp_path)["DOC-IRA-2017"].retrieved_at == first


def test_html_instead_of_pdf_is_rejected(tmp_path):
    doc = load_registry().get("DOC-IRA-2017")
    session = FakeSession({doc.url: FakeResponse(b"<html>login</html>", ctype="text/html")})
    [outcome] = dl.download_all(tmp_path, only=doc.document_id, session=session)
    assert outcome.status == "failed" and "expected a PDF" in outcome.detail
    assert not (tmp_path / doc.filename).exists()


def test_one_failure_does_not_stop_the_rest(tmp_path):
    session = _all_ok()
    session.answers.pop(load_registry().get("DOC-IRA-AMEND-45-2022").url)
    outcomes = dl.download_all(tmp_path, session=session)
    assert [o.status for o in outcomes].count("failed") == 1
    assert len(read_manifest(tmp_path)) == len(load_registry().documents) - 1


def test_register_local_fallback(tmp_path):
    hand = tmp_path / "from-browser.pdf"
    hand.write_bytes(FAKE_PDF)
    raw = tmp_path / "raw"
    outcome = dl.register_local("DOC-IRA-AMEND-02-2025", hand, raw)
    assert outcome.status == "new"
    row = read_manifest(raw)["DOC-IRA-AMEND-02-2025"]
    assert row.method == "manual" and (raw / row.filename).is_file()


def test_register_local_rejects_non_pdf(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(dl.NotAPdfError):
        dl.register_local("DOC-IRA-2017", bad, tmp_path / "raw")


# --- link discovery (scraping) ------------------------------------------------------------

PAGE = """
<html><body>
  <a href="/en/publications/Acts_Income%20Tax_2017/IR_Act_No_24_2017_E.pdf">
    Inland Revenue Act No. 24 of 2017</a>
  <a href="docs/IR_Amendment_Act_2027_E.pdf"> Inland Revenue (Amendment) Act, 2027 </a>
  <a href="docs/VAT_Act.pdf">Value Added Tax Act</a>
  <a href="/about.html">About us</a>
  <a href="docs/IR_Amendment_Act_2027_E.pdf">duplicate link</a>
</body></html>
"""


def test_discover_finds_pdf_links_as_absolute_urls():
    links = find_pdf_links(PAGE, "https://www.ird.gov.lk/en/publications/page.html")
    urls = [lk.url for lk in links]
    assert len(urls) == 3  # html link dropped, duplicate dropped
    assert urls[1] == "https://www.ird.gov.lk/en/publications/docs/IR_Amendment_Act_2027_E.pdf"


def test_discover_marks_known_documents_and_filters_by_keyword():
    links = find_pdf_links(PAGE, "https://www.ird.gov.lk/en/publications/page.html", ["inland"])
    assert [lk.known for lk in links] == [True, False]
