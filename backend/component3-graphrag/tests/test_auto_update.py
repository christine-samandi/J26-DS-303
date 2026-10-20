"""Automatic update on the REAL Acts: a corpus that does not have Act No. 2 of 2025 yet, the
monitor finds it, and the safety checks decide. Neo4j is replaced by a function that records
what would be written, so no database is needed.
"""

import json
import shutil
from datetime import date

import pytest

pytest.importorskip("spacy")
pytest.importorskip("pymupdf")

from c3_common.config import COMPONENT_DIR, get_settings  # noqa: E402
from data_collection.manifest import read_manifest, write_manifest  # noqa: E402
from data_collection.sources import REGISTRY_PATH, load_registry  # noqa: E402
from knowledge_graph.auto_update import check_values, identify, ingest  # noqa: E402

RAW = COMPONENT_DIR / "raw_regulatory_docs"
NEW_ID = "DOC-IRA-AMEND-02-2025"
NEW_PDF = (RAW / "IRA_AMEND_02_2025_E.pdf").read_bytes() if (RAW / "IRA_AMEND_02_2025_E.pdf"
                                                             ).is_file() else None  # fmt: skip
URL = "https://www.ird.gov.lk/en/publications/Acts_Income%20Tax_2017/IR_Act_No_02-2025_E.pdf"

pytestmark = pytest.mark.skipif(NEW_PDF is None, reason="the Act PDFs are not in the checkout")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A copy of the corpus as it was before Act No. 2 of 2025, parsed and extracted."""
    from data_collection.parse import parse_corpus
    from entity_recognition.extract import extract_corpus

    root = tmp_path_factory.mktemp("corpus")
    raw, processed, extracted = root / "raw", root / "processed", root / "extracted"
    raw.mkdir()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["documents"] = [d for d in registry["documents"] if d["document_id"] != NEW_ID]
    (root / "corpus_sources.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    rows = {k: v for k, v in read_manifest(RAW).items() if k != NEW_ID}
    for row in rows.values():
        shutil.copyfile(RAW / row.filename, raw / row.filename)
    write_manifest(raw, rows)
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, root)
        parse_corpus(raw, processed)
        extract_corpus(processed, extracted)
    return root


def _point_at(mp, root):
    mp.setenv("C3_REGISTRY_PATH", str(root / "corpus_sources.json"))
    mp.setenv("C3_RAW_DOCS_DIR", str(root / "raw"))
    mp.setenv("C3_PROCESSED_DIR", str(root / "processed"))
    mp.setenv("C3_EXTRACTED_DIR", str(root / "extracted"))
    get_settings.cache_clear()
    load_registry.cache_clear()


@pytest.fixture
def env(corpus, monkeypatch, tmp_path):
    """Each test gets the 'before 2025' corpus back, so a test that adds the Act undoes it."""
    root = tmp_path / "corpus"
    shutil.copytree(corpus, root)
    _point_at(monkeypatch, root)
    monkeypatch.setenv("C3_NOTIFICATIONS_FILE", str(tmp_path / "notifications.json"))
    yield root
    get_settings.cache_clear()
    load_registry.cache_clear()


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_new_act_that_passes_every_check_is_added(env):
    written = []
    out = ingest(URL, lambda url: NEW_PDF, apply=lambda ids: written.append(ids) or ["x"] * 4)
    assert out.status == "added", out.reason
    assert [c.split(" ")[0] for c in out.checks] == ["1", "2", "3", "4", "5", "6", "7"]
    assert written == [[NEW_ID]] and out.document_id == NEW_ID
    doc = load_registry().get(NEW_ID)  # registered from its own cover page
    assert (doc.title, doc.act_number, doc.certified_date) == (
        "Inland Revenue (Amendment) Act, No. 2 of 2025",
        "2 of 2025",
        date(2025, 3, 20),
    )
    assert (env / "raw" / "IRA_AMEND_02_2025_E.pdf").is_file()
    assert NEW_ID in read_manifest(env / "raw")


@pytest.mark.parametrize(
    "content, reason",
    [
        (b"<html>Not found</html>", "did not return a PDF"),
        ("principal", "not an Inland Revenue (Amendment) Act"),  # the 2017 Act itself
    ],
)
def test_a_wrong_document_changes_nothing(env, content, reason):
    if content == "principal":
        content = (RAW / "IRA_24_2017_E.pdf").read_bytes()
    before = _snapshot(env)
    out = ingest(URL, lambda url: content, apply=lambda ids: pytest.fail("must not write"))
    assert out.status == "held" and reason in out.reason
    assert _snapshot(env) == before


def test_an_act_already_in_the_corpus_is_skipped(env):
    act10 = (RAW / "IRA_AMEND_10_2021_E.pdf").read_bytes()
    out = ingest("https://example.org/copy.pdf", lambda url: act10,
                 apply=lambda ids: pytest.fail("must not write"))  # fmt: skip
    assert out.status == "skipped" and "already in the corpus" in out.reason


def test_graph_down_means_everything_is_put_back(env):
    before = _snapshot(env)

    def down(ids):
        raise ConnectionError("Neo4j is not running")

    out = ingest(URL, lambda url: NEW_PDF, apply=down)
    assert out.status == "held" and "graph could not be updated" in out.reason
    assert _snapshot(env) == before  # registry, manifest and files exactly as before
    assert NEW_ID not in {d.document_id for d in load_registry().documents}


def test_cover_page_identifies_the_act():
    assert identify("INLAND REVENUE (AMENDMENT) ACT, No. 2 OF 2025", "2025-03-20") == (
        NEW_ID, "Inland Revenue (Amendment) Act, No. 2 of 2025", "2 of 2025",
        "IRA_AMEND_02_2025_E.pdf")  # fmt: skip
    with pytest.raises(Exception, match="not an Inland Revenue"):
        identify("VALUE ADDED TAX (AMENDMENT) ACT, No. 3 OF 2025", "2025-03-20")


def test_impossible_values_are_caught():
    from knowledge_graph.versions import Version

    v = Version("RULE-X-V1", "x", 1, 250.0, date(2025, 4, 1), None, "active", "D", "C", "t",
                value_kind="rate")  # fmt: skip
    assert check_values({"x": [v]}) == ["RULE-X-V1: rate 250.0%"]
