
import pytest

from human_review.repository import ReviewCaseRepository
from human_review.storage import build_review_repository


def test_memory_repository_selected(monkeypatch):
    monkeypatch.setenv("C4_REVIEW_STORAGE", "memory")

    repository = build_review_repository()

    assert isinstance(repository, ReviewCaseRepository)


def test_invalid_storage_backend_rejected(monkeypatch):
    monkeypatch.setenv("C4_REVIEW_STORAGE", "invalid")

    with pytest.raises(ValueError, match="Unsupported"):
        build_review_repository()


def test_mongodb_requires_connection_uri(monkeypatch):
    monkeypatch.setenv("C4_REVIEW_STORAGE", "mongodb")
    monkeypatch.setenv("MONGODB_URI", "")

    with pytest.raises(RuntimeError, match="MONGODB_URI"):
        build_review_repository()
