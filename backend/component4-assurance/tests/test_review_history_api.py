
import mongomock
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from human_review.mongo_repository import MongoReviewCaseRepository
from human_review.repository import ReviewCaseRepository


client = TestClient(api_main.app)

BASE = "/human-review/cases"


@pytest.fixture(autouse=True)
def mock_mongodb_repository(monkeypatch):
    """Use a fresh fake MongoDB collection for each test."""

    mongo_client = mongomock.MongoClient()
    collection = mongo_client["c4_assurance"]["review_cases"]

    repository = MongoReviewCaseRepository(collection)

    monkeypatch.setattr(
        api_main,
        "review_repository",
        repository,
    )

    return collection


def create_case():
    payload = {
        "request_id": "AUDIT-TEST-001",
        "reported_total": "180000",
        "line_items": [
            "50000",
            "70000",
            "30000",
        ],
        "claims": [
            {
                "claim_id": "CL-001",
                "source_id": "DOC-001",
                "excerpt": "example relief",
            }
        ],
        "sources": [
            {
                "source_id": "DOC-001",
                "text": "The example relief is available.",
            }
        ],
        "regeneration_attempts": 0,
    }

    response = client.post(BASE, json=payload)

    assert response.status_code == 201

    return response.json()["case_id"]


def test_case_creation_history():
    case_id = create_case()

    response = client.get(f"{BASE}/{case_id}/history")

    assert response.status_code == 200

    history = response.json()

    assert len(history) == 1
    assert history[0]["event_type"] == "CASE_CREATED"
    assert history[0]["case_id"] == case_id


def test_complete_review_history():
    case_id = create_case()

    start_response = client.post(f"{BASE}/{case_id}/start")
    assert start_response.status_code == 200

    complete_response = client.post(
        f"{BASE}/{case_id}/complete",
        json={
            "reviewer_id": "reviewer-001",
            "decision": "REQUEST_CORRECTION",
            "comment": "Incorrect reported total.",
        },
    )
    assert complete_response.status_code == 200

    response = client.get(f"{BASE}/{case_id}/history")

    assert response.status_code == 200

    history = response.json()

    assert [event["event_type"] for event in history] == [
        "CASE_CREATED",
        "REVIEW_STARTED",
        "REVIEW_COMPLETED",
    ]

    assert history[2]["actor_id"] == "reviewer-001"
    assert history[2]["decision"] == "REQUEST_CORRECTION"
    assert history[2]["occurred_at"] is not None


def test_unknown_case_history_returns_404():
    response = client.get(f"{BASE}/REV-UNKNOWN/history")

    assert response.status_code == 404


def test_memory_backend_returns_501(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "review_repository",
        ReviewCaseRepository(),
    )

    response = client.get(f"{BASE}/REV-001/history")

    assert response.status_code == 501


def test_history_available_from_new_repository_instance(
    mock_mongodb_repository,
    monkeypatch,
):
    case_id = create_case()

    client.post(f"{BASE}/{case_id}/start")

    # Simulate a new repository instance using the same database.
    new_repository = MongoReviewCaseRepository(
        mock_mongodb_repository
    )

    monkeypatch.setattr(
        api_main,
        "review_repository",
        new_repository,
    )

    response = client.get(f"{BASE}/{case_id}/history")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_legacy_case_without_history(
    mock_mongodb_repository,
):
    case_id = create_case()

    # Simulate a case created before audit history existed.
    mock_mongodb_repository.update_one(
        {"_id": case_id},
        {"$unset": {"events": ""}},
    )

    response = client.get(f"{BASE}/{case_id}/history")

    assert response.status_code == 200
    assert response.json() == []
