
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from human_review.repository import ReviewCaseRepository


client = TestClient(api_main.app)

BASE = "/human-review/cases"


@pytest.fixture(autouse=True)
def fresh_repository():
    api_main.review_repository = ReviewCaseRepository()


def review_payload(reported_total="180000"):
    return {
        "request_id": "req-001",
        "reported_total": reported_total,
        "line_items": ["50000", "70000", "30000"],
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


def create_case():
    response = client.post(BASE, json=review_payload())
    assert response.status_code == 201
    return response.json()["case_id"]


def test_create_pending_case():
    response = client.post(BASE, json=review_payload())

    assert response.status_code == 201
    assert response.json()["status"] == "PENDING"
    assert response.json()["feedback"] is None


def test_get_case():
    case_id = create_case()

    response = client.get(f"{BASE}/{case_id}")

    assert response.status_code == 200
    assert response.json()["case_id"] == case_id


def test_list_pending_cases():
    create_case()

    response = client.get(BASE, params={"status": "PENDING"})

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_start_review():
    case_id = create_case()

    response = client.post(f"{BASE}/{case_id}/start")

    assert response.status_code == 200
    assert response.json()["status"] == "IN_REVIEW"


@pytest.mark.parametrize(
    "decision",
    ["APPROVE", "REJECT", "REQUEST_CORRECTION"],
)
def test_complete_review(decision):
    case_id = create_case()

    client.post(f"{BASE}/{case_id}/start")

    response = client.post(
        f"{BASE}/{case_id}/complete",
        json={
            "reviewer_id": "reviewer-001",
            "decision": decision,
            "comment": "Reviewed the calculation.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["feedback"]["decision"] == decision


def test_pass_result_cannot_create_case():
    response = client.post(
        BASE,
        json=review_payload(reported_total="150000"),
    )

    assert response.status_code == 409


def test_unknown_case_returns_404():
    response = client.get(f"{BASE}/REV-UNKNOWN")

    assert response.status_code == 404


def test_pending_case_cannot_be_completed():
    case_id = create_case()

    response = client.post(
        f"{BASE}/{case_id}/complete",
        json={
            "reviewer_id": "reviewer-001",
            "decision": "REJECT",
            "comment": "Incorrect calculation.",
        },
    )

    assert response.status_code == 409
