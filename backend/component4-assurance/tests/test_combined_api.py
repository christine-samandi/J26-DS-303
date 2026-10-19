
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)
ENDPOINT = "/assurance/combined-check"


def valid_payload():
    return {
        "reported_total": "150000",
        "line_items": [
            "50000",
            "70000",
            "30000",
        ],
        "claims": [
            {
                "claim_id": "CL-001",
                "source_id": "DOC-001",
                "excerpt": "relief may apply",
            },
            {
                "claim_id": "CL-002",
                "source_id": "DOC-002",
                "excerpt": "uses the applicable rate",
            },
        ],
        "sources": [
            {
                "source_id": "DOC-001",
                "text": "Example relief may apply subject to conditions.",
            },
            {
                "source_id": "DOC-002",
                "text": "Example calculation uses the applicable rate.",
            },
        ],
        "regeneration_attempts": 0,
    }


def test_combined_valid_request_passes():
    response = client.post(ENDPOINT, json=valid_payload())

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["action"] == "PASS"
    assert data["decision"]["failed_checks"] == 0
    assert len(data["findings"]) == 3


def test_missing_citation_triggers_regeneration():
    payload = valid_payload()
    payload["claims"][1]["source_id"] = None
    payload["claims"][1]["excerpt"] = None

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "REGENERATE"


def test_unknown_source_triggers_human_review():
    payload = valid_payload()
    payload["claims"][1]["source_id"] = "UNKNOWN-DOC"

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "HUMAN_REVIEW"


def test_wrong_total_triggers_human_review():
    payload = valid_payload()
    payload["reported_total"] = "180000"

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["action"] == "HUMAN_REVIEW"
    assert data["findings"][0]["passed"] is False


def test_multiple_missing_citations():
    payload = valid_payload()

    for claim in payload["claims"]:
        claim["source_id"] = None
        claim["excerpt"] = None

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "REGENERATE"
    assert response.json()["decision"]["failed_checks"] == 2


def test_empty_claims_rejected():
    payload = valid_payload()
    payload["claims"] = []

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_invalid_money_format_rejected():
    payload = valid_payload()
    payload["reported_total"] = 150000

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_duplicate_claim_ids_rejected():
    payload = valid_payload()
    payload["claims"][1]["claim_id"] = "CL-001"

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_duplicate_source_ids_rejected():
    payload = valid_payload()
    payload["sources"][1]["source_id"] = "DOC-001"

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422
