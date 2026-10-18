
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

ENDPOINT = "/assurance/citation-check"

SOURCES = [
    {
        "source_id": "TAX-DOC-001",
        "text": (
            "The example deduction is available "
            "subject to the stated conditions."
        ),
    }
]


def make_request(
    source_id="TAX-DOC-001",
    excerpt="available subject to the stated conditions",
    regeneration_attempts=0,
):
    return {
        "claim": {
            "claim_id": "CLAIM-001",
            "source_id": source_id,
            "excerpt": excerpt,
        },
        "sources": SOURCES,
        "regeneration_attempts": regeneration_attempts,
    }


def test_valid_citation_passes():
    response = client.post(ENDPOINT, json=make_request())

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "PASS"


def test_missing_citation_triggers_regeneration():
    response = client.post(
        ENDPOINT,
        json=make_request(source_id=None, excerpt=None),
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "REGENERATE"


def test_unknown_source_triggers_human_review():
    response = client.post(
        ENDPOINT,
        json=make_request(source_id="UNKNOWN-DOC"),
    )

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["action"] == "HUMAN_REVIEW"
    assert data["findings"][0]["error_code"] == "UNKNOWN_SOURCE"


def test_missing_excerpt_triggers_regeneration():
    response = client.post(
        ENDPOINT,
        json=make_request(excerpt=None),
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "REGENERATE"


def test_unmatched_excerpt_triggers_human_review():
    response = client.post(
        ENDPOINT,
        json=make_request(excerpt="This passage does not exist."),
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "HUMAN_REVIEW"


def test_retry_limit_escalates():
    response = client.post(
        ENDPOINT,
        json=make_request(
            source_id=None,
            excerpt=None,
            regeneration_attempts=2,
        ),
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "HUMAN_REVIEW"


def test_blank_claim_id_rejected():
    payload = make_request()
    payload["claim"]["claim_id"] = "   "

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_duplicate_source_ids_rejected():
    payload = make_request()
    payload["sources"] = [SOURCES[0], SOURCES[0]]

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_negative_retry_count_rejected():
    payload = make_request()
    payload["regeneration_attempts"] = -1

    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 422
