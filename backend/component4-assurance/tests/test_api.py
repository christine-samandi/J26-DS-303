
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_matching_totals_return_pass():
    response = client.post(
        "/assurance/total-check",
        json={
            "reported_total": "150000.00",
            "calculated_total": "150000.00",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["action"] == "PASS"
    assert data["findings"][0]["passed"] is True


def test_mismatching_totals_require_review():
    response = client.post(
        "/assurance/total-check",
        json={
            "reported_total": "180000.00",
            "calculated_total": "150000.00",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["action"] == "HUMAN_REVIEW"
    assert data["findings"][0]["error_code"] == "TOTAL_MISMATCH"


def test_missing_total_is_rejected():
    response = client.post(
        "/assurance/total-check",
        json={"reported_total": "150000.00"},
    )

    assert response.status_code == 422


def test_invalid_decimal_is_rejected():
    response = client.post(
        "/assurance/total-check",
        json={
            "reported_total": "not-a-number",
            "calculated_total": "150000.00",
        },
    )

    assert response.status_code == 422


def test_negative_retry_count_is_rejected():
    response = client.post(
        "/assurance/total-check",
        json={
            "reported_total": "150000.00",
            "calculated_total": "150000.00",
            "regeneration_attempts": -1,
        },
    )

    assert response.status_code == 422


def test_numeric_json_total_is_rejected():
    response = client.post(
        "/assurance/total-check",
        json={
            "reported_total": 150000.00,
            "calculated_total": "150000.00",
        },
    )

    assert response.status_code == 422
