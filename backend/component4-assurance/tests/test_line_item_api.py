
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

ENDPOINT = "/assurance/line-item-check"


def test_matching_line_items_return_pass():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "150000.00",
            "line_items": ["50000.00", "70000.00", "30000.00"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["decision"]["action"] == "PASS"
    assert data["findings"][0]["passed"] is True
    assert data["findings"][0]["expected_value"] == "150000.00"


def test_mismatching_line_items_require_review():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "180000.00",
            "line_items": ["50000.00", "70000.00", "30000.00"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["decision"]["action"] == "HUMAN_REVIEW"
    assert data["decision"]["failed_checks"] == 1
    assert data["findings"][0]["error_code"] == (
        "LINE_ITEM_SUM_MISMATCH"
    )


def test_empty_line_items_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "0",
            "line_items": [],
        },
    )

    assert response.status_code == 422


def test_invalid_line_item_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "100",
            "line_items": ["NaN"],
        },
    )

    assert response.status_code == 422


def test_numeric_line_item_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "100",
            "line_items": [100],
        },
    )

    assert response.status_code == 422


def test_missing_reported_total_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "line_items": ["100"],
        },
    )

    assert response.status_code == 422


def test_negative_retry_count_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "100",
            "line_items": ["100"],
            "regeneration_attempts": -1,
        },
    )

    assert response.status_code == 422


def test_unexpected_field_rejected():
    response = client.post(
        ENDPOINT,
        json={
            "reported_total": "100",
            "line_items": ["100"],
            "unrecognized_field": "unexpected",
        },
    )

    assert response.status_code == 422
