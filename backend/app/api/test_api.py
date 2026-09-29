import pytest
from fastapi.testclient import TestClient

from backend.app.database.database import (
    SessionLocal,
)
from backend.app.main import app
from backend.app.models.analysis import Analysis


@pytest.fixture(autouse=True)
def clean_database():
    """
    Keep API tests isolated.

    The analysis endpoints now persist successful
    analyses, so each test must start and finish
    with a clean database.
    """

    db = SessionLocal()

    try:
        db.query(Analysis).delete()
        db.commit()
    finally:
        db.close()

    yield

    db = SessionLocal()

    try:
        db.query(Analysis).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "ThreatGuard API"
    assert data["status"] == "online"


def test_health(client):
    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_info(client):
    response = client.get(
        "/api/v1/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert "url" in data["models"]
    assert "sms" in data["models"]


def test_url_analysis(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={
            "url": (
                "https://secure-account-verification."
                "example.com/login"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input_type"] == "url"
    assert data["prediction"] == "phishing"
    assert data["risk_score"] >= 50
    assert data["risk_band"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }


def test_sms_analysis(client):
    response = client.post(
        "/api/v1/analyze/sms",
        json={
            "text": (
                "URGENT! Your bank account will be "
                "blocked today. Verify KYC immediately."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input_type"] == "sms"
    assert data["prediction"] == "scam"
    assert data["risk_score"] >= 50


def test_unified_url_analysis(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "input_type": "url",
            "value": "https://github.com",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input_type"] == "url"


def test_unified_sms_analysis(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "input_type": "sms",
            "value": (
                "Hey bro, are you coming to class tomorrow?"
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input_type"] == "sms"


def test_empty_url_rejected(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={
            "url": "",
        },
    )

    assert response.status_code == 422


def test_empty_sms_rejected(client):
    response = client.post(
        "/api/v1/analyze/sms",
        json={
            "text": "",
        },
    )

    assert response.status_code == 422


def test_history_empty(client):
    response = client.get(
        "/api/v1/history"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 0
    assert data["items"] == []


def test_history_after_analysis(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={
            "url": "https://github.com",
        },
    )

    assert response.status_code == 200

    history = client.get(
        "/api/v1/history"
    )

    assert history.status_code == 200

    data = history.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1

    item = data["items"][0]

    assert item["input_type"] == "url"
    assert item["input_value"] == (
        "https://github.com"
    )
    assert "prediction" in item
    assert "risk_score" in item
    assert "risk_band" in item
    assert "confidence" in item
    assert "model_name" in item
    assert "model_version" in item
    assert "reasons" in item
    assert "created_at" in item


def test_history_limit(client):
    messages = [
        "Hey bro, are you coming to class tomorrow?",
        "Hello, how are you?",
        "Good morning, see you at class.",
    ]

    for message in messages:
        response = client.post(
            "/api/v1/analyze/sms",
            json={
                "text": message,
            },
        )

        assert response.status_code == 200

    history = client.get(
        "/api/v1/history?limit=2"
    )

    assert history.status_code == 200

    data = history.json()

    assert data["total"] == 3
    assert len(data["items"]) == 2


def test_history_limit_validation(client):
    response = client.get(
        "/api/v1/history?limit=0"
    )

    assert response.status_code == 422

    response = client.get(
        "/api/v1/history?limit=101"
    )

    assert response.status_code == 422
