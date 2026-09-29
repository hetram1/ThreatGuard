from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "ThreatGuard API"
    assert data["status"] == "online"


def test_health():
    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_info():
    response = client.get(
        "/api/v1/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert "url" in data["models"]
    assert "sms" in data["models"]


def test_url_analysis():
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


def test_sms_analysis():
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


def test_unified_url_analysis():
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


def test_unified_sms_analysis():
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


def test_empty_url_rejected():
    response = client.post(
        "/api/v1/analyze/url",
        json={
            "url": "",
        },
    )

    assert response.status_code == 422


def test_empty_sms_rejected():
    response = client.post(
        "/api/v1/analyze/sms",
        json={
            "text": "",
        },
    )

    assert response.status_code == 422
