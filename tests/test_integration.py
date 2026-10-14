import pytest


@pytest.mark.parametrize(
    "value",
    [
        "not a url",
        "https://",
        "javascript:alert(1)",
        "ftp://example.com",
        "https://example.com:bad",
        "https://bad host.com",
        "https://[::1]evil.com",
        "https://example.com:",
        "https://a@b@example.com",
    ],
)
@pytest.mark.parametrize("unified", [False, True])
def test_reject_invalid_without_history(client, value, unified):
    route = "/api/v1/analyze" if unified else "/api/v1/analyze/url"
    payload = {"input_type": "url", "value": value} if unified else {"url": value}
    response = client.post(route, json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert response.json()["errors"]
    assert client.get("/api/v1/history").json()["total"] == 0


def test_api_matches_direct_and_persists(client):
    from backend.app.main import threat_service

    value = "https://docs.python.org/3/library/urllib.parse.html"
    direct = threat_service.analyze_url(value)
    response = client.post("/api/v1/analyze/url", json={"url": value})
    assert response.status_code == 200
    for key in ["prediction", "probability", "model_evidence", "model", "warnings"]:
        assert response.json()[key] == direct[key]
    history = client.get("/api/v1/history").json()
    assert history["total"] == 1
    assert history["items"][0]["model_version"] == "reliable-v1"
    assert history["items"][0]["input_value"] == value


def test_equivalent_routes(client):
    a = client.post("/api/v1/analyze/url", json={"url": "github.com/"}).json()
    b = client.post(
        "/api/v1/analyze",
        json={"input_type": "url", "value": "https://github.com:443/"},
    ).json()
    assert a["probability"] == b["probability"]
    assert a["normalized_input"] == b["normalized_input"] == "https://github.com"


def test_no_confidence_claim(client):
    data = client.post(
        "/api/v1/analyze/sms", json={"text": "hello, see you tomorrow"}
    ).json()
    assert "not probability" in data["score_description"]
    assert data["warnings"]
    assert data["model_evidence"]
