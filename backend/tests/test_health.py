"""Tests for GET /api/health endpoint."""

from fastapi.testclient import TestClient


def test_health_check_returns_200(client: TestClient) -> None:
    """GET /api/health should return status 200 with expected structure."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "VoiceShield API"
    assert "version" in data
    assert data["version"] == "0.1.0"
