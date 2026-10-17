"""Tests for GET /api/system/status endpoint."""

from fastapi.testclient import TestClient


def test_system_status_returns_200(client: TestClient) -> None:
    """GET /api/system/status should return 200 with runtime info."""
    response = client.get("/api/system/status")
    assert response.status_code == 200

    data = response.json()
    assert data["backend"] == "online"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"
