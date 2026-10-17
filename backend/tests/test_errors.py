"""Tests for error handling and invalid API routes."""

from fastapi.testclient import TestClient


def test_invalid_api_route_returns_404_json(client: TestClient) -> None:
    """Requesting an unknown route should return structured 404 JSON."""
    response = client.get("/api/non_existent_route")
    assert response.status_code == 404

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 404
    assert "Not Found" in data["error"]["message"]
    assert data["error"]["path"] == "/api/non_existent_route"
