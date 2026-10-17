"""Tests for CORS middleware configuration."""

from fastapi.testclient import TestClient


def test_cors_preflight_request(client: TestClient) -> None:
    """Preflight OPTIONS request from allowed origin should receive CORS headers."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_get_request_header(client: TestClient) -> None:
    """Standard GET request with Origin header should include allow-origin header."""
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
