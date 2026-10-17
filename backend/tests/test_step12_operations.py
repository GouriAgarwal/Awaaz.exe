"""Unit and integration tests for Step 12: Production Deployment, Observability & Operational Readiness."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.errors import ErrorCode, OperationalException, error_tracker
from app.observability.metrics import metrics_registry
from app.observability.service import observability_service


@pytest.fixture
def client():
    return TestClient(app)


def test_backward_compatible_health(client):
    """Test that existing /api/health endpoint remains functional."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data


def test_liveness_probe_fast_non_blocking(client):
    """Test that /api/health/live responds quickly and accurately."""
    response = client.get("/api/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LIVE"
    assert "timestamp" in data


def test_readiness_probe_components(client):
    """Test that /api/health/ready evaluates all required subsystem checks."""
    response = client.get("/api/health/ready")
    assert response.status_code in [200, 503]
    data = response.json()
    assert data["status"] in ["READY", "DEGRADED", "NOT_READY"]
    assert "components" in data
    assert "strict_integrity_mode" in data

    # Verify key components are probed
    expected_components = [
        "aasist_detector",
        "ecapa_speaker_verifier",
        "streaming_pipeline",
    ]
    for comp in expected_components:
        assert comp in data["components"]
        assert "healthy" in data["components"][comp]


def test_detailed_operational_status(client):
    """Test that /api/health/status returns detailed multi-system status."""
    response = client.get("/api/health/status")
    assert response.status_code == 200
    data = response.json()
    assert "application_status" in data
    assert "environment" in data
    assert "uptime_seconds" in data
    assert "active_detector" in data
    assert "speaker_model_available" in data
    assert "streaming_available" in data
    assert "active_sessions" in data
    assert "max_active_sessions" in data
    assert "model_integrity_status" in data
    assert "evaluation_status_summary" in data


def test_observability_metrics_endpoint(client):
    """Test that /api/observability/metrics returns comprehensive telemetry."""
    response = client.get("/api/observability/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "http" in data
    assert "aasist" in data
    assert "ecapa" in data
    assert "fusion" in data
    assert "streaming" in data
    assert data["http"]["total_requests"] >= 0


def test_observability_models_endpoint(client):
    """Test that /api/observability/models returns lifecycle tracking for both models."""
    response = client.get("/api/observability/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2

    model_ids = [m["model_id"] for m in data]
    assert "VoiceShield-AASIST-Pretrained-v1" in model_ids
    assert "SpeechBrain-ECAPA-TDNN-v1" in model_ids

    for m in data:
        assert "expected_sha256" in m
        assert "actual_sha256" in m
        assert "scientific_status" in m


def test_observability_streaming_endpoint(client):
    """Test that /api/observability/streaming returns capacity and safeguard metrics."""
    response = client.get("/api/observability/streaming")
    assert response.status_code == 200
    data = response.json()
    assert "active_sessions" in data
    assert "max_active_sessions" in data
    assert "utilization_percentage" in data
    assert "buffer_memory_cap_sec" in data
    assert data["buffer_memory_cap_sec"] == 15.0


def test_observability_errors_endpoint(client):
    """Test that /api/observability/errors tracks and returns taxonomy distributions."""
    # Seed a known error
    error_tracker.record_error(
        error_code=ErrorCode.AUDIO_INVALID,
        message="Test invalid audio error",
        request_id="test-req-123",
    )

    response = client.get("/api/observability/errors?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total_errors_recorded" in data
    assert "error_counts_by_code" in data
    assert data["total_errors_recorded"] >= 1
    assert data["error_counts_by_code"].get(ErrorCode.AUDIO_INVALID.value, 0) >= 1


def test_request_id_tracing_header(client):
    """Test that incoming requests receive X-Request-ID propagation."""
    # Test client without custom ID
    res1 = client.get("/api/health/live")
    assert "x-request-id" in res1.headers
    gen_id = res1.headers["x-request-id"]
    assert len(gen_id) > 0

    # Test client with custom ID
    custom_id = "custom-trace-uuid-456"
    res2 = client.get("/api/health/live", headers={"x-request-id": custom_id})
    assert res2.headers.get("x-request-id") == custom_id

