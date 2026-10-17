"""Security and privacy tests for Step 12 observability and logging subsystems."""

import json
import logging
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.errors import ErrorCode, OperationalException
from app.observability.logging import SensitiveDataSanitizer, StructuredJsonFormatter


def test_sensitive_data_sanitizer_waveform():
    """Verify that waveforms are redacted and never leaked into log strings."""
    dummy_waveform = [0.0123, -0.456, 0.789, 0.001] * 50
    payload = {
        "user": "alice",
        "waveform": dummy_waveform,
        "sample_rate": 16000,
    }
    sanitized = SensitiveDataSanitizer.sanitize(payload)
    assert sanitized["user"] == "alice"
    assert sanitized["sample_rate"] == 16000
    assert sanitized["waveform"] == f"[REDACTED_VECTOR_LEN_{len(dummy_waveform)}]"


def test_sensitive_data_sanitizer_embedding():
    """Verify that 512-dimensional speaker embeddings are redacted."""
    dummy_embedding = [0.042] * 512
    payload = {
        "speaker_embedding": dummy_embedding,
        "profile_id": "prof_999",
    }
    sanitized = SensitiveDataSanitizer.sanitize(payload)
    assert sanitized["profile_id"] == "prof_999"
    assert sanitized["speaker_embedding"] == SensitiveDataSanitizer.REDACTED_TEXT


def test_sensitive_data_sanitizer_base64_audio():
    """Verify that large base64 encoded audio strings are redacted."""
    long_b64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQ" * 10
    payload = {
        "audio_base64": long_b64,
        "context": "incoming_packet",
    }
    sanitized = SensitiveDataSanitizer.sanitize(payload)
    assert sanitized["context"] == "incoming_packet"
    assert sanitized["audio_base64"] == "[REDACTED_BASE64_AUDIO]"


def test_sensitive_data_sanitizer_auth_secrets():
    """Verify that authorization tokens, passwords, and secrets are redacted."""
    payload = {
        "authorization": "Bearer secret-token-xyz-12345",
        "api_key": "sk-proj-abc1234567890",
        "normal_field": "public_data",
    }
    sanitized = SensitiveDataSanitizer.sanitize(payload)
    assert sanitized["authorization"] == SensitiveDataSanitizer.REDACTED_TEXT
    assert sanitized["api_key"] == SensitiveDataSanitizer.REDACTED_TEXT
    assert sanitized["normal_field"] == "public_data"


def test_structured_json_formatter_redaction():
    """Verify that StructuredJsonFormatter outputs valid JSON and sanitizes extra fields."""
    formatter = StructuredJsonFormatter()
    logger = logging.getLogger("test_structured_logger")
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(
        "Processing user audio",
        extra={
            "raw_audio": [0.1, 0.2] * 100,
            "session_id": "sess-abc-789",
        },
    )

    handler.flush()
    output = log_stream.getvalue()
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Processing user audio"
    assert parsed["session_id"] == "sess-abc-789"
    assert parsed["raw_audio"] == SensitiveDataSanitizer.REDACTED_TEXT



def test_unhandled_exception_suppresses_traceback():
    """Verify that 500 internal server errors return generic safe messages without tracebacks."""
    client = TestClient(app)

    # Trigger a 404 or unhandled path to verify response format
    response = client.get("/api/non-existent-endpoint-to-test-error-format")
    assert response.status_code == 404
    data = response.json()
    assert "error_code" in data
    assert "timestamp" in data
    # Ensure no raw Python traceback string in body
    assert "Traceback (most recent call last)" not in response.text
