"""Integration tests for POST /api/features/extract endpoint."""

import io
import soundfile as sf
import numpy as np
from fastapi.testclient import TestClient


def _make_wav_harmonic_bytes(duration: float = 1.0, f0: float = 200.0) -> bytes:
    """Generate in-memory WAV harmonic audio bytes."""
    t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
    signal = (0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, 16000, format="WAV")
    return buf.getvalue()


def test_extract_features_endpoint_success(client: TestClient) -> None:
    """Test 16 & 17: POST /api/features/extract returns structured feature payload and latency."""
    wav_bytes = _make_wav_harmonic_bytes(duration=1.5, f0=220.0)

    response = client.post(
        "/api/features/extract",
        files={"file": ("speech.wav", wav_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # Schema Validation
    assert data["success"] is True
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] > 0
    assert "features" in data
    assert "explainability" in data

    features = data["features"]
    assert "time_domain" in features
    assert "spectral" in features
    assert "mfcc" in features
    assert "pitch" in features
    assert "voice_quality" in features
    assert "prosody" in features

    # Verify specific feature contents
    assert len(features["mfcc"]["means"]) == 20
    assert features["pitch"]["f0_mean_hz"] is not None
    assert 200.0 <= features["pitch"]["f0_mean_hz"] <= 240.0

    # Ensure explainability metadata contains descriptions
    assert "spectral_centroid" in data["explainability"]
    assert "jitter" in data["explainability"]

    # Test Security: No raw audio data in response
    assert "samples" not in features
    assert "pcm" not in features
    assert "samples" not in data


def test_extract_features_invalid_audio_returns_400(client: TestClient) -> None:
    """Test 18: Uploading empty or corrupted audio returns HTTP 4xx error."""
    response = client.post(
        "/api/features/extract",
        files={"file": ("bad.wav", b"NOT_VALID_AUDIO_DATA", "audio/wav")},
    )
    assert response.status_code in [400, 415]


def test_extract_features_empty_file_returns_400(client: TestClient) -> None:
    """Uploading empty file returns HTTP 400."""
    response = client.post(
        "/api/features/extract",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
