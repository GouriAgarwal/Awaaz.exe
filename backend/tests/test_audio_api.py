"""Integration tests for POST /api/audio/inspect endpoint."""

import io
import soundfile as sf
import numpy as np
from fastapi.testclient import TestClient


def _make_wav_payload(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory WAV bytes for API tests."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV")
    return buf.getvalue()


def test_inspect_audio_endpoint_success(client: TestClient) -> None:
    """Test O & P: POST /api/audio/inspect returns expected schema and no raw audio."""
    wav_bytes = _make_wav_payload(duration=1.5, sample_rate=16000)

    response = client.post(
        "/api/audio/inspect",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify Response Schema
    assert data["success"] is True
    assert "audio" in data
    assert "quality" in data

    audio_info = data["audio"]
    assert audio_info["sample_rate"] == 16000
    assert audio_info["channels"] == 1
    assert 1.4 <= audio_info["duration_seconds"] <= 1.6
    assert audio_info["original_format"] == "wav"

    quality_info = data["quality"]
    assert "rms_db" in quality_info
    assert "peak_db" in quality_info
    assert "clipping_ratio" in quality_info
    assert "silence_ratio" in quality_info
    assert quality_info["quality"] in ["good", "warning", "invalid"]

    # Test P: Verify raw audio samples are NOT leaked into response
    assert "samples" not in audio_info
    assert "samples" not in data
    assert "pcm" not in data


def test_inspect_audio_empty_file_returns_400(client: TestClient) -> None:
    """Uploading an empty file returns HTTP 400 error."""
    response = client.post(
        "/api/audio/inspect",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_inspect_audio_corrupted_file_returns_415(client: TestClient) -> None:
    """Uploading corrupted data returns HTTP 415/400 error."""
    response = client.post(
        "/api/audio/inspect",
        files={"file": ("corrupt.bin", b"INVALID_GARBAGE_BYTES_12345", "audio/wav")},
    )
    assert response.status_code in [400, 415]
