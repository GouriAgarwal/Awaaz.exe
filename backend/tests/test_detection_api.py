"""Integration tests for detection API endpoints."""

import io
import soundfile as sf
import numpy as np
import pytest
from fastapi.testclient import TestClient


def create_test_wav_bytes(duration_sec: float = 3.0, freq: float = 440.0, sr: int = 16000) -> bytes:
    """Create in-memory WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf.read()


def test_detection_analyze_valid_wav(client: TestClient):
    """POST /api/detection/analyze processes audio and returns DetectionResponse."""
    wav_bytes = create_test_wav_bytes(duration_sec=3.5)
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}

    response = client.post("/api/detection/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data

    det_result = data["data"]
    assert "score" in det_result
    assert 0.0 <= det_result["score"] <= 1.0
    assert det_result["classification"] in ["NATURAL", "SYNTHETIC", "UNCERTAIN"]
    assert det_result["score_type"] in ["uncalibrated_model_score", "heuristic_fallback_score"]
    assert "detector_metadata" in det_result
    assert det_result["total_windows"] >= 1
    assert len(det_result["window_scores"]) == det_result["total_windows"]


def test_detection_analyze_explicit_fallback(client: TestClient):
    """POST /api/detection/analyze with detector=fallback uses fallback engine."""
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    files = {"file": ("test_sample.wav", wav_bytes, "audio/wav")}
    data_payload = {"detector": "fallback"}

    response = client.post("/api/detection/analyze", files=files, data=data_payload)

    assert response.status_code == 200
    res = response.json()["data"]
    assert res["detector_metadata"]["is_fallback"] is True
    assert res["score_type"] == "heuristic_fallback_score"
    assert len(res["warnings"]) > 0


def test_detection_analyze_empty_file(client: TestClient):
    """POST /api/detection/analyze rejects 0-byte file with 400 Bad Request."""
    files = {"file": ("empty.wav", b"", "audio/wav")}

    response = client.post("/api/detection/analyze", files=files)

    assert response.status_code == 400


def test_detection_analyze_corrupt_file(client: TestClient):
    """POST /api/detection/analyze rejects non-audio garbage bytes."""
    corrupt_bytes = b"This is random garbage not an audio file" * 50
    files = {"file": ("corrupt.wav", corrupt_bytes, "audio/wav")}

    response = client.post("/api/detection/analyze", files=files)

    assert response.status_code in [400, 415, 422]


def test_list_models_endpoint(client: TestClient):
    """GET /api/detection/models returns registered detector models."""
    response = client.get("/api/detection/models")

    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 2
    assert any(m["model_name"] == "VoiceShield-AASIST-v1" for m in models)


def test_active_model_endpoint(client: TestClient):
    """GET /api/detection/active returns active detector metadata."""
    response = client.get("/api/detection/active")

    assert response.status_code == 200
    meta = response.json()
    assert "model_name" in meta
    assert "scientific_disclaimer" in meta


def test_system_status_includes_detector_status(client: TestClient):
    """GET /api/system/status contains detector_available."""
    response = client.get("/api/system/status")

    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "online"
    assert data.get("detector_available") is True
