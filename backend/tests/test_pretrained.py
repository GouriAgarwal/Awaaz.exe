"""Step 6 Pretrained AASIST Detection Test Suite.

Validates:
1. Checkpoint file integrity (existence, size, SHA-256).
2. Strict PyTorch architecture loading and parameter count (297,866).
3. Forward pass execution and tensor output shapes.
4. Windowing and inference across short (<3s), standard (3s), and long (>6s) audio.
5. Score semantics, finiteness, and absence of NaNs/Infs.
6. Provenance metadata reporting PRETRAINED_NOT_YET_VALIDATED.
7. Registry integration and separation of pretrained vs untrained models.
8. Robust error handling for corrupt or missing checkpoints.
9. API integration for POST /api/detection/analyze and GET /api/detection/provenance.
"""

import io
import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from app.detection.pretrained.aasist_architecture import AASISTModel, get_default_aasist_config
from app.detection.pretrained.adapter import AASISTPretrainedAdapter
from app.detection.pretrained.loader import (
    AASIST_EXPECTED_PARAMS,
    AASIST_EXPECTED_SHA256,
    PretrainedModelLoader,
)
from app.detection.pretrained.provenance import get_pretrained_aasist_provenance
from app.detection.provenance_schema import PretrainedStatus, CalibrationStatus
from app.detection.registry import detector_registry
from app.detection.schemas import ClassificationLabel, ScoreType
from app.main import app
from app.audio.schemas import AudioData

client = TestClient(app)


def _generate_sine_audio(duration: float = 3.0, sr: int = 16000, freq: float = 440.0) -> AudioData:
    """Generate a clean synthetic sine wave AudioData."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * freq * t)
    return AudioData(
        samples=samples,
        sample_rate=sr,
        duration_seconds=duration,
    )


def _generate_audio_bytes(duration: float = 3.0, sr: int = 16000) -> bytes:
    """Generate in-memory WAV bytes for API tests."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


class TestPretrainedCheckpointIntegrity:
    """Tests for physical checkpoint file and cryptographic integrity."""

    def test_checkpoint_file_exists_and_nonzero(self):
        checkpoint_path = PretrainedModelLoader.get_checkpoint_path()
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
        file_size = os.path.getsize(checkpoint_path)
        assert file_size > 1_000_000, f"Checkpoint size ({file_size} bytes) is suspiciously small"
        assert file_size == 1_281_532, f"Expected 1,281,532 bytes, got {file_size}"

    def test_checkpoint_sha256_integrity(self):
        is_valid, actual_sha, size = PretrainedModelLoader.verify_checkpoint_integrity()
        assert is_valid is True
        assert actual_sha == AASIST_EXPECTED_SHA256
        assert size == 1_281_532


class TestPretrainedArchitectureAndLoading:
    """Tests for AASIST PyTorch architecture and strict state_dict loading."""

    def test_model_instantiation(self):
        config = get_default_aasist_config()
        model = AASISTModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == AASIST_EXPECTED_PARAMS

    def test_strict_loader_loading(self):
        model = PretrainedModelLoader.load_aasist_model(device="cpu", download_if_missing=False)
        assert isinstance(model, AASISTModel)
        assert not model.training  # Must be in eval() mode
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == AASIST_EXPECTED_PARAMS

    def test_forward_pass_tensor_shapes(self):
        model = PretrainedModelLoader.load_aasist_model(device="cpu")
        batch_size = 2
        seq_len = 64600
        x = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        with torch.no_grad():
            last_hidden, logits = model(x)

        assert last_hidden.shape == (batch_size, 160)
        assert logits.shape == (batch_size, 2)
        assert torch.isfinite(logits).all()


class TestPretrainedAdapterInference:
    """Tests for AASISTPretrainedAdapter behavior and score calculations."""

    @pytest.fixture
    def adapter(self):
        return AASISTPretrainedAdapter(device="cpu")

    def test_adapter_metadata(self, adapter):
        meta = adapter.metadata()
        assert meta.model_name == "VoiceShield-AASIST-Pretrained-v1"
        assert meta.model_type == "pretrained_graph_attention"
        assert "AASIST.pth" in (meta.checkpoint_or_source or "")
        assert meta.is_fallback is False

    def test_adapter_short_audio_tiling(self, adapter):
        # 1.0 second audio (< 3.0s window, < 64600 samples)
        audio = _generate_sine_audio(duration=1.0)
        result = adapter.predict(audio)

        assert result.detector_metadata.model_name == "VoiceShield-AASIST-Pretrained-v1"
        assert result.score_type == ScoreType.UNCALIBRATED_MODEL_SCORE
        assert 0.0 <= result.score <= 1.0
        assert result.classification in [
            ClassificationLabel.NATURAL,
            ClassificationLabel.SYNTHETIC,
            ClassificationLabel.UNCERTAIN,
        ]
        assert len(result.window_scores) == 1
        assert result.window_scores[0].start_sec == 0.0
        assert result.window_scores[0].end_sec == 1.0
        assert np.isfinite(result.score)

    def test_adapter_exact_window_audio(self, adapter):
        # Exactly 3.0 seconds
        audio = _generate_sine_audio(duration=3.0)
        result = adapter.predict(audio)

        assert result.detector_metadata.model_name == "VoiceShield-AASIST-Pretrained-v1"
        assert len(result.window_scores) == 1
        assert 0.0 <= result.score <= 1.0
        assert np.isfinite(result.score)

    def test_adapter_multi_window_audio(self, adapter):
        # 6.0 seconds: windows at [0..3], [1.5..4.5], [3..6] -> 3 windows
        audio = _generate_sine_audio(duration=6.0)
        result = adapter.predict(audio)

        assert len(result.window_scores) == 3
        assert result.window_scores[0].start_sec == 0.0
        assert result.window_scores[0].end_sec == 3.0
        assert result.window_scores[1].start_sec == 1.5
        assert result.window_scores[1].end_sec == 4.5
        assert result.window_scores[2].start_sec == 3.0
        assert result.window_scores[2].end_sec == 6.0
        assert 0.0 <= result.score <= 1.0

    def test_adapter_very_short_audio_zero_length(self, adapter):
        # Extremely short audio (0.1s)
        audio = _generate_sine_audio(duration=0.1)
        result = adapter.predict(audio)
        assert len(result.window_scores) == 1
        assert 0.0 <= result.score <= 1.0


class TestPretrainedProvenanceAndRegistry:
    """Tests for provenance audit records and detector registry."""

    def test_pretrained_provenance_attributes(self):
        record = get_pretrained_aasist_provenance()
        assert record.detector_name == "VoiceShield-AASIST-Pretrained-v1"
        assert record.pretrained is True
        assert record.validated is False
        assert record.pretrained_status == PretrainedStatus.PRETRAINED_NOT_YET_VALIDATED
        assert record.calibration_status == CalibrationStatus.NOT_CALIBRATED
        assert record.training_dataset == "ASVspoof 2019 Logical Access (LA) Training Benchmark"
        assert "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0" in record.checkpoint_identifier

    def test_registry_contains_pretrained_and_untrained(self):
        detectors = detector_registry.list_detectors()
        names = [d.model_name for d in detectors]

        assert "VoiceShield-AASIST-Pretrained-v1" in names
        assert "VoiceShield-AASIST-v1" in names
        assert "VoiceShield-SpecCNN-v1" in names

        pretrained_det = detector_registry.get_detector("aasist_pretrained")
        assert "AASIST.pth" in pretrained_det.metadata().checkpoint_or_source

        untrained_det = detector_registry.get_detector("aasist_untrained")
        assert untrained_det.metadata().model_name == "VoiceShield-AASIST-v1"

    def test_registry_active_detector_is_pretrained(self):
        active_det = detector_registry.get_detector()
        assert "AASIST.pth" in active_det.metadata().checkpoint_or_source
        assert active_det.metadata().model_name == "VoiceShield-AASIST-Pretrained-v1"


class TestCorruptedCheckpointHandling:
    """Tests that corrupted weights fail verification and are rejected."""

    def test_corrupted_file_sha_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
            tf.write(b"CORRUPTED FAKE WEIGHTS DATA 1234567890")
            temp_path = tf.name

        try:
            is_valid, sha, size = PretrainedModelLoader.verify_checkpoint_integrity(temp_path)
            assert is_valid is False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestPretrainedApiIntegration:
    """Tests for HTTP API endpoints with the pretrained model."""

    def test_analyze_endpoint_with_pretrained(self):
        audio_bytes = _generate_audio_bytes(duration=3.0)
        files = {"file": ("test_3s.wav", audio_bytes, "audio/wav")}
        response = client.post("/api/detection/analyze?model_name=aasist_pretrained", files=files)

        assert response.status_code == 200
        res = response.json()
        assert res["success"] is True
        data = res["data"]
        assert data["detector_metadata"]["model_name"] == "VoiceShield-AASIST-Pretrained-v1"
        assert data["score_type"] == "uncalibrated_model_score"
        assert 0.0 <= data["score"] <= 1.0
        assert data["classification"] in ["NATURAL", "SYNTHETIC", "UNCERTAIN"]
        assert len(data["window_scores"]) >= 1

    def test_provenance_endpoint_reports_pretrained(self):
        response = client.get("/api/detection/provenance")
        assert response.status_code == 200
        data = response.json()

        assert "aasist_pretrained" in data
        pretrained_record = data["aasist_pretrained"]
        assert pretrained_record["pretrained"] is True
        assert pretrained_record["validated"] is False
        assert pretrained_record["pretrained_status"] == "PRETRAINED_NOT_YET_VALIDATED"
        assert pretrained_record["calibration_status"] == "NOT_CALIBRATED"

    def test_detectors_list_endpoint_includes_pretrained(self):
        response = client.get("/api/detection/detectors")
        assert response.status_code == 200
        data = response.json()
        model_names = [d["model_name"] for d in data]
        assert "VoiceShield-AASIST-Pretrained-v1" in model_names
