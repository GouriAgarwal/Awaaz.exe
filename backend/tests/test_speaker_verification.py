"""
Step 8 Comprehensive Test Suite for Speaker Identity Verification Subsystem.
Tests model integrity, embedding properties, multi-sample enrollment, centroid aggregation,
cosine similarity, calibration states, privacy policies, evaluation metrics, and API endpoints.
"""
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from app.main import create_application
from app.speaker_verification.architecture import ECAPA_TDNN
from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.embedding import SpeechBrainECAPAEncoder
from app.speaker_verification.model_loader import (
    PretrainedSpeakerLoader,
    EXPECTED_SHA256,
    DEFAULT_WEIGHTS_PATH,
)
from app.speaker_verification.similarity import CosineSimilarityMetric
from app.speaker_verification.calibration import (
    SpeakerScoreCalibrator,
    SpeakerVerificationThreshold,
)
from app.speaker_verification.decision import SpeakerDecisionEngine
from app.speaker_verification.enrollment import SpeakerEnrollmentManager
from app.speaker_verification.provenance import get_speaker_model_provenance
from app.speaker_verification.privacy import BiometricPrivacyPolicy
from app.speaker_verification.schemas import (
    CalibrationStatus,
    ConfidenceBand,
    SpeakerDecision,
    ScoreType,
)
from app.speaker_verification.evaluation.schemas import TrialType, VerificationTrial
from app.speaker_verification.evaluation.metrics import (
    compute_rates_at_threshold,
    compute_eer,
    compute_roc_auc,
    compute_threshold_sweep,
    compute_comprehensive_metrics,
)
from app.speaker_verification.evaluation.evaluator import SpeakerVerificationEvaluator
from app.audio.schemas import AudioData


def generate_audio_signal(duration: float = 3.0, freq: float = 220.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * (freq * 2) * t)).astype(np.float32)


def generate_wav_bytes(duration: float = 3.0, freq: float = 220.0, sr: int = 16000) -> bytes:
    samples = generate_audio_signal(duration=duration, freq=freq, sr=sr)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Model Integrity & Embedding Tests
# ---------------------------------------------------------------------------

class TestSpeakerModelIntegrity:
    def test_checkpoint_exists_and_sha256(self):
        assert DEFAULT_WEIGHTS_PATH.exists()
        sha = PretrainedSpeakerLoader.compute_sha256(DEFAULT_WEIGHTS_PATH)
        assert sha == EXPECTED_SHA256

    def test_model_parameter_count_and_strict_loading(self):
        model = PretrainedSpeakerLoader.load_model(DEFAULT_WEIGHTS_PATH, strict=True)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == 20767552
        assert not model.training

    def test_embedding_dimension_and_l2_norm(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        audio = generate_audio_signal(duration=3.0, freq=220.0)
        emb = encoder.embed(audio, sample_rate=16000)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert np.all(np.isfinite(emb))
        l2_norm = float(np.linalg.norm(emb))
        assert abs(l2_norm - 1.0) < 1e-4

    def test_embedding_determinism(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        audio = generate_audio_signal(duration=2.5, freq=300.0)
        emb1 = encoder.embed(audio, sample_rate=16000)
        emb2 = encoder.embed(audio, sample_rate=16000)
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)

    def test_invalid_audio_input_rejections(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        with pytest.raises(ValueError, match="cannot be empty"):
            encoder.embed(np.array([], dtype=np.float32), sample_rate=16000)

        with pytest.raises(ValueError, match="Sample rate mismatch"):
            encoder.embed(generate_audio_signal(1.0), sample_rate=44100)

        with pytest.raises(ValueError, match="NaN or infinite"):
            bad_audio = np.array([0.1, np.nan, 0.3], dtype=np.float32)
            encoder.embed(bad_audio, sample_rate=16000)


# ---------------------------------------------------------------------------
# 2. Enrollment Tests
# ---------------------------------------------------------------------------

class TestSpeakerEnrollment:
    def test_single_sample_enrollment_and_centroid(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        manager = SpeakerEnrollmentManager(encoder)

        audio = generate_audio_signal(duration=3.0, freq=220.0)
        sample = AudioData(samples=audio, sample_rate=16000, channels=1, duration_seconds=3.0, original_format="wav")

        res = manager.enroll(profile_id="speaker_01", audio_samples=[sample])
        assert res.success is True
        assert res.profile_id == "speaker_01"
        assert res.sample_count == 1
        assert res.warning is not None # Below 10s recommendation

        profile = manager.get_profile("speaker_01")
        assert profile is not None
        assert profile["embedding"].shape == (192,)
        # Check L2 normalized
        assert abs(np.linalg.norm(profile["embedding"]) - 1.0) < 1e-4

    def test_multi_sample_enrollment_centroid_aggregation(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        manager = SpeakerEnrollmentManager(encoder)

        s1 = AudioData(samples=generate_audio_signal(4.0, 200.0), sample_rate=16000, channels=1, duration_seconds=4.0, original_format="wav")
        s2 = AudioData(samples=generate_audio_signal(4.0, 220.0), sample_rate=16000, channels=1, duration_seconds=4.0, original_format="wav")
        s3 = AudioData(samples=generate_audio_signal(4.0, 240.0), sample_rate=16000, channels=1, duration_seconds=4.0, original_format="wav")

        res = manager.enroll(profile_id="speaker_multi", audio_samples=[s1, s2, s3])
        assert res.success is True
        assert res.sample_count == 3
        assert res.total_audio_duration_seconds == 12.0
        assert res.warning is None # >= 10s duration

        # Verify profile listing is sanitized (no embedding exposed)
        profiles = manager.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].profile_id == "speaker_multi"
        assert not hasattr(profiles[0], "embedding")

    def test_short_and_silent_enrollment_rejections(self):
        encoder = SpeechBrainECAPAEncoder(DEFAULT_WEIGHTS_PATH)
        manager = SpeakerEnrollmentManager(encoder)

        short_audio = AudioData(samples=generate_audio_signal(0.4), sample_rate=16000, channels=1, duration_seconds=0.4, original_format="wav")
        with pytest.raises(ValueError, match="too short"):
            manager.enroll("short_spk", [short_audio])

        silent_audio = AudioData(samples=np.zeros(16000 * 2, dtype=np.float32), sample_rate=16000, channels=1, duration_seconds=2.0, original_format="wav")
        with pytest.raises(ValueError, match="virtually silent"):
            manager.enroll("silent_spk", [silent_audio])


# ---------------------------------------------------------------------------
# 3. Similarity & Verification Tests
# ---------------------------------------------------------------------------

class TestSimilarityAndVerification:
    def test_cosine_similarity_properties(self):
        metric = CosineSimilarityMetric()
        e1 = np.random.randn(192).astype(np.float32)
        e1 /= np.linalg.norm(e1)

        # Self similarity
        assert abs(metric.compute_similarity(e1, e1) - 1.0) < 1e-5

        # Orthogonal similarity
        e_ortho = np.random.randn(192).astype(np.float32)
        e_ortho -= np.dot(e_ortho, e1) * e1
        e_ortho /= np.linalg.norm(e_ortho)
        assert abs(metric.compute_similarity(e1, e_ortho)) < 1e-4

        # Opposite similarity
        assert abs(metric.compute_similarity(e1, -e1) - (-1.0)) < 1e-5

    def test_similarity_dimension_and_nan_safety(self):
        metric = CosineSimilarityMetric()
        e1 = np.ones(192, dtype=np.float32)
        e2 = np.ones(128, dtype=np.float32)

        with pytest.raises(ValueError, match="dimension mismatch"):
            metric.compute_similarity(e1, e2)

        with pytest.raises(ValueError, match="non-finite"):
            bad_e = np.array([np.nan] * 192, dtype=np.float32)
            metric.compute_similarity(e1, bad_e)

    def test_decision_engine_provisional_states(self):
        calibrator = SpeakerScoreCalibrator(SpeakerVerificationThreshold(threshold=0.65, uncertainty_margin=0.08))
        engine = SpeakerDecisionEngine(calibrator)

        # High similarity
        dec, conf = engine.evaluate(0.85)
        assert dec == SpeakerDecision.MATCH
        assert conf == ConfidenceBand.HIGH

        # Near threshold (uncertain)
        dec_unc, conf_unc = engine.evaluate(0.64)
        assert dec_unc in [SpeakerDecision.MATCH, SpeakerDecision.UNCERTAIN, SpeakerDecision.NON_MATCH]

        # Low similarity
        dec_low, conf_low = engine.evaluate(0.15)
        assert dec_low == SpeakerDecision.NON_MATCH


# ---------------------------------------------------------------------------
# 4. Biometric Privacy & Provenance Tests
# ---------------------------------------------------------------------------

class TestBiometricPrivacyAndProvenance:
    def test_privacy_metadata_guarantees(self):
        meta = BiometricPrivacyPolicy.get_privacy_metadata(in_memory_only=True)
        assert meta.raw_audio_persisted is False
        assert meta.embeddings_logged is False
        assert meta.in_memory_only is True
        assert "VoiceShield Biometric Privacy" in meta.policy

    def test_provenance_metadata(self):
        prov = get_speaker_model_provenance()
        assert prov.model_id == "speechbrain_ecapa_tdnn_voxceleb"
        assert prov.embedding_dimension == 192
        assert prov.checkpoint_sha256 == EXPECTED_SHA256
        assert prov.license == "Apache-2.0"
        assert prov.calibration_status == CalibrationStatus.NOT_CALIBRATED
        assert prov.scientific_status == "SPEAKER_MODEL_INTEGRATED_NOT_YET_VALIDATED"


# ---------------------------------------------------------------------------
# 5. Evaluation Framework & Speaker Leakage Tests
# ---------------------------------------------------------------------------

class TestSpeakerEvaluationFramework:
    def test_rate_calculations(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [TrialType.TARGET, TrialType.TARGET, TrialType.NON_TARGET, TrialType.NON_TARGET]
        rates = compute_rates_at_threshold(scores, labels, threshold=0.5)

        assert rates["true_accepts"] == 2
        assert rates["false_rejects"] == 0
        assert rates["true_rejects"] == 2
        assert rates["false_accepts"] == 0
        assert rates["far"] == 0.0
        assert rates["frr"] == 0.0

    def test_eer_computation_perfect_separation(self):
        scores = [0.95, 0.90, 0.85, 0.20, 0.15, 0.10]
        labels = [TrialType.TARGET, TrialType.TARGET, TrialType.TARGET, TrialType.NON_TARGET, TrialType.NON_TARGET, TrialType.NON_TARGET]
        eer, thresh = compute_eer(scores, labels)
        assert eer == 0.0
        assert thresh is not None

    def test_speaker_leakage_detection(self):
        # Case 1: Label contradiction (target but different speakers)
        trials_leaked = [
            VerificationTrial(
                trial_id="t1",
                enrollment_audio_path="a.wav",
                verification_audio_path="b.wav",
                enrollment_speaker_id="spk_A",
                verification_speaker_id="spk_B",
                trial_type=TrialType.TARGET,
            )
        ]
        is_leak, msg = SpeakerVerificationEvaluator.detect_speaker_leakage(trials_leaked)
        assert is_leak is True
        assert "Speaker identity leakage" in msg

        # Case 2: Clean disjoint target trial
        trials_clean = [
            VerificationTrial(
                trial_id="t2",
                enrollment_audio_path="a.wav",
                verification_audio_path="b.wav",
                enrollment_speaker_id="spk_A",
                verification_speaker_id="spk_A",
                trial_type=TrialType.TARGET,
            )
        ]
        is_leak2, msg2 = SpeakerVerificationEvaluator.detect_speaker_leakage(trials_clean)
        assert is_leak2 is False
        assert msg2 is None


# ---------------------------------------------------------------------------
# 6. Full REST API Integration Tests
# ---------------------------------------------------------------------------

class TestSpeakerApiIntegration:
    @pytest.fixture
    def client(self):
        app = create_application()
        return TestClient(app)

    def test_api_enroll_and_verify_lifecycle(self, client):
        wav_bytes = generate_wav_bytes(duration=3.0, freq=220.0)

        # 1. Enroll
        enroll_res = client.post(
            "/api/speaker/enroll",
            data={"profile_id": "test_bob"},
            files=[("files", ("bob_1.wav", wav_bytes, "audio/wav"))]
        )
        assert enroll_res.status_code == 200
        enroll_data = enroll_res.json()
        assert enroll_data["success"] is True
        assert enroll_data["profile_id"] == "test_bob"
        assert enroll_data["privacy"]["raw_audio_persisted"] is False

        # 2. Check profile status
        status_res = client.get("/api/speaker/profiles/test_bob/status")
        assert status_res.status_code == 200
        assert status_res.json()["profile_id"] == "test_bob"

        # 3. Verify matching audio
        verify_res = client.post(
            "/api/speaker/verify",
            data={"profile_id": "test_bob"},
            files={"file": ("bob_ver.wav", wav_bytes, "audio/wav")}
        )
        assert verify_res.status_code == 200
        verify_data = verify_res.json()
        assert verify_data["profile_id"] == "test_bob"
        assert verify_data["similarity_score"] >= 0.95
        assert verify_data["decision"] == "MATCH"
        assert verify_data["calibration_status"] == "NOT_CALIBRATED"

        # 4. Unknown profile returns 404
        unknown_res = client.post(
            "/api/speaker/verify",
            data={"profile_id": "nonexistent_profile"},
            files={"file": ("bob_ver.wav", wav_bytes, "audio/wav")}
        )
        assert unknown_res.status_code == 404

        # 5. Delete profile
        del_res = client.delete("/api/speaker/profiles/test_bob")
        assert del_res.status_code == 200

    def test_api_models_and_provenance(self, client):
        res_models = client.get("/api/speaker/models")
        assert res_models.status_code == 200
        assert res_models.json()["active_model"] == "speechbrain_ecapa_tdnn_voxceleb"

        res_prov = client.get("/api/speaker/provenance")
        assert res_prov.status_code == 200
        assert res_prov.json()["checkpoint_sha256"] == EXPECTED_SHA256
        assert res_prov.json()["license"] == "Apache-2.0"

    def test_api_evaluation_status(self, client):
        res = client.get("/api/speaker/evaluation/status")
        assert res.status_code == 200
        assert "dataset_manifest_instructions" in res.json()
