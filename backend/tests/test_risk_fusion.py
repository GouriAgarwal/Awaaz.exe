"""
Step 9 Comprehensive Test Suite for Controlled Fusion & Impersonation Risk Engine.
Tests the 4 primary fusion scenarios, intermediate rules, audio degradation, security boundaries,
4-quadrant benchmark evaluation, and REST API endpoints.
"""
import io
import math
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import create_application
from app.risk.schemas import (
    RiskLevel,
    RecommendedAction,
    EvidenceSeverity,
    RiskSimulationRequest,
)
from app.risk.normalization import SignalNormalizer, SyntheticBand, SpeakerSimilarityBand
from app.risk.rules import FusionRuleMatrix
from app.risk.confidence import EvidenceConfidenceEvaluator
from app.risk.actions import ActionPolicyMapper
from app.risk.fusion import ControlledImpersonationRiskEngine
from app.risk.service import ImpersonationRiskService
from app.risk.provenance import get_risk_provenance
from app.risk.evaluation.schemas import (
    SpeakerLabel,
    SyntheticLabel,
    AttackLabel,
    FusionTrial,
)
from app.risk.evaluation.metrics import compute_fusion_metrics
from app.risk.evaluation.evaluator import FusionEvaluator


def generate_wav_bytes(duration: float = 3.0, freq: float = 220.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    samples = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Four Primary Fusion Scenarios
# ---------------------------------------------------------------------------

class TestFourPrimaryScenarios:
    def test_scenario_a_authorized_natural(self):
        # Scenario A: High Speaker Similarity + Low Synthetic Score
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.12,
            synth_classification="NATURAL",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.88,
            spk_decision="MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
            audio_quality_status="good",
        )
        assert result.risk_level == RiskLevel.LOW
        assert result.risk_score < 25
        assert result.decision == "LIKELY_LEGITIMATE_SPEAKER"
        assert result.recommended_action == RecommendedAction.ALLOW
        assert any(e.code == "AUTHORIZED_NATURAL_VOICE" for e in result.evidence)

    def test_scenario_b_potential_voice_clone(self):
        # Scenario B: High Speaker Similarity + High Synthetic Score (Targeted Clone)
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.92,
            synth_classification="SYNTHETIC",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.85,
            spk_decision="MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
            audio_quality_status="good",
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.risk_score >= 75
        assert result.decision == "POTENTIAL_VOICE_CLONE"
        assert result.recommended_action == RecommendedAction.BLOCK_OR_ESCALATE
        assert any(e.code == "POTENTIAL_VOICE_CLONE" and e.severity == EvidenceSeverity.CRITICAL for e in result.evidence)

    def test_scenario_c_unknown_natural(self):
        # Scenario C: Low Speaker Similarity + Low Synthetic Score
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.15,
            synth_classification="NATURAL",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.18,
            spk_decision="NON_MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
            audio_quality_status="good",
        )
        assert result.risk_level == RiskLevel.MEDIUM
        assert 25 <= result.risk_score < 50
        assert result.decision == "UNKNOWN_NATURAL_SPEAKER"
        assert result.recommended_action == RecommendedAction.MONITOR

    def test_scenario_d_unknown_synthetic(self):
        # Scenario D: Low Speaker Similarity + High Synthetic Score
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.88,
            synth_classification="SYNTHETIC",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.10,
            spk_decision="NON_MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
            audio_quality_status="good",
        )
        assert result.risk_level == RiskLevel.HIGH
        assert 50 <= result.risk_score < 75
        assert result.decision == "UNKNOWN_SYNTHETIC_SPEAKER"
        assert result.recommended_action == RecommendedAction.STEP_UP_VERIFICATION


# ---------------------------------------------------------------------------
# 2. Intermediate & Ambiguous Combinations
# ---------------------------------------------------------------------------

class TestIntermediateCombinations:
    def test_high_speaker_medium_synthetic(self):
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.50,
            synth_classification="UNCERTAIN",
            synth_confidence="MEDIUM",
            synth_detector_id="aasist",
            spk_similarity=0.82,
            spk_decision="MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
        )
        assert result.risk_level == RiskLevel.HIGH
        assert result.decision == "SUSPICIOUS_ACOUSTIC_ANOMALY"

    def test_medium_speaker_high_synthetic(self):
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.85,
            synth_classification="SYNTHETIC",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.60,
            spk_decision="UNCERTAIN",
            spk_confidence="LOW",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
        )
        assert result.risk_level == RiskLevel.HIGH
        assert result.decision == "POTENTIAL_SYNTHETIC_ATTACK"

    def test_medium_speaker_medium_synthetic(self):
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.48,
            synth_classification="UNCERTAIN",
            synth_confidence="MEDIUM",
            synth_detector_id="aasist",
            spk_similarity=0.58,
            spk_decision="UNCERTAIN",
            spk_confidence="LOW",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
        )
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.decision == "AMBIGUOUS_EVIDENCE"


# ---------------------------------------------------------------------------
# 3. Audio Quality & Confidence Evaluation
# ---------------------------------------------------------------------------

class TestAudioQualityAndConfidence:
    def test_degraded_audio_decreases_confidence(self):
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=0.90,
            synth_classification="SYNTHETIC",
            synth_confidence="HIGH",
            synth_detector_id="aasist",
            spk_similarity=0.85,
            spk_decision="MATCH",
            spk_confidence="HIGH",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
            audio_quality_status="invalid",
        )
        assert result.decision == "DEGRADED_AUDIO_INCONCLUSIVE"
        assert result.evidence_confidence == "LOW"
        assert result.recommended_action == RecommendedAction.MONITOR

    def test_short_audio_penalizes_confidence(self):
        conf = EvidenceConfidenceEvaluator.evaluate_confidence(
            audio_quality_status="good",
            duration_seconds=1.2, # Short
            synthetic_confidence="HIGH",
            speaker_confidence="HIGH",
        )
        assert conf in ["MEDIUM", "LOW"]


# ---------------------------------------------------------------------------
# 4. Security & Boundary Robustness
# ---------------------------------------------------------------------------

class TestSecurityAndRobustness:
    def test_nan_and_infinite_handling(self):
        result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=float("nan"),
            synth_classification="UNCERTAIN",
            synth_confidence="LOW",
            synth_detector_id="aasist",
            spk_similarity=float("inf"),
            spk_decision="UNCERTAIN",
            spk_confidence="LOW",
            spk_profile_id="alice",
            spk_model_id="ecapa",
            duration_seconds=3.0,
        )
        assert isinstance(result.risk_score, int)
        assert 0 <= result.risk_score <= 100
        assert math.isfinite(result.signals["synthetic_detection"]["synthetic_score"])
        assert math.isfinite(result.signals["speaker_verification"]["speaker_similarity"])

    def test_determinism_on_identical_inputs(self):
        r1 = ControlledImpersonationRiskEngine.analyze_signals(
            0.75, "SYNTHETIC", "HIGH", "aasist", 0.70, "MATCH", "HIGH", "alice", "ecapa", 3.0
        )
        r2 = ControlledImpersonationRiskEngine.analyze_signals(
            0.75, "SYNTHETIC", "HIGH", "aasist", 0.70, "MATCH", "HIGH", "alice", "ecapa", 3.0
        )
        assert r1.risk_score == r2.risk_score
        assert r1.decision == r2.decision
        assert r1.recommended_action == r2.recommended_action


# ---------------------------------------------------------------------------
# 5. 4-Quadrant Benchmark Evaluation Framework
# ---------------------------------------------------------------------------

class TestFusionEvaluationMetrics:
    def test_four_quadrant_matrix_calculation(self):
        trials = [
            FusionTrial(trial_id="t1", audio_path="a1.wav", profile_id="p1", speaker_id="s1", speaker_label=SpeakerLabel.AUTHORIZED, synthetic_label=SyntheticLabel.NATURAL, attack_label=AttackLabel.LEGITIMATE),
            FusionTrial(trial_id="t2", audio_path="a2.wav", profile_id="p1", speaker_id="s1", speaker_label=SpeakerLabel.AUTHORIZED, synthetic_label=SyntheticLabel.SYNTHETIC, attack_label=AttackLabel.IMPERSONATION),
            FusionTrial(trial_id="t3", audio_path="a3.wav", profile_id="p1", speaker_id="s2", speaker_label=SpeakerLabel.UNAUTHORIZED, synthetic_label=SyntheticLabel.NATURAL, attack_label=AttackLabel.LEGITIMATE),
            FusionTrial(trial_id="t4", audio_path="a4.wav", profile_id="p1", speaker_id="s2", speaker_label=SpeakerLabel.UNAUTHORIZED, synthetic_label=SyntheticLabel.SYNTHETIC, attack_label=AttackLabel.IMPERSONATION),
        ]
        # Predictions: t1 -> LOW, t2 -> CRITICAL, t3 -> MEDIUM, t4 -> HIGH
        pred_levels = [RiskLevel.LOW, RiskLevel.CRITICAL, RiskLevel.MEDIUM, RiskLevel.HIGH]

        metrics = compute_fusion_metrics(trials, pred_levels)
        assert metrics.total_trials == 4
        assert metrics.four_quadrant_matrix.authorized_natural == 1
        assert metrics.four_quadrant_matrix.authorized_synthetic == 1
        assert metrics.four_quadrant_matrix.unauthorized_natural == 1
        assert metrics.four_quadrant_matrix.unauthorized_synthetic == 1
        assert metrics.accuracy == 1.0 # Perfect binary separation (LOW/MEDIUM vs HIGH/CRITICAL)
        assert metrics.f1_score == 1.0

    def test_fusion_speaker_leakage_detection(self):
        trials = [
            FusionTrial(trial_id="t1", audio_path="a.wav", profile_id="p1", speaker_id="leaked_spk", speaker_label=SpeakerLabel.AUTHORIZED, synthetic_label=SyntheticLabel.NATURAL, attack_label=AttackLabel.LEGITIMATE)
        ]
        is_leak, msg = FusionEvaluator.detect_speaker_leakage(trials, training_speakers={"leaked_spk"})
        assert is_leak is True
        assert "Speaker identity leakage" in msg


# ---------------------------------------------------------------------------
# 6. REST API Integration Tests
# ---------------------------------------------------------------------------

class TestRiskApiIntegration:
    @pytest.fixture
    def client(self):
        app = create_application()
        return TestClient(app)

    def test_api_config_and_provenance(self, client):
        res_cfg = client.get("/api/risk/config")
        assert res_cfg.status_code == 200
        assert "risk_bands" in res_cfg.json()
        assert "action_policies" in res_cfg.json()

        res_prov = client.get("/api/risk/provenance")
        assert res_prov.status_code == 200
        assert res_prov.json()["calibration_status"] == "NOT_CALIBRATED"
        assert "synthetic_detector" in res_prov.json()
        assert "speaker_verifier" in res_prov.json()

    def test_api_simulate_scenarios(self, client):
        # Simulate Scenario B (Targeted Clone)
        res_clone = client.post(
            "/api/risk/simulate",
            json={"synthetic_score": 0.95, "speaker_similarity": 0.85, "audio_quality": "good"}
        )
        assert res_clone.status_code == 200
        data_clone = res_clone.json()
        assert data_clone["risk_level"] == "CRITICAL"
        assert data_clone["decision"] == "POTENTIAL_VOICE_CLONE"
        assert data_clone["recommended_action"] == "BLOCK_OR_ESCALATE"

        # Simulate Scenario A (Legitimate)
        res_legit = client.post(
            "/api/risk/simulate",
            json={"synthetic_score": 0.10, "speaker_similarity": 0.85, "audio_quality": "good"}
        )
        assert res_legit.status_code == 200
        assert res_legit.json()["risk_level"] == "LOW"
        assert res_legit.json()["recommended_action"] == "ALLOW"

    def test_api_analyze_lifecycle_with_enrolled_profile(self, client):
        wav_bytes = generate_wav_bytes(duration=3.0, freq=220.0)

        # 1. Enroll profile
        enroll_res = client.post(
            "/api/speaker/enroll",
            data={"profile_id": "risk_test_spk"},
            files=[("files", ("enr.wav", wav_bytes, "audio/wav"))]
        )
        assert enroll_res.status_code == 200

        # 2. Run multi-modal risk analysis
        analyze_res = client.post(
            "/api/risk/analyze",
            data={"profile_id": "risk_test_spk"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")}
        )
        assert analyze_res.status_code == 200
        data = analyze_res.json()
        assert "risk_score" in data
        assert "recommended_action" in data
        assert "signals" in data
        assert data["signals"]["speaker_verification"]["profile_id"] == "risk_test_spk"
        assert data["privacy"]["raw_audio_persisted"] is False
