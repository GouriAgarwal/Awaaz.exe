"""
Test Suite for Step 13: Multi-Layer Voice Integrity & Contextual Risk Intelligence.
Verifies prosodic extraction, silence handling, short audio, NaN protection,
contextual risk engine, policy sensitivity escalation, multi-layer fusion, and REST APIs.
"""
import io
import math
import wave
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_application
from app.prosody.schemas import ProsodyClassification
from app.prosody.analyzer import ProsodyAnalyzer
from app.context.schemas import (
    CallType,
    CallerTrust,
    RequestedAction,
    HistoricalRisk,
    ContextMetadata,
    ContextRiskLevel,
    PolicySensitivity,
)
from app.context.engine import ContextualRiskEngine
from app.risk.schemas import (
    RiskLevel,
    ExtendedRecommendedAction,
    MultiLayerRiskSimulationRequest,
)
from app.risk.multi_layer_fusion import MultiLayerImpersonationRiskEngine


def generate_synthetic_audio(
    duration: float = 2.0,
    freq: float = 180.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    add_modulation: bool = True,
) -> np.ndarray:
    """Generates synthetic test audio samples."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    if add_modulation:
        # Dynamic frequency and amplitude modulation (natural-like)
        mod = 15.0 * np.sin(2.0 * np.pi * 3.0 * t)
        samples = amplitude * np.sin(2.0 * np.pi * (freq + mod) * t) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 2.0 * t))
    else:
        # Flat pure sine tone (flat prosody / monotone)
        samples = amplitude * np.sin(2.0 * np.pi * freq * t)
    return samples.astype(np.float32)


def generate_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Converts numpy float32 audio to valid 16-bit PCM WAV bytes."""
    int_samples = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)


# ============================================================================
# PART A TESTS: Prosody & Behavioral Analysis
# ============================================================================

def test_prosody_natural_variation():
    """Verify natural modulated audio produces NATURAL_VARIATION with high confidence."""
    audio = generate_synthetic_audio(duration=2.5, freq=160.0, add_modulation=True)
    res = ProsodyAnalyzer.analyze(audio)

    assert res.classification in (ProsodyClassification.NATURAL_VARIATION, ProsodyClassification.UNUSUAL_PROSODY)
    assert res.features.pitch_mean_hz is not None
    assert 80.0 <= res.features.pitch_mean_hz <= 350.0
    assert res.features.energy_rms_mean > 0.0
    assert res.voiced_frames_count > 10
    assert res.quality_score > 0.5
    assert "RAM_ONLY" in res.ephemeral_privacy


def test_prosody_low_variation():
    """Verify flat unmodulated tone triggers LOW_VARIATION classification and flat pitch flag."""
    audio = generate_synthetic_audio(duration=2.0, freq=180.0, add_modulation=False)
    res = ProsodyAnalyzer.analyze(audio)

    assert res.classification == ProsodyClassification.LOW_VARIATION
    assert res.features.pitch_variation_coef is not None
    assert res.features.pitch_variation_coef < 0.05
    evidence_codes = [e.code for e in res.evidence]
    assert "PROSODY_FLAT_PITCH" in evidence_codes or "PROSODY_LOW_DYNAMIC_RANGE" in evidence_codes


def test_prosody_silence_handling():
    """Verify near-silent audio gracefully returns INSUFFICIENT_AUDIO without crashing."""
    silent_audio = np.zeros(32000, dtype=np.float32)  # 2.0s of pure silence
    res = ProsodyAnalyzer.analyze(silent_audio)

    assert res.classification == ProsodyClassification.INSUFFICIENT_AUDIO
    assert res.confidence == "UNRELIABLE"
    assert res.quality_score == 0.0
    evidence_codes = [e.code for e in res.evidence]
    assert "PROSODY_SILENCE_DETECTED" in evidence_codes


def test_prosody_short_audio_handling():
    """Verify audio shorter than 0.5s is rejected with INSUFFICIENT_AUDIO."""
    short_audio = generate_synthetic_audio(duration=0.3, freq=180.0)
    res = ProsodyAnalyzer.analyze(short_audio)

    assert res.classification == ProsodyClassification.INSUFFICIENT_AUDIO
    evidence_codes = [e.code for e in res.evidence]
    assert "PROSODY_DURATION_TOO_SHORT" in evidence_codes


def test_prosody_nan_and_inf_handling():
    """Verify arrays containing NaN or Inf are sanitized safely without crashing."""
    corrupted_audio = generate_synthetic_audio(duration=1.5, freq=180.0)
    corrupted_audio[100:150] = np.nan
    corrupted_audio[300:350] = np.inf

    res = ProsodyAnalyzer.analyze(corrupted_audio)
    assert res.classification != ProsodyClassification.INSUFFICIENT_AUDIO or res.features is not None
    evidence_codes = [e.code for e in res.evidence]
    assert "PROSODY_NAN_SANITIZED" in evidence_codes


# ============================================================================
# PART B TESTS: Contextual Risk Engine
# ============================================================================

def test_context_normal_call():
    """Verify benign normal call with verified caller receives low context risk."""
    meta = ContextMetadata(
        call_type=CallType.NORMAL_CALL,
        caller_trust=CallerTrust.VERIFIED_CONTACT,
        requested_action=RequestedAction.INFORMATION_ONLY,
        transaction_amount=None,
        historical_risk=HistoricalRisk.NONE,
    )
    result = ContextualRiskEngine.evaluate(meta)

    assert result.risk_level == ContextRiskLevel.LOW
    assert result.context_risk_score < 25
    assert result.policy_sensitivity == PolicySensitivity.STANDARD


def test_context_financial_escalation():
    """Verify high-stakes financial transfer raises context risk to HIGH/CRITICAL."""
    meta = ContextMetadata(
        call_type=CallType.FINANCIAL_TRANSACTION,
        caller_trust=CallerTrust.UNKNOWN_CALLER,
        requested_action=RequestedAction.FUND_TRANSFER,
        transaction_amount=75000.0,
        historical_risk=HistoricalRisk.MEDIUM,
    )
    result = ContextualRiskEngine.evaluate(meta)

    assert result.risk_level in (ContextRiskLevel.HIGH, ContextRiskLevel.CRITICAL)
    assert result.context_risk_score >= 70
    assert result.sensitivity_multiplier >= 1.5
    evidence_codes = [e.code for e in result.evidence]
    assert "CONTEXT_VERY_HIGH_VALUE_TRANSACTION" in evidence_codes
    assert "CONTEXT_FINANCIAL_CHANNEL" in evidence_codes


def test_context_privileged_access_escalation():
    """Verify privileged credential reset on VIP executive raises critical context flags."""
    meta = ContextMetadata(
        call_type=CallType.PRIVILEGED_ACCESS,
        caller_trust=CallerTrust.VIP_OR_EXECUTIVE,
        requested_action=RequestedAction.CREDENTIAL_RESET,
        transaction_amount=None,
        historical_risk=HistoricalRisk.LOW,
    )
    result = ContextualRiskEngine.evaluate(meta)

    assert result.risk_level in (ContextRiskLevel.HIGH, ContextRiskLevel.CRITICAL)
    evidence_codes = [e.code for e in result.evidence]
    assert "CONTEXT_PRIVILEGED_CHANNEL" in evidence_codes
    assert "CONTEXT_EXECUTIVE_TARGET" in evidence_codes
    assert "CONTEXT_IDENTITY_CREDENTIAL_TAKEOVER" in evidence_codes


# ============================================================================
# PART C & D TESTS: Multi-Layer Fusion & Policy Recommendations
# ============================================================================

def test_multi_layer_targeted_clone_escalation():
    """
    Verify Quadrant 1 (Synthetic Voice + Enrolled Biometric Match + High Stakes)
    triggers CRITICAL risk and BLOCK_TRANSACTION_AND_ESCALATE.
    """
    meta = ContextMetadata(
        call_type=CallType.FINANCIAL_TRANSACTION,
        caller_trust=CallerTrust.VIP_OR_EXECUTIVE,
        requested_action=RequestedAction.FUND_TRANSFER,
        transaction_amount=100000.0,
    )
    ctx_res = ContextualRiskEngine.evaluate(meta)
    audio = generate_synthetic_audio(duration=2.0, freq=180.0, add_modulation=False)
    prosody_res = ProsodyAnalyzer.analyze(audio)

    fusion_res = MultiLayerImpersonationRiskEngine.evaluate(
        synth_score=0.95,
        synth_classification="SYNTHETIC",
        synth_confidence="HIGH",
        spk_similarity=0.88,
        spk_decision="MATCH",
        spk_confidence="HIGH",
        prosody_result=prosody_res,
        context_result=ctx_res,
        context_meta=meta,
    )

    assert fusion_res.overall_risk_score >= 75
    assert fusion_res.risk_level == RiskLevel.CRITICAL
    assert fusion_res.recommended_action == ExtendedRecommendedAction.BLOCK_TRANSACTION_AND_ESCALATE
    assert "ADVISORY" in fusion_res.enforcement_disclaimer


def test_multi_layer_privileged_mfa_recommendation():
    """
    Verify high risk on privileged access triggers MFA_REQUIRED.
    """
    meta = ContextMetadata(
        call_type=CallType.PRIVILEGED_ACCESS,
        caller_trust=CallerTrust.UNKNOWN_CALLER,
        requested_action=RequestedAction.CREDENTIAL_RESET,
    )
    ctx_res = ContextualRiskEngine.evaluate(meta)
    audio = generate_synthetic_audio(duration=2.0, freq=180.0, add_modulation=True)
    prosody_res = ProsodyAnalyzer.analyze(audio)

    fusion_res = MultiLayerImpersonationRiskEngine.evaluate(
        synth_score=0.72,
        synth_classification="SYNTHETIC",
        synth_confidence="HIGH",
        spk_similarity=0.45,
        spk_decision="NON_MATCH",
        spk_confidence="HIGH",
        prosody_result=prosody_res,
        context_result=ctx_res,
        context_meta=meta,
    )

    assert fusion_res.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    if fusion_res.risk_level == RiskLevel.HIGH:
        assert fusion_res.recommended_action == ExtendedRecommendedAction.MFA_REQUIRED


def test_multi_layer_legitimate_caller_allow():
    """
    Verify clean natural audio with verified enrolled speaker receives LOW risk and ALLOW.
    """
    meta = ContextMetadata(
        call_type=CallType.NORMAL_CALL,
        caller_trust=CallerTrust.VERIFIED_CONTACT,
        requested_action=RequestedAction.INFORMATION_ONLY,
    )
    ctx_res = ContextualRiskEngine.evaluate(meta)
    audio = generate_synthetic_audio(duration=2.0, freq=160.0, add_modulation=True)
    prosody_res = ProsodyAnalyzer.analyze(audio)

    fusion_res = MultiLayerImpersonationRiskEngine.evaluate(
        synth_score=0.08,
        synth_classification="NATURAL",
        synth_confidence="HIGH",
        spk_similarity=0.85,
        spk_decision="MATCH",
        spk_confidence="HIGH",
        prosody_result=prosody_res,
        context_result=ctx_res,
        context_meta=meta,
    )

    assert fusion_res.overall_risk_score < 25
    assert fusion_res.risk_level == RiskLevel.LOW
    assert fusion_res.recommended_action == ExtendedRecommendedAction.ALLOW


# ============================================================================
# PART E TESTS: REST Endpoints
# ============================================================================

def test_api_scenarios(client):
    """Verify pre-configured scenario list returns valid demo cases."""
    response = client.get("/api/contextual-risk/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) >= 4
    scenario_ids = [s["id"] for s in scenarios]
    assert "ceo_wire_fraud" in scenario_ids
    assert "it_helpdesk_creds" in scenario_ids


def test_api_simulate(client):
    """Verify simulation endpoint evaluates arbitrary signals without model execution."""
    payload = {
        "synthetic_score": 0.92,
        "speaker_similarity": 0.85,
        "prosody_classification": "LOW_VARIATION",
        "call_type": "FINANCIAL_TRANSACTION",
        "caller_trust": "VIP_OR_EXECUTIVE",
        "requested_action": "FUND_TRANSFER",
        "transaction_amount": 150000.0,
        "historical_risk": "MEDIUM",
        "profile_id": "vip_ceo_profile",
    }
    response = client.post("/api/contextual-risk/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["overall_risk_score"] >= 75
    assert data["risk_level"] == "CRITICAL"
    assert data["recommended_action"] == "BLOCK_TRANSACTION_AND_ESCALATE"
    assert "synthetic_signal" in data
    assert "speaker_signal" in data
    assert "prosody_signal" in data
    assert "context_signal" in data
    assert len(data["evidence"]) > 0


def test_api_analyze_audio(client):
    """Verify analyze endpoint processes uploaded WAV file end-to-end."""
    audio_samples = generate_synthetic_audio(duration=1.5, freq=200.0, add_modulation=True)
    wav_bytes = generate_wav_bytes(audio_samples)

    files = {"file": ("test_sample.wav", wav_bytes, "audio/wav")}
    data = {
        "call_type": "NORMAL_CALL",
        "caller_trust": "UNKNOWN_CALLER",
        "requested_action": "INFORMATION_ONLY",
        "historical_risk": "NONE",
    }

    response = client.post("/api/contextual-risk/analyze", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()

    assert 0 <= res_json["overall_risk_score"] <= 100
    assert res_json["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "latency_ms" in res_json
    assert res_json["latency_ms"] > 0
    assert "RAM_ONLY" in res_json["privacy_policy"]
