"""
REST API Endpoints for Multi-Layer Voice Integrity & Contextual Risk Intelligence.
"""
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from app.audio.processor import AudioProcessor
from app.detection.registry import DetectorRegistry
from app.speaker_verification.registry import SpeakerVerificationService
from app.prosody.analyzer import ProsodyAnalyzer
from app.context.schemas import (
    CallType,
    CallerTrust,
    RequestedAction,
    HistoricalRisk,
    ContextMetadata,
)
from app.context.engine import ContextualRiskEngine
from app.risk.schemas import (
    MultiLayerRiskAnalysisResult,
    MultiLayerRiskSimulationRequest,
)
from app.risk.multi_layer_fusion import MultiLayerImpersonationRiskEngine
from app.prosody.schemas import (
    ProsodyClassification,
    ProsodyAnalysisResult,
    ProsodyFeatureSummary,
    ProsodyEvidenceItem,
    ProsodySeverity,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contextual-risk", tags=["Contextual Risk Intelligence"])


@router.post(
    "/analyze",
    response_model=MultiLayerRiskAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-layer voice cloning detection, biometric verification, prosody analysis, and contextual risk scoring."
)
async def analyze_voice_integrity(
    file: UploadFile = File(..., description="Audio recording (WAV, MP3, FLAC, OGG)."),
    profile_id: Optional[str] = Form(None, description="Optional claimed speaker profile ID for biometric verification."),
    call_type: str = Form("NORMAL_CALL", description="Call interaction type."),
    caller_trust: str = Form("UNKNOWN_CALLER", description="Caller trust status."),
    requested_action: str = Form("INFORMATION_ONLY", description="Action requested during call."),
    transaction_amount: Optional[float] = Form(None, description="Optional monetary value of requested transaction."),
    historical_risk: str = Form("NONE", description="Historical risk indicator for account or phone number."),
):
    """
    Step 13 Flagship Endpoint:
    Combines AASIST synthetic voice detection, SpeechBrain ECAPA-TDNN biometric verification,
    deterministic acoustic prosody tracking, and business context risk into an explainable assessment.
    """
    start_time = time.perf_counter()

    # 1. Read and validate audio payload
    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio payload is empty.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading audio file: {str(e)}")

    # 2. Audio standardization (16 kHz mono float32)
    try:
        audio_data, quality_metrics = AudioProcessor.process(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

    # 3. Layer 1: Synthetic Voice Detection (AASIST)
    detector_reg = DetectorRegistry()
    detector = detector_reg.get_detector()
    try:
        detection_result = detector.predict(audio_data)
        synth_score = float(detection_result.score)
        synth_class = (
            detection_result.classification.value
            if hasattr(detection_result.classification, "value")
            else str(detection_result.classification)
        )
        synth_conf = detection_result.confidence_band
        meta = detector.metadata() if callable(detector.metadata) else detector.metadata
        synth_model_name = getattr(meta, "model_name", "AASIST")
    except Exception as e:
        logger.error("AASIST detection error: %s", str(e))
        synth_score = 0.5
        synth_class = "UNCERTAIN"
        synth_conf = "LOW"
        synth_model_name = "AASIST"

    # 4. Layer 2: Speaker Identity Verification (ECAPA-TDNN)
    speaker_service = SpeakerVerificationService.get_instance()
    spk_model_name = "SpeechBrain ECAPA-TDNN"
    spk_sim = 0.0
    spk_decision = "NON_MATCH"
    spk_confidence = "LOW"

    if profile_id and profile_id.strip():
        pid = profile_id.strip()
        try:
            verif = speaker_service.verify(profile_id=pid, audio_data=audio_data)
            spk_sim = float(verif.similarity_score)
            spk_decision = verif.decision.value if hasattr(verif.decision, "value") else str(verif.decision)
            spk_confidence = verif.confidence_band.value if hasattr(verif.confidence_band, "value") else str(verif.confidence_band)
            spk_model_name = verif.model_id
        except KeyError:
            # Claimed profile is not enrolled in system
            spk_sim = 0.0
            spk_decision = "PROFILE_NOT_FOUND"
            spk_confidence = "UNVERIFIED"
        except Exception as e:
            logger.warning(f"Speaker verification error: {str(e)}")
            spk_sim = 0.0
            spk_decision = "VERIFICATION_ERROR"
            spk_confidence = "LOW"
    else:
        # Anonymous / unverified call
        spk_sim = 0.0
        spk_decision = "ANONYMOUS_UNCLAIMED"
        spk_confidence = "UNVERIFIED"

    # 5. Layer 3: Prosodic & Behavioral Dynamics Analysis
    try:
        prosody_result = ProsodyAnalyzer.analyze(audio_data.samples, duration_seconds=audio_data.duration_seconds)
    except Exception as e:
        logger.error(f"Prosody extraction error: {str(e)}")
        prosody_result = ProsodyAnalyzer._insufficient_result(
            reason=f"Prosody extraction error: {str(e)}",
            duration=audio_data.duration_seconds,
            evidence_code="PROSODY_PROCESSING_ERROR",
        )

    # 6. Layer 4: Contextual Risk Intelligence Engine
    try:
        call_type_enum = CallType(call_type)
    except ValueError:
        call_type_enum = CallType.NORMAL_CALL

    try:
        caller_trust_enum = CallerTrust(caller_trust)
    except ValueError:
        caller_trust_enum = CallerTrust.UNKNOWN_CALLER

    try:
        requested_action_enum = RequestedAction(requested_action)
    except ValueError:
        requested_action_enum = RequestedAction.INFORMATION_ONLY

    try:
        historical_risk_enum = HistoricalRisk(historical_risk)
    except ValueError:
        historical_risk_enum = HistoricalRisk.NONE

    context_meta = ContextMetadata(
        call_type=call_type_enum,
        caller_trust=caller_trust_enum,
        requested_action=requested_action_enum,
        transaction_amount=transaction_amount,
        historical_risk=historical_risk_enum,
    )

    context_result = ContextualRiskEngine.evaluate(context_meta)

    # 7. Layer 5: Extended Multi-Layer Fusion
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    result = MultiLayerImpersonationRiskEngine.evaluate(
        synth_score=synth_score,
        synth_classification=synth_class,
        synth_confidence=synth_conf,
        spk_similarity=spk_sim,
        spk_decision=spk_decision,
        spk_confidence=spk_confidence,
        prosody_result=prosody_result,
        context_result=context_result,
        context_meta=context_meta,
        profile_id=profile_id.strip() if profile_id else None,
        synth_model_name=synth_model_name,
        spk_model_name=spk_model_name,
        latency_ms=total_latency_ms,
    )

    return result


@router.post(
    "/simulate",
    response_model=MultiLayerRiskAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Simulate multi-layer voice integrity analysis with arbitrary signals for UI demos and security policy testing."
)
async def simulate_voice_integrity(req: MultiLayerRiskSimulationRequest):
    """
    Developer & Demo Simulation Endpoint:
    Directly feeds controlled synthetic, speaker, prosodic, and contextual signals into the multi-layer engine.
    """
    start_time = time.perf_counter()

    # Map synthetic classification
    if req.synthetic_score < 0.35:
        synth_class = "NATURAL"
        synth_conf = "HIGH"
    elif req.synthetic_score < 0.65:
        synth_class = "UNCERTAIN"
        synth_conf = "MEDIUM"
    else:
        synth_class = "SYNTHETIC"
        synth_conf = "HIGH"

    # Map speaker decision
    if req.speaker_similarity >= 0.65:
        spk_dec = "MATCH"
        spk_conf = "HIGH"
    elif req.speaker_similarity >= 0.50:
        spk_dec = "UNCERTAIN"
        spk_conf = "MEDIUM"
    else:
        spk_dec = "NON_MATCH"
        spk_conf = "HIGH"

    # Construct simulated prosody result
    try:
        p_class = ProsodyClassification(req.prosody_classification)
    except ValueError:
        p_class = ProsodyClassification.NATURAL_VARIATION

    prosody_result = ProsodyAnalysisResult(
        classification=p_class,
        features=ProsodyFeatureSummary(
            pitch_mean_hz=145.0,
            pitch_std_hz=6.0 if p_class == ProsodyClassification.LOW_VARIATION else 24.0,
            pitch_min_hz=130.0 if p_class == ProsodyClassification.LOW_VARIATION else 95.0,
            pitch_max_hz=155.0 if p_class == ProsodyClassification.LOW_VARIATION else 210.0,
            pitch_range_hz=25.0 if p_class == ProsodyClassification.LOW_VARIATION else 115.0,
            pitch_variation_coef=0.04 if p_class == ProsodyClassification.LOW_VARIATION else 0.165,
            energy_rms_mean=0.045,
            energy_rms_std=0.008 if p_class == ProsodyClassification.LOW_VARIATION else 0.022,
            energy_dynamic_range_db=8.5 if p_class == ProsodyClassification.LOW_VARIATION else 26.5,
            voiced_unvoiced_ratio=1.4,
            speech_activity_ratio=0.82,
            pause_ratio=0.18,
            pause_count=3,
            pause_mean_duration_ms=210.0,
            pause_max_duration_ms=350.0,
            speech_rhythm_proxy=3.8,
            microvariation_jitter_proxy=0.045 if p_class != ProsodyClassification.UNUSUAL_PROSODY else 0.42,
            energy_delta_mean=0.005,
        ),
        quality_score=0.90,
        confidence="HIGH",
        evidence=[
            ProsodyEvidenceItem(
                code="PROSODY_SIMULATED",
                severity=ProsodySeverity.INFO,
                message=f"Simulated prosody state: {p_class.value}."
            )
        ],
        duration_seconds=3.5,
        voiced_frames_count=180,
        total_frames_count=350,
    )

    # Parse context
    try:
        c_type = CallType(req.call_type)
    except ValueError:
        c_type = CallType.NORMAL_CALL

    try:
        c_trust = CallerTrust(req.caller_trust)
    except ValueError:
        c_trust = CallerTrust.UNKNOWN_CALLER

    try:
        r_action = RequestedAction(req.requested_action)
    except ValueError:
        r_action = RequestedAction.INFORMATION_ONLY

    try:
        h_risk = HistoricalRisk(req.historical_risk)
    except ValueError:
        h_risk = HistoricalRisk.NONE

    context_meta = ContextMetadata(
        call_type=c_type,
        caller_trust=c_trust,
        requested_action=r_action,
        transaction_amount=req.transaction_amount,
        historical_risk=h_risk,
    )

    context_result = ContextualRiskEngine.evaluate(context_meta)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return MultiLayerImpersonationRiskEngine.evaluate(
        synth_score=req.synthetic_score,
        synth_classification=synth_class,
        synth_confidence=synth_conf,
        spk_similarity=req.speaker_similarity,
        spk_decision=spk_dec,
        spk_confidence=spk_conf,
        prosody_result=prosody_result,
        context_result=context_result,
        context_meta=context_meta,
        profile_id=req.profile_id,
        synth_model_name="AASIST (Simulated)",
        spk_model_name="SpeechBrain ECAPA-TDNN (Simulated)",
        latency_ms=latency_ms,
    )


@router.get(
    "/scenarios",
    summary="Get realistic pre-configured banking and enterprise attack scenarios for instant demonstration."
)
async def list_attack_scenarios() -> List[Dict[str, Any]]:
    """
    Returns pre-configured demonstration scenarios illustrating different layers of threat.
    """
    return [
        {
            "id": "ceo_wire_fraud",
            "name": "CEO Voice Cloning Wire Transfer Attack",
            "description": "High-fidelity AI voice clone of CEO calling treasury requesting urgent $250,000 foreign supplier payment.",
            "synthetic_score": 0.94,
            "speaker_similarity": 0.88,
            "prosody_classification": "LOW_VARIATION",
            "call_type": "FINANCIAL_TRANSACTION",
            "caller_trust": "VIP_OR_EXECUTIVE",
            "requested_action": "FUND_TRANSFER",
            "transaction_amount": 250000.0,
            "historical_risk": "MEDIUM",
            "expected_outcome": {
                "risk_level": "CRITICAL",
                "recommended_action": "BLOCK_TRANSACTION_AND_ESCALATE"
            }
        },
        {
            "id": "it_helpdesk_creds",
            "name": "IT Helpdesk Credential Takeover (Spear-Phishing)",
            "description": "Attacker using cloned voice of Senior VP demanding immediate password reset on AWS admin console.",
            "synthetic_score": 0.89,
            "speaker_similarity": 0.79,
            "prosody_classification": "LOW_VARIATION",
            "call_type": "PRIVILEGED_ACCESS",
            "caller_trust": "VIP_OR_EXECUTIVE",
            "requested_action": "CREDENTIAL_RESET",
            "transaction_amount": None,
            "historical_risk": "NONE",
            "expected_outcome": {
                "risk_level": "HIGH",
                "recommended_action": "MFA_REQUIRED"
            }
        },
        {
            "id": "legit_customer_balance",
            "name": "Legitimate Customer Account Inquiry",
            "description": "Genuine customer calling from registered phone number to check recent checking account balance.",
            "synthetic_score": 0.06,
            "speaker_similarity": 0.84,
            "prosody_classification": "NATURAL_VARIATION",
            "call_type": "NORMAL_CALL",
            "caller_trust": "VERIFIED_CONTACT",
            "requested_action": "INFORMATION_ONLY",
            "transaction_amount": None,
            "historical_risk": "NONE",
            "expected_outcome": {
                "risk_level": "LOW",
                "recommended_action": "ALLOW"
            }
        },
        {
            "id": "synthetic_robocall_survey",
            "name": "Automated Synthetic Robocall (Generic)",
            "description": "Mass-dialing automated robocall using basic synthetic text-to-speech engine.",
            "synthetic_score": 0.92,
            "speaker_similarity": 0.12,
            "prosody_classification": "LOW_VARIATION",
            "call_type": "NORMAL_CALL",
            "caller_trust": "UNKNOWN_CALLER",
            "requested_action": "INFORMATION_ONLY",
            "transaction_amount": None,
            "historical_risk": "NONE",
            "expected_outcome": {
                "risk_level": "MEDIUM",
                "recommended_action": "MONITOR"
            }
        }
    ]
