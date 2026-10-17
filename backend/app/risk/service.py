"""
Service Layer for Impersonation Risk Analysis.
Coordinates audio ingestion, AASIST synthetic voice detection, ECAPA-TDNN speaker verification, and fusion.
"""
import time
import logging
from typing import Optional, Dict, Any

from app.audio.processor import AudioProcessor
from app.audio.schemas import AudioData
from app.detection.registry import DetectorRegistry
from app.speaker_verification.registry import SpeakerVerificationService
from app.risk.schemas import RiskAnalysisResult, ContextualSignals, RiskSimulationRequest
from app.risk.fusion import ControlledImpersonationRiskEngine

logger = logging.getLogger(__name__)


class ImpersonationRiskService:
    """
    Singleton service orchestrating end-to-end multi-modal voice risk analysis.
    """
    _instance: Optional["ImpersonationRiskService"] = None

    def __init__(self):
        self.detector_registry = DetectorRegistry()
        self.speaker_service = SpeakerVerificationService.get_instance()

    @classmethod
    def get_instance(cls) -> "ImpersonationRiskService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_audio(
        self,
        file_bytes: bytes,
        profile_id: str,
        contextual_signals: Optional[ContextualSignals] = None,
    ) -> RiskAnalysisResult:
        """
        Executes complete multi-modal analysis on incoming audio payload.
        """
        start_time = time.perf_counter()

        # 1. Step 2: Audio Ingestion & Standardization
        audio_data, quality_metrics = AudioProcessor.process(file_bytes)

        # 2. Step 4/6: Pretrained AASIST Synthetic Detection
        detector = self.detector_registry.get_detector()
        detection_result = detector.predict(audio_data)

        # 3. Step 8: Pretrained ECAPA-TDNN Speaker Identity Verification
        verification_response = self.speaker_service.verify(profile_id=profile_id, audio_data=audio_data)

        # Measure end-to-end compute latency
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        meta = detector.metadata() if callable(detector.metadata) else detector.metadata
        synth_model_name = getattr(meta, "model_name", "AASIST")

        # 4. Step 9: Controlled Fusion
        risk_result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=detection_result.score,
            synth_classification=detection_result.classification.value if hasattr(detection_result.classification, "value") else str(detection_result.classification),
            synth_confidence=detection_result.confidence_band,
            synth_detector_id=synth_model_name,
            spk_similarity=verification_response.similarity_score,
            spk_decision=verification_response.decision.value if hasattr(verification_response.decision, "value") else str(verification_response.decision),
            spk_confidence=verification_response.confidence_band.value if hasattr(verification_response.confidence_band, "value") else str(verification_response.confidence_band),
            spk_profile_id=profile_id,
            spk_model_id=verification_response.model_id,
            duration_seconds=audio_data.duration_seconds,
            audio_quality_status=quality_metrics.quality,
            contextual_signals=contextual_signals,
            latency_ms=total_latency_ms,
        )

        return risk_result

    def simulate(self, req: RiskSimulationRequest) -> RiskAnalysisResult:
        """
        Simulates fusion evaluation given controlled synthetic and speaker signals (for UI testing and developer verification).
        """
        start_time = time.perf_counter()

        # Map simulated score to classification
        if req.synthetic_score < 0.35:
            synth_class = "NATURAL"
            synth_conf = "HIGH"
        elif req.synthetic_score < 0.65:
            synth_class = "UNCERTAIN"
            synth_conf = "MEDIUM"
        else:
            synth_class = "SYNTHETIC"
            synth_conf = "HIGH"

        # Map simulated similarity to decision
        if req.speaker_similarity >= 0.65:
            spk_dec = "MATCH"
            spk_conf = "HIGH"
        elif req.speaker_similarity >= 0.55:
            spk_dec = "UNCERTAIN"
            spk_conf = "LOW"
        else:
            spk_dec = "NON_MATCH"
            spk_conf = "HIGH"

        sim_latency = (time.perf_counter() - start_time) * 1000.0

        return ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=req.synthetic_score,
            synth_classification=synth_class,
            synth_confidence=synth_conf,
            synth_detector_id="simulation_synthetic_source",
            spk_similarity=req.speaker_similarity,
            spk_decision=spk_dec,
            spk_confidence=spk_conf,
            spk_profile_id=req.profile_id or "simulated_profile",
            spk_model_id="simulation_speaker_source",
            duration_seconds=3.0,
            audio_quality_status=req.audio_quality or "good",
            contextual_signals=None,
            latency_ms=sim_latency,
        )
