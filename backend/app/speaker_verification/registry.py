"""
Speaker Verification Registry and Service Singleton.
"""
import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List

from app.audio.schemas import AudioData
from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.embedding import SpeechBrainECAPAEncoder
from app.speaker_verification.similarity import CosineSimilarityMetric
from app.speaker_verification.calibration import SpeakerScoreCalibrator, SpeakerVerificationThreshold
from app.speaker_verification.decision import SpeakerDecisionEngine
from app.speaker_verification.enrollment import SpeakerEnrollmentManager
from app.speaker_verification.privacy import BiometricPrivacyPolicy
from app.speaker_verification.schemas import (
    VerificationResponse,
    AudioQualitySummary,
    SpeakerDecision,
    ScoreType,
)

logger = logging.getLogger(__name__)


class SpeakerVerificationService:
    """
    Central service orchestrating enrollment, embedding extraction, and verification.
    """
    _instance: Optional["SpeakerVerificationService"] = None

    def __init__(self, encoder: Optional[BaseSpeakerEncoder] = None):
        self.encoder = encoder or SpeechBrainECAPAEncoder()
        self.similarity_metric = CosineSimilarityMetric()
        self.threshold_config = SpeakerVerificationThreshold()
        self.calibrator = SpeakerScoreCalibrator(self.threshold_config)
        self.decision_engine = SpeakerDecisionEngine(self.calibrator)
        self.enrollment_manager = SpeakerEnrollmentManager(self.encoder)

    @classmethod
    def get_instance(cls) -> "SpeakerVerificationService":
        if cls._instance is None:
            logger.info("Initializing SpeakerVerificationService singleton with SpeechBrain ECAPA-TDNN...")
            cls._instance = cls()
        return cls._instance

    def verify(
        self,
        profile_id: str,
        audio_data: AudioData,
    ) -> VerificationResponse:
        """
        Verifies audio against an enrolled speaker profile.
        """
        start_time = time.perf_counter()

        profile = self.enrollment_manager.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Enrolled profile '{profile_id}' not found. Please enroll the profile first.")

        enrolled_embedding = profile["embedding"]

        # Validate input audio
        dur = audio_data.duration_seconds
        if dur < 0.5:
            raise ValueError(f"Verification audio duration ({dur:.2f}s) is too short. Minimum is 0.5s.")

        samples_np = audio_data.samples
        if len(samples_np) == 0:
            raise ValueError("Verification audio contains no samples.")

        rms = float(np.sqrt(np.mean(samples_np ** 2)))
        peak = float(np.max(np.abs(samples_np)))
        rms_db = float(20.0 * np.log10(max(rms, 1e-6)))
        peak_db = float(20.0 * np.log10(max(peak, 1e-6)))
        snr_est = max(0.0, float(rms_db - (-50.0)))
        clipping_detected = peak >= 0.999

        if rms < 0.001:
            raise ValueError("Verification audio is virtually silent (RMS < -60 dBFS).")

        # Extract embedding for verification sample
        verification_emb = self.encoder.embed(samples_np, sample_rate=16000)

        # Compute cosine similarity
        similarity = self.similarity_metric.compute_similarity(enrolled_embedding, verification_emb)

        # Render decision
        decision, confidence = self.decision_engine.evaluate(similarity)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        quality_summary = AudioQualitySummary(
            duration_seconds=round(dur, 2),
            sample_rate=audio_data.sample_rate,
            rms_db=round(rms_db, 2),
            peak_db=round(peak_db, 2),
            snr_estimate_db=round(snr_est, 2),
            clipping_detected=clipping_detected,
            silence_ratio=0.0,
        )

        privacy_meta = BiometricPrivacyPolicy.get_privacy_metadata(in_memory_only=True)

        return VerificationResponse(
            profile_id=profile_id,
            model_id=self.encoder.model_id,
            similarity_score=round(similarity, 4),
            score_type=ScoreType.COSINE_SIMILARITY,
            decision=decision,
            confidence_band=confidence,
            threshold=self.threshold_config.threshold,
            threshold_version=self.threshold_config.threshold_version,
            calibration_status=self.threshold_config.calibration_status,
            provisional_warning=(
                "PROVISIONAL — NOT SCIENTIFICALLY VALIDATED: Cosine similarity is not an identity probability. "
                "Threshold is uncalibrated."
            ),
            duration_seconds=round(dur, 2),
            latency_ms=round(latency_ms, 2),
            audio_quality=quality_summary,
            privacy=privacy_meta,
        )
