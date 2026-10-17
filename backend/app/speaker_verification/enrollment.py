"""
Speaker Enrollment Subsystem with Multi-Sample Ingestion and L2 Centroid Aggregation.
"""
import datetime
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.audio.schemas import AudioData
from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.schemas import (
    AudioQualitySummary,
    EnrollmentResponse,
    SpeakerProfileSummary,
)
from app.speaker_verification.privacy import BiometricPrivacyPolicy

logger = logging.getLogger(__name__)

MIN_SAMPLE_DURATION_S = 1.0
RECOMMENDED_TOTAL_DURATION_S = 10.0
MAX_TOTAL_DURATION_S = 120.0


class SpeakerEnrollmentManager:
    """
    Manages in-memory speaker enrollment profiles and centroid embedding aggregation.
    """
    def __init__(self, encoder: BaseSpeakerEncoder):
        self.encoder = encoder
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves internal profile with embedding vector (internal use only).
        """
        return self._profiles.get(profile_id)

    def list_profiles(self) -> List[SpeakerProfileSummary]:
        """
        Returns sanitized list of enrolled profiles without embedding vectors.
        """
        summaries = []
        for pid, p in self._profiles.items():
            summaries.append(
                SpeakerProfileSummary(
                    profile_id=p["profile_id"],
                    model_id=p["model_id"],
                    embedding_dimension=p["embedding_dimension"],
                    sample_count=p["sample_count"],
                    total_audio_duration_seconds=p["total_audio_duration_seconds"],
                    created_at=p["created_at"],
                    updated_at=p["updated_at"],
                    in_memory_only=p.get("in_memory_only", True),
                )
            )
        return summaries

    def delete_profile(self, profile_id: str) -> bool:
        """
        Deletes an enrolled profile.
        """
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            logger.info(f"Deleted speaker profile: {profile_id}")
            return True
        return False

    def enroll(
        self,
        profile_id: str,
        audio_samples: List[AudioData],
    ) -> EnrollmentResponse:
        """
        Enrolls a speaker using one or more audio samples.
        Aggregates embeddings into an L2-normalized centroid.
        """
        if not profile_id or not profile_id.strip():
            raise ValueError("Profile ID must be a non-empty string.")

        profile_id = profile_id.strip()

        if not audio_samples:
            raise ValueError("At least one audio sample must be provided for enrollment.")

        embeddings: List[np.ndarray] = []
        total_duration = 0.0
        total_rms_sq = 0.0
        max_peak = 0.0
        clipping_detected = False
        sample_rates = set()

        for idx, sample in enumerate(audio_samples):
            if sample.sample_rate != 16000:
                sample_rates.add(sample.sample_rate)

            dur = sample.duration_seconds
            if dur < MIN_SAMPLE_DURATION_S:
                raise ValueError(
                    f"Sample #{idx+1} duration ({dur:.2f}s) is too short. Minimum is {MIN_SAMPLE_DURATION_S:.1f}s."
                )

            total_duration += dur
            samples_np = sample.samples
            if len(samples_np) == 0:
                raise ValueError(f"Sample #{idx+1} contains no audio samples.")

            rms = float(np.sqrt(np.mean(samples_np ** 2)))
            peak = float(np.max(np.abs(samples_np)))
            max_peak = max(max_peak, peak)
            total_rms_sq += (rms ** 2) * dur

            if peak >= 0.999:
                clipping_detected = True

            if rms < 0.001:
                raise ValueError(f"Sample #{idx+1} is virtually silent (RMS < -60 dBFS).")

            # Extract embedding
            emb = self.encoder.embed(samples_np, sample_rate=16000)
            embeddings.append(emb)

        if sample_rates:
            raise ValueError(f"Inconsistent or unsupported sample rates: {sample_rates}. Expected 16000 Hz.")

        if total_duration > MAX_TOTAL_DURATION_S:
            raise ValueError(
                f"Total enrollment audio ({total_duration:.1f}s) exceeds maximum allowable ({MAX_TOTAL_DURATION_S}s)."
            )

        # Aggregate L2 centroid
        stacked = np.stack(embeddings, axis=0) # (M, D)
        centroid = np.mean(stacked, axis=0) # (D,)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-12:
            raise ValueError("Aggregated centroid embedding has zero norm.")
        normalized_centroid = (centroid / centroid_norm).astype(np.float32)

        avg_rms = float(np.sqrt(total_rms_sq / max(total_duration, 1e-6)))
        rms_db = float(20.0 * np.log10(max(avg_rms, 1e-6)))
        peak_db = float(20.0 * np.log10(max(max_peak, 1e-6)))
        snr_est = max(0.0, float(rms_db - (-50.0)))

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        is_update = profile_id in self._profiles

        self._profiles[profile_id] = {
            "profile_id": profile_id,
            "model_id": self.encoder.model_id,
            "embedding_dimension": self.encoder.embedding_dimension,
            "embedding": normalized_centroid,
            "sample_count": len(audio_samples),
            "total_audio_duration_seconds": round(total_duration, 2),
            "created_at": self._profiles[profile_id]["created_at"] if is_update else now_iso,
            "updated_at": now_iso,
            "in_memory_only": True,
        }

        logger.info(
            f"Enrolled profile '{profile_id}' ({len(audio_samples)} samples, {total_duration:.2f}s speech, "
            f"model={self.encoder.model_id})"
        )

        warning = None
        if total_duration < RECOMMENDED_TOTAL_DURATION_S:
            warning = (
                f"Enrollment audio duration ({total_duration:.1f}s) is below the recommended "
                f"{RECOMMENDED_TOTAL_DURATION_S:.0f}s. Adding more speech will improve verification accuracy."
            )

        quality_summary = AudioQualitySummary(
            duration_seconds=round(total_duration, 2),
            sample_rate=16000,
            rms_db=round(rms_db, 2),
            peak_db=round(peak_db, 2),
            snr_estimate_db=round(snr_est, 2),
            clipping_detected=clipping_detected,
            silence_ratio=0.0,
        )

        privacy_meta = BiometricPrivacyPolicy.get_privacy_metadata(in_memory_only=True)

        return EnrollmentResponse(
            success=True,
            profile_id=profile_id,
            sample_count=len(audio_samples),
            total_audio_duration_seconds=round(total_duration, 2),
            audio_quality=quality_summary,
            privacy=privacy_meta,
            message=f"Speaker profile '{profile_id}' enrolled successfully ({len(audio_samples)} samples).",
            warning=warning,
        )
