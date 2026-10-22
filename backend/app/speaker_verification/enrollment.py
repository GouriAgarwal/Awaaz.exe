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


from pathlib import Path
import json

class SpeakerEnrollmentManager:
    """
    Manages speaker enrollment profiles, centroid embedding aggregation,
    and local persistent storage across backend restarts.
    """
    def __init__(self, encoder: BaseSpeakerEncoder, storage_dir: Optional[Path] = None):
        self.encoder = encoder
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self.storage_dir = storage_dir
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_profiles()

    def _load_persisted_profiles(self) -> None:
        """
        Loads all persisted speaker profiles from the storage directory.
        """
        if not self.storage_dir or not self.storage_dir.exists():
            return

        for meta_path in self.storage_dir.glob("*.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                profile_id = meta.get("profile_id")
                emb_filename = meta.get("embedding_file")
                if not profile_id or not emb_filename:
                    continue

                emb_path = self.storage_dir / emb_filename
                if not emb_path.exists():
                    logger.warning(f"Embedding file {emb_path} missing for profile {profile_id}")
                    continue

                emb = np.load(emb_path).astype(np.float32)
                self._profiles[profile_id] = {
                    "profile_id": profile_id,
                    "model_id": meta.get("model_id", self.encoder.model_id),
                    "embedding_dimension": meta.get("embedding_dimension", len(emb)),
                    "embedding": emb,
                    "sample_count": meta.get("sample_count", 1),
                    "total_audio_duration_seconds": meta.get("total_audio_duration_seconds", 0.0),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "in_memory_only": False,
                }
                logger.info(f"Loaded persisted speaker profile '{profile_id}' from {meta_path}")
            except Exception as e:
                logger.error(f"Failed loading persisted profile from {meta_path}: {e}")

    def _persist_profile(self, profile_id: str, profile_data: Dict[str, Any]) -> None:
        """
        Persists profile metadata and embedding array to storage_dir.
        """
        if not self.storage_dir:
            return

        try:
            emb_filename = f"{profile_id}.npy"
            meta_filename = f"{profile_id}.json"

            np.save(self.storage_dir / emb_filename, profile_data["embedding"])

            meta = {
                "profile_id": profile_id,
                "model_id": profile_data["model_id"],
                "embedding_dimension": profile_data["embedding_dimension"],
                "embedding_file": emb_filename,
                "sample_count": profile_data["sample_count"],
                "total_audio_duration_seconds": profile_data["total_audio_duration_seconds"],
                "created_at": profile_data["created_at"],
                "updated_at": profile_data["updated_at"],
            }
            with open(self.storage_dir / meta_filename, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Persisted profile '{profile_id}' to {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to persist profile '{profile_id}': {e}")

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
        Deletes an enrolled profile from memory and disk.
        """
        deleted = False
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            deleted = True

        if self.storage_dir:
            meta_path = self.storage_dir / f"{profile_id}.json"
            emb_path = self.storage_dir / f"{profile_id}.npy"
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
            if emb_path.exists():
                emb_path.unlink()
                deleted = True

        if deleted:
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
            "in_memory_only": self.storage_dir is None,
        }

        if self.storage_dir is not None:
            self._persist_profile(profile_id, self._profiles[profile_id])

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

        privacy_meta = BiometricPrivacyPolicy.get_privacy_metadata(in_memory_only=(self.storage_dir is None))

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
