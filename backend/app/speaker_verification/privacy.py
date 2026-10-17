"""
Biometric Privacy Controls for VoiceShield Speaker Subsystem.
"""
import logging
from typing import Dict, Any

from app.speaker_verification.schemas import PrivacyMetadata

logger = logging.getLogger(__name__)


class BiometricPrivacyPolicy:
    """
    Enforces zero raw audio persistence and strict biometric privacy standards.
    """

    @staticmethod
    def get_privacy_metadata(in_memory_only: bool = True) -> PrivacyMetadata:
        """
        Returns the standard privacy metadata block.
        """
        return PrivacyMetadata(
            raw_audio_persisted=False,
            embeddings_logged=False,
            in_memory_only=in_memory_only,
            policy=(
                "VoiceShield Biometric Privacy: Raw voice recordings are processed in memory "
                "and immediately discarded. Raw audio is never persisted to disk or logged. "
                "Biometric embedding vectors are not logged or exposed in client responses."
            )
        )

    @staticmethod
    def sanitize_profile_for_client(profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips numerical embedding vectors before returning profile information to clients.
        """
        return {
            "profile_id": profile.get("profile_id"),
            "model_id": profile.get("model_id"),
            "embedding_dimension": profile.get("embedding_dimension"),
            "sample_count": profile.get("sample_count", 0),
            "total_audio_duration_seconds": profile.get("total_audio_duration_seconds", 0.0),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
            "in_memory_only": profile.get("in_memory_only", True),
        }
