"""Health, Liveness, and Multi-Factor Readiness Probes (Step 12)."""

from datetime import datetime, timezone
import time
from typing import Dict
from app.core.config import settings
from app.detection.pretrained.loader import PretrainedModelLoader
from app.observability.metrics import MetricsRegistry
from app.observability.schemas import (
    ComponentHealth,
    LivenessResponse,
    OperationalStatusResponse,
    ReadinessResponse,
    ReadinessStatus,
)
from app.speaker_verification.model_loader import DEFAULT_WEIGHTS_PATH, EXPECTED_SHA256, PretrainedSpeakerLoader
from app.streaming.service import StreamingSessionManager


class HealthProbes:
    """Production health, liveness, and multi-factor readiness probes."""

    @classmethod
    def check_liveness(cls) -> LivenessResponse:
        """Fast non-blocking liveness probe indicating process is up."""
        return LivenessResponse(
            status="LIVE",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def check_readiness(cls) -> ReadinessResponse:
        """Multi-factor readiness check assessing all critical subsystems and dependencies."""
        components: Dict[str, ComponentHealth] = {}
        all_ready = True
        degraded = False

        # 1. AASIST Model Checkpoint & Integrity
        aasist_valid, aasist_hash, _ = PretrainedModelLoader.verify_checkpoint_integrity()
        components["aasist_detector"] = ComponentHealth(
            name="AASIST Synthetic Voice Detector",
            status="READY" if aasist_valid else "FAILED",
            healthy=aasist_valid,
            details={"sha256_verified": aasist_valid, "hash": aasist_hash[:16] + "..."},
            message=None if aasist_valid else "AASIST checkpoint missing or SHA-256 mismatch.",
        )
        if not aasist_valid:
            if settings.STRICT_MODEL_INTEGRITY_CHECK:
                all_ready = False
            else:
                degraded = True

        # 2. ECAPA-TDNN Model Checkpoint & Integrity
        from pathlib import Path
        ecapa_path = Path(DEFAULT_WEIGHTS_PATH)
        ecapa_exists = ecapa_path.exists() and ecapa_path.stat().st_size > 0
        ecapa_hash = PretrainedSpeakerLoader.compute_sha256(ecapa_path) if ecapa_exists else "missing"
        ecapa_valid = ecapa_exists and (ecapa_hash.lower() == EXPECTED_SHA256.lower())

        components["ecapa_speaker_verifier"] = ComponentHealth(
            name="SpeechBrain ECAPA-TDNN Speaker Identity Verifier",
            status="READY" if ecapa_valid else "FAILED",
            healthy=ecapa_valid,
            details={"sha256_verified": ecapa_valid, "hash": ecapa_hash[:16] + "..."},
            message=None if ecapa_valid else "ECAPA-TDNN weights missing or corrupted.",
        )
        if not ecapa_valid:
            if settings.STRICT_MODEL_INTEGRITY_CHECK:
                all_ready = False
            else:
                degraded = True

        # 3. Streaming Manager & Session Capacity
        manager = StreamingSessionManager.get_instance()
        active_sessions = len(manager._sessions)
        capacity_ok = active_sessions < settings.MAX_ACTIVE_SESSIONS
        components["streaming_pipeline"] = ComponentHealth(
            name="Real-Time Streaming WebSocket Pipeline",
            status="READY" if capacity_ok else "AT_CAPACITY",
            healthy=capacity_ok,
            details={"active_sessions": active_sessions, "max_capacity": settings.MAX_ACTIVE_SESSIONS},
            message=None if capacity_ok else "Streaming session capacity saturated.",
        )
        if not capacity_ok:
            degraded = True

        # Determine overall readiness
        if not all_ready:
            overall_status = ReadinessStatus.NOT_READY
            msg = "Critical subsystem failed integrity or initialization checks."
        elif degraded:
            overall_status = ReadinessStatus.DEGRADED
            msg = "Application is operational with non-critical warnings or high load."
        else:
            overall_status = ReadinessStatus.READY
            msg = "All subsystems operational and ready to receive traffic."

        return ReadinessResponse(
            status=overall_status,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strict_integrity_mode=settings.STRICT_MODEL_INTEGRITY_CHECK,
            message=msg,
        )

    @classmethod
    def get_detailed_status(cls) -> OperationalStatusResponse:
        """Detailed operational status with safe disclosures and zero secret leakage."""
        metrics = MetricsRegistry()
        readiness = cls.check_readiness()
        manager = StreamingSessionManager.get_instance()

        return OperationalStatusResponse(
            application_status=readiness.status.value,
            environment=settings.ENVIRONMENT,
            uptime_seconds=metrics.uptime_seconds,
            active_detector="VoiceShield-AASIST-Pretrained-v1",
            speaker_model_available=readiness.components["ecapa_speaker_verifier"].healthy,
            streaming_available=readiness.components["streaming_pipeline"].healthy,
            active_sessions=len(manager._sessions),
            max_active_sessions=settings.MAX_ACTIVE_SESSIONS,
            model_integrity_status={
                "aasist": readiness.components["aasist_detector"].healthy,
                "ecapa": readiness.components["ecapa_speaker_verifier"].healthy,
            },
            evaluation_status_summary={
                "aasist": "PRETRAINED_NOT_YET_VALIDATED",
                "calibration": "NOT_CALIBRATED",
                "speaker_threshold": "PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED",
                "fusion": "FUSION_IMPLEMENTED_NOT_YET_VALIDATED",
                "streaming": "STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL",
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
