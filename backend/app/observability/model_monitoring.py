"""Neural Model Lifecycle and Operational Tracking (Step 12)."""

from pathlib import Path
from typing import List

from app.core.config import settings
from app.detection.pretrained.loader import (
    CHECKPOINT_PATH as AASIST_PATH,
    EXPECTED_SHA256 as AASIST_SHA256,
    PretrainedModelLoader,
)
from app.observability.metrics import MetricsRegistry
from app.observability.schemas import ModelMonitoringInfo
from app.speaker_verification.model_loader import (
    DEFAULT_WEIGHTS_PATH as ECAPA_PATH,
    EXPECTED_SHA256 as ECAPA_SHA256,
    PretrainedSpeakerLoader,
)


class ModelLifecycleMonitor:
    """Monitors model health, load events, latency, and scientific disclosures."""

    @classmethod
    def get_models_status(cls) -> List[ModelMonitoringInfo]:
        metrics = MetricsRegistry()
        models: List[ModelMonitoringInfo] = []

        # 1. AASIST Status
        aasist_valid, aasist_actual_hash, _ = PretrainedModelLoader.verify_checkpoint_integrity(AASIST_PATH)
        aasist_avg_latency = (
            round(sum(metrics._aasist_latencies_ms) / len(metrics._aasist_latencies_ms), 2)
            if metrics._aasist_latencies_ms
            else 0.0
        )
        models.append(
            ModelMonitoringInfo(
                model_id="VoiceShield-AASIST-Pretrained-v1",
                display_name="AASIST Graph Attention Synthetic Voice Detector",
                architecture="Integrated Spectro-Temporal Graph Attention Network",
                weights_path=str(AASIST_PATH),
                expected_sha256=AASIST_SHA256,
                actual_sha256=aasist_actual_hash,
                integrity_status="VALIDATED" if aasist_valid else "CORRUPTED_OR_MISSING",
                checkpoint_verified=aasist_valid,
                availability="AVAILABLE" if aasist_valid else "UNAVAILABLE",
                load_count=1,
                inference_count=metrics._aasist_inferences,
                inference_failure_count=metrics._aasist_failures,
                average_latency_ms=aasist_avg_latency,
                scientific_status="PRETRAINED_NOT_YET_VALIDATED",
            )
        )

        # 2. ECAPA-TDNN Status
        ecapa_path = Path(ECAPA_PATH)
        if ecapa_path.exists() and ecapa_path.stat().st_size > 0:
            ecapa_actual_hash = PretrainedSpeakerLoader.compute_sha256(ecapa_path)
            ecapa_valid = (ecapa_actual_hash.lower() == ECAPA_SHA256.lower())
        else:
            ecapa_actual_hash = "missing"
            ecapa_valid = False

        ecapa_avg_latency = (
            round(sum(metrics._ecapa_latencies_ms) / len(metrics._ecapa_latencies_ms), 2)
            if metrics._ecapa_latencies_ms
            else 0.0
        )
        models.append(
            ModelMonitoringInfo(
                model_id="SpeechBrain-ECAPA-TDNN-v1",
                display_name="SpeechBrain ECAPA-TDNN Speaker Identity Verifier",
                architecture="Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network",
                weights_path=str(ecapa_path),
                expected_sha256=ECAPA_SHA256,
                actual_sha256=ecapa_actual_hash,
                integrity_status="VALIDATED" if ecapa_valid else "CORRUPTED_OR_MISSING",
                checkpoint_verified=ecapa_valid,
                availability="AVAILABLE" if ecapa_valid else "UNAVAILABLE",
                load_count=1,
                inference_count=metrics._ecapa_verifications + metrics._ecapa_extractions,
                inference_failure_count=metrics._ecapa_failures,
                average_latency_ms=ecapa_avg_latency,
                scientific_status="PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED",
            )
        )

        return models
