"""In-Memory Thread-Safe Application Metrics Registry (Step 12)."""

from datetime import datetime, timezone
import threading
import time
from typing import Dict, List, Optional

from app.core.config import settings
from app.observability.schemas import (
    AasistMetrics,
    ApplicationMetricsResponse,
    EcapaMetrics,
    FusionMetrics,
    HttpMetrics,
    StreamingMetrics,
)


class MetricsRegistry:
    """Thread-safe metrics aggregator for HTTP, models, fusion, and streaming pipelines."""

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self) -> None:
        self.start_time: float = time.time()

        # HTTP metrics
        self._http_requests_total: int = 0
        self._http_status_2xx: int = 0
        self._http_status_4xx: int = 0
        self._http_status_5xx: int = 0
        self._http_latencies_ms: List[float] = []

        # AASIST metrics
        self._aasist_inferences: int = 0
        self._aasist_failures: int = 0
        self._aasist_latencies_ms: List[float] = []

        # ECAPA metrics
        self._ecapa_extractions: int = 0
        self._ecapa_verifications: int = 0
        self._ecapa_failures: int = 0
        self._ecapa_latencies_ms: List[float] = []

        # Fusion metrics
        self._fusion_evaluations: int = 0
        self._fusion_risk_distribution: Dict[str, int] = {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0,
            "CRITICAL": 0,
        }
        self._fusion_action_distribution: Dict[str, int] = {
            "ALLOW": 0,
            "MONITOR": 0,
            "STEP_UP_VERIFICATION": 0,
            "BLOCK_OR_ESCALATE": 0,
        }

        # Streaming metrics
        self._streaming_sessions_created: int = 0
        self._streaming_sessions_closed: int = 0
        self._streaming_chunks_processed: int = 0
        self._streaming_windows_analyzed: int = 0
        self._streaming_window_latencies_ms: List[float] = []
        self._streaming_max_lag_ms: float = 0.0
        self._streaming_rate_limits: int = 0
        self._streaming_size_rejections: int = 0
        self._streaming_timeouts: int = 0

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 2)

    def record_http_request(self, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._http_requests_total += 1
            if 200 <= status_code < 300:
                self._http_status_2xx += 1
            elif 400 <= status_code < 500:
                self._http_status_4xx += 1
            elif status_code >= 500:
                self._http_status_5xx += 1

            self._http_latencies_ms.append(latency_ms)
            if len(self._http_latencies_ms) > 1000:
                self._http_latencies_ms = self._http_latencies_ms[-1000:]

    def record_aasist_inference(self, latency_ms: float, success: bool = True) -> None:
        with self._lock:
            self._aasist_inferences += 1
            if not success:
                self._aasist_failures += 1
            self._aasist_latencies_ms.append(latency_ms)
            if len(self._aasist_latencies_ms) > 500:
                self._aasist_latencies_ms = self._aasist_latencies_ms[-500:]

    def record_ecapa_verification(self, latency_ms: float, is_enrollment: bool = False, success: bool = True) -> None:
        with self._lock:
            if is_enrollment:
                self._ecapa_extractions += 1
            else:
                self._ecapa_verifications += 1
            if not success:
                self._ecapa_failures += 1
            self._ecapa_latencies_ms.append(latency_ms)
            if len(self._ecapa_latencies_ms) > 500:
                self._ecapa_latencies_ms = self._ecapa_latencies_ms[-500:]

    def record_fusion_evaluation(self, risk_level: str, action: str) -> None:
        with self._lock:
            self._fusion_evaluations += 1
            r_key = risk_level.upper()
            if r_key in self._fusion_risk_distribution:
                self._fusion_risk_distribution[r_key] += 1
            a_key = action.upper()
            if a_key in self._fusion_action_distribution:
                self._fusion_action_distribution[a_key] += 1

    def record_streaming_session_lifecycle(self, event_type: str) -> None:
        with self._lock:
            if event_type == "created":
                self._streaming_sessions_created += 1
            elif event_type == "closed":
                self._streaming_sessions_closed += 1
            elif event_type == "rate_limit":
                self._streaming_rate_limits += 1
            elif event_type == "size_rejection":
                self._streaming_size_rejections += 1
            elif event_type == "timeout":
                self._streaming_timeouts += 1

    def record_streaming_chunk(self) -> None:
        with self._lock:
            self._streaming_chunks_processed += 1

    def record_streaming_window(self, latency_ms: float, lag_ms: float) -> None:
        with self._lock:
            self._streaming_windows_analyzed += 1
            self._streaming_window_latencies_ms.append(latency_ms)
            if len(self._streaming_window_latencies_ms) > 500:
                self._streaming_window_latencies_ms = self._streaming_window_latencies_ms[-500:]
            if lag_ms > self._streaming_max_lag_ms:
                self._streaming_max_lag_ms = round(lag_ms, 2)

    def get_metrics_snapshot(self) -> ApplicationMetricsResponse:
        with self._lock:
            # Active sessions query
            from app.streaming.service import StreamingSessionManager
            manager = StreamingSessionManager.get_instance()
            active_sessions_count = len(manager._sessions)

            http_avg = (
                round(sum(self._http_latencies_ms) / len(self._http_latencies_ms), 2)
                if self._http_latencies_ms
                else 0.0
            )
            aasist_avg = (
                round(sum(self._aasist_latencies_ms) / len(self._aasist_latencies_ms), 2)
                if self._aasist_latencies_ms
                else 0.0
            )
            ecapa_avg = (
                round(sum(self._ecapa_latencies_ms) / len(self._ecapa_latencies_ms), 2)
                if self._ecapa_latencies_ms
                else 0.0
            )
            stream_avg = (
                round(sum(self._streaming_window_latencies_ms) / len(self._streaming_window_latencies_ms), 2)
                if self._streaming_window_latencies_ms
                else 0.0
            )

            from app.detection.pretrained.loader import PretrainedModelLoader
            aasist_valid, _, _ = PretrainedModelLoader.verify_checkpoint_integrity()
            from app.speaker_verification.model_loader import PretrainedSpeakerLoader, DEFAULT_WEIGHTS_PATH, EXPECTED_SHA256
            from pathlib import Path
            p = Path(DEFAULT_WEIGHTS_PATH)
            ecapa_valid = p.exists() and (PretrainedSpeakerLoader.compute_sha256(p).lower() == EXPECTED_SHA256.lower())

            return ApplicationMetricsResponse(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=self.uptime_seconds,
                http=HttpMetrics(
                    total_requests=self._http_requests_total,
                    status_2xx=self._http_status_2xx,
                    status_4xx=self._http_status_4xx,
                    status_5xx=self._http_status_5xx,
                    average_latency_ms=http_avg,
                ),
                aasist=AasistMetrics(
                    inference_count=self._aasist_inferences,
                    inference_failure_count=self._aasist_failures,
                    average_latency_ms=aasist_avg,
                    model_available=True,
                    integrity_verified=aasist_valid,
                    scientific_status="PRETRAINED_NOT_YET_VALIDATED",
                ),
                ecapa=EcapaMetrics(
                    embedding_extraction_count=self._ecapa_extractions,
                    verification_count=self._ecapa_verifications,
                    failure_count=self._ecapa_failures,
                    average_latency_ms=ecapa_avg,
                    model_available=True,
                    integrity_verified=ecapa_valid,
                    operating_threshold_status="PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED",
                ),
                fusion=FusionMetrics(
                    fusion_count=self._fusion_evaluations,
                    risk_distribution=dict(self._fusion_risk_distribution),
                    action_distribution=dict(self._fusion_action_distribution),
                    scientific_status="FUSION_IMPLEMENTED_NOT_YET_VALIDATED",
                ),
                streaming=StreamingMetrics(
                    active_sessions=active_sessions_count,
                    total_sessions_created=self._streaming_sessions_created,
                    total_sessions_closed=self._streaming_sessions_closed,
                    total_chunks_processed=self._streaming_chunks_processed,
                    total_windows_analyzed=self._streaming_windows_analyzed,
                    average_window_latency_ms=stream_avg,
                    max_processing_lag_ms=self._streaming_max_lag_ms,
                    rate_limit_events=self._streaming_rate_limits,
                    message_size_rejections=self._streaming_size_rejections,
                    timeout_events=self._streaming_timeouts,
                    scientific_status="STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL",
                ),
                session_capacity={
                    "current_active": active_sessions_count,
                    "max_capacity": settings.MAX_ACTIVE_SESSIONS,
                    "utilization_percent": round((active_sessions_count / settings.MAX_ACTIVE_SESSIONS) * 100, 1),
                },
            )


# Global singleton instance for app imports
metrics_registry = MetricsRegistry()

