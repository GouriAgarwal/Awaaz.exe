"""Pydantic Schemas for Step 12 Observability, Health, and Operational Readiness."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: str = "LIVE"
    timestamp: str


class ReadinessStatus(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    NOT_READY = "NOT_READY"


class ComponentHealth(BaseModel):
    name: str
    status: str
    healthy: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None


class ReadinessResponse(BaseModel):
    status: ReadinessStatus
    components: Dict[str, ComponentHealth]
    timestamp: str
    strict_integrity_mode: bool
    message: str


class OperationalStatusResponse(BaseModel):
    application_status: str
    environment: str
    uptime_seconds: float
    active_detector: str
    speaker_model_available: bool
    streaming_available: bool
    active_sessions: int
    max_active_sessions: int
    model_integrity_status: Dict[str, bool]
    evaluation_status_summary: Dict[str, str]
    timestamp: str


class HttpMetrics(BaseModel):
    total_requests: int = 0
    status_2xx: int = 0
    status_4xx: int = 0
    status_5xx: int = 0
    average_latency_ms: float = 0.0


class AasistMetrics(BaseModel):
    inference_count: int = 0
    inference_failure_count: int = 0
    average_latency_ms: float = 0.0
    model_available: bool = True
    integrity_verified: bool = True
    scientific_status: str = "PRETRAINED_NOT_YET_VALIDATED"


class EcapaMetrics(BaseModel):
    embedding_extraction_count: int = 0
    verification_count: int = 0
    failure_count: int = 0
    average_latency_ms: float = 0.0
    model_available: bool = True
    integrity_verified: bool = True
    operating_threshold_status: str = "PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED"


class FusionMetrics(BaseModel):
    fusion_count: int = 0
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    )
    action_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"ALLOW": 0, "MONITOR": 0, "STEP_UP_VERIFICATION": 0, "BLOCK_OR_ESCALATE": 0}
    )
    scientific_status: str = "FUSION_IMPLEMENTED_NOT_YET_VALIDATED"


class StreamingMetrics(BaseModel):
    active_sessions: int = 0
    total_sessions_created: int = 0
    total_sessions_closed: int = 0
    total_chunks_processed: int = 0
    total_windows_analyzed: int = 0
    average_window_latency_ms: float = 0.0
    max_processing_lag_ms: float = 0.0
    rate_limit_events: int = 0
    message_size_rejections: int = 0
    timeout_events: int = 0
    scientific_status: str = "STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL"


class ApplicationMetricsResponse(BaseModel):
    timestamp: str
    uptime_seconds: float
    http: HttpMetrics
    aasist: AasistMetrics
    ecapa: EcapaMetrics
    fusion: FusionMetrics
    streaming: StreamingMetrics
    session_capacity: Dict[str, int]


class ModelMonitoringInfo(BaseModel):
    model_id: str
    display_name: str
    architecture: str
    weights_path: str
    expected_sha256: str
    actual_sha256: str
    integrity_status: str
    checkpoint_verified: bool
    availability: str
    load_count: int = 1
    inference_count: int = 0
    inference_failure_count: int = 0
    average_latency_ms: float = 0.0
    scientific_status: str


class StreamingOperationalInfo(BaseModel):
    active_sessions: int
    max_active_sessions: int
    utilization_percentage: float
    total_chunks_ingested: int
    total_windows_analyzed: int
    rate_limit_count: int
    buffer_memory_cap_sec: float
    ephemeral_privacy_policy: str
    status: str


class OperationalErrorSummary(BaseModel):
    error_counts_by_code: Dict[str, int]
    total_errors_recorded: int
    timestamp: str


class OperationalErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: Optional[str] = None
    timestamp: str
