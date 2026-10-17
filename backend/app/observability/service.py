"""Observability and Operational Management Service (Step 12)."""

from datetime import datetime, timezone
from typing import List, Optional

from app.observability.errors import ErrorTracker
from app.observability.health import HealthProbes
from app.observability.metrics import MetricsRegistry
from app.observability.model_monitoring import ModelLifecycleMonitor
from app.observability.schemas import (
    ApplicationMetricsResponse,
    LivenessResponse,
    ModelMonitoringInfo,
    OperationalErrorSummary,
    OperationalStatusResponse,
    ReadinessResponse,
    StreamingOperationalInfo,
)
from app.observability.streaming_monitoring import StreamingOperationalMonitor


class ObservabilityService:
    """Central service coordinating health checks, metrics, model tracking, and error taxonomy."""

    _instance: Optional["ObservabilityService"] = None

    def __new__(cls) -> "ObservabilityService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_liveness(self) -> LivenessResponse:
        return HealthProbes.check_liveness()

    def get_readiness(self) -> ReadinessResponse:
        return HealthProbes.check_readiness()

    def get_operational_status(self) -> OperationalStatusResponse:
        return HealthProbes.get_detailed_status()

    def get_metrics(self) -> ApplicationMetricsResponse:
        registry = MetricsRegistry()
        return registry.get_metrics_snapshot()

    def get_application_metrics(self) -> ApplicationMetricsResponse:
        return self.get_metrics()

    def get_models_status(self) -> List[ModelMonitoringInfo]:
        return ModelLifecycleMonitor.get_models_status()

    def get_model_monitoring_info(self) -> List[ModelMonitoringInfo]:
        return self.get_models_status()

    def get_streaming_status(self) -> StreamingOperationalInfo:
        return StreamingOperationalMonitor.get_streaming_telemetry()

    def get_streaming_operational_info(self) -> StreamingOperationalInfo:
        return self.get_streaming_status()


    def get_error_summary(self, limit: int = 20, error_code: Optional[str] = None) -> OperationalErrorSummary:
        counts = ErrorTracker.get_error_counts()
        total = ErrorTracker.get_total_errors()
        return OperationalErrorSummary(
            error_counts_by_code=counts,
            total_errors_recorded=total,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# Global singleton instance for app imports
observability_service = ObservabilityService()

