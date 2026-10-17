"""Operational telemetry and observability endpoints for Step 12."""

from typing import List, Optional
from fastapi import APIRouter, Query, status

from app.observability.schemas import (
    ApplicationMetricsResponse,
    ModelMonitoringInfo,
    StreamingOperationalInfo,
    OperationalErrorSummary,
)
from app.observability.service import observability_service

router = APIRouter(prefix="/observability", tags=["Observability & Telemetry"])


@router.get(
    "/metrics",
    response_model=ApplicationMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Operational Metrics",
    description="Returns aggregate real-time metrics including request counts, latency percentiles, error rates, model throughput, and streaming performance.",
)
async def get_metrics() -> ApplicationMetricsResponse:
    """Retrieve system-wide operational metrics."""
    return observability_service.get_application_metrics()


@router.get(
    "/models",
    response_model=List[ModelMonitoringInfo],
    status_code=status.HTTP_200_OK,
    summary="Model Operational Lifecycle",
    description="Returns lifecycle status, integrity verification, load count, inference count, and latency tracking for AASIST and ECAPA models.",
)
async def get_models() -> List[ModelMonitoringInfo]:
    """Retrieve operational lifecycle tracking for all AI models."""
    return observability_service.get_model_monitoring_info()


@router.get(
    "/streaming",
    response_model=StreamingOperationalInfo,
    status_code=status.HTTP_200_OK,
    summary="Streaming Telemetry & Capacity",
    description="Returns real-time session capacity, active connections, chunk/window metrics, audio processing lag, and security safeguards.",
)
async def get_streaming() -> StreamingOperationalInfo:
    """Retrieve real-time streaming operations telemetry."""
    return observability_service.get_streaming_operational_info()


@router.get(
    "/errors",
    response_model=OperationalErrorSummary,
    status_code=status.HTTP_200_OK,
    summary="Error Taxonomy & Recent Failures",
    description="Returns structured error taxonomy distributions, total failure counts, and recent error events with request correlation IDs.",
)
async def get_errors(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of recent errors to return"),
    error_code: Optional[str] = Query(default=None, description="Optional error code filter"),
) -> OperationalErrorSummary:
    """Retrieve operational error summary and recent failure records."""
    return observability_service.get_error_summary(limit=limit, error_code=error_code)
