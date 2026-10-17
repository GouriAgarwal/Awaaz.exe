"""Health check and readiness probe endpoints."""

from fastapi import APIRouter, Response, status
from app.models.schemas import HealthResponse
from app.services.system_service import SystemService
from app.observability.schemas import (
    LivenessResponse,
    ReadinessResponse,
    OperationalStatusResponse,
)
from app.observability.service import observability_service

router = APIRouter(tags=["Health & Probes"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check (Backward Compatible)",
    description="Returns the current operational status, service name, and version of VoiceShield API.",
)
async def get_health() -> HealthResponse:
    """Check the basic health status of the API service."""
    return SystemService.get_health()


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Fast, non-blocking check to verify the process is alive and responsive. Used by Kubernetes/Docker orchestrators.",
)
async def get_liveness() -> LivenessResponse:
    """Fast non-blocking liveness probe."""
    return observability_service.get_liveness()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Comprehensive readiness check verifying model availability, SHA-256 integrity, registry status, and session capacity.",
)
async def get_readiness(response: Response) -> ReadinessResponse:
    """Readiness probe evaluating system components. Returns 503 if NOT_READY, 200 if READY or DEGRADED."""
    result = observability_service.get_readiness()
    if result.status == "NOT_READY":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK
    return result


@router.get(
    "/health/status",
    response_model=OperationalStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed Operational Status",
    description="Comprehensive operational status report covering model states, session capacities, resource health, and system disclosures.",
)
async def get_operational_status() -> OperationalStatusResponse:
    """Returns detailed operational status of all subsystems."""
    return observability_service.get_operational_status()

