"""System status endpoints."""

from fastapi import APIRouter, status
from app.models.schemas import SystemStatusResponse
from app.services.system_service import SystemService

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="System Status",
    description="Returns backend runtime state, version, and environment configuration.",
)
async def get_system_status() -> SystemStatusResponse:
    """Retrieve detailed backend system status."""
    return SystemService.get_system_status()
