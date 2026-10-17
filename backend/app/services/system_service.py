"""System service providing health and operational status information."""

from app.core.config import settings
from app.models.schemas import HealthResponse, SystemStatusResponse


class SystemService:
    """Service layer handling health and system status inspection."""

    @staticmethod
    def get_health() -> HealthResponse:
        """Return application health details."""
        return HealthResponse(
            status="ok",
            service=settings.PROJECT_NAME,
            version=settings.VERSION,
        )

    @staticmethod
    def get_system_status() -> SystemStatusResponse:
        """Return runtime system status details."""
        try:
            from app.detection.registry import detector_registry
            from app.detection.model_loader import ModelLoader
            active_det = detector_registry.get_active_detector_name()
            torch_avail = ModelLoader().is_torch_available
        except Exception:
            active_det = "fallback"
            torch_avail = False

        return SystemStatusResponse(
            backend="online",
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            detector_available=True,
            active_detector=active_det,
            torch_available=torch_avail,
        )
