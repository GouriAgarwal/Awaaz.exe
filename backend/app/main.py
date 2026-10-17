"""VoiceShield FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.errors import setup_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.observability.tracing import TracingMiddleware
from app.streaming.service import StreamingSessionManager

# Initialize structured logging
setup_logging(settings.LOG_LEVEL)
logger = get_logger("voiceshield.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and graceful shutdown events."""
    logger.info(
        "Starting %s v%s in %s mode (Host: %s:%s)...",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
        settings.HOST,
        settings.PORT,
    )
    logger.info("Configured CORS origins: %s", settings.CORS_ORIGINS)

    # Step 11/12 Security: Model Integrity Verification on Startup
    try:
        from app.evaluation.security_audit import SecurityAuditor
        model_audits = SecurityAuditor.audit_model_integrity()
        for audit in model_audits:
            if audit.integrity_verified:
                logger.info("Verified %s (SHA-256: %s)", audit.model_name, audit.actual_sha256[:16] + "...")
            else:
                logger.warning("Integrity check warning for %s: %s", audit.model_name, audit.status)
    except Exception as e:
        logger.error("Error executing startup model integrity audit: %s", str(e))

    yield

    # Graceful Shutdown Sequence
    logger.info("Initiating graceful shutdown for %s...", settings.PROJECT_NAME)
    try:
        # Cleanup active streaming sessions without leaking memory
        manager = StreamingSessionManager.get_instance()
        active_count = len(manager._sessions)
        if active_count > 0:
            logger.info("Cleaning up %d active streaming session(s)...", active_count)
            manager.cleanup_stale_sessions()
    except Exception as e:
        logger.error("Error during streaming session shutdown cleanup: %s", str(e))

    logger.info("Shutdown complete for %s.", settings.PROJECT_NAME)



def create_application() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Step 12 Observability Middleware (Correlation IDs & HTTP metrics)
    application.add_middleware(TracingMiddleware)

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handling
    setup_exception_handlers(application)

    # API routers
    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @application.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        """Redirect root path to interactive Swagger documentation."""
        return RedirectResponse(url="/docs")

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

