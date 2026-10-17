"""Main API router combining all endpoint modules."""

from fastapi import APIRouter
from app.api.v1 import (
    health,
    system,
    audio,
    features,
    detection,
    speaker,
    risk,
    streaming,
    evaluation,
    observability,
    contextual_risk,
    prevention,
)

api_router = APIRouter()

# Include endpoint sub-routers
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(audio.router)
api_router.include_router(features.router)
api_router.include_router(detection.router)
api_router.include_router(speaker.router)
api_router.include_router(risk.router)
api_router.include_router(streaming.router)
api_router.include_router(evaluation.router)
api_router.include_router(observability.router)
api_router.include_router(contextual_risk.router)
api_router.include_router(prevention.router)



