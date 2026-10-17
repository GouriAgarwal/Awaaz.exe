"""Feature extraction API endpoints."""

from fastapi import APIRouter, File, UploadFile, status, HTTPException
from app.audio.validator import AudioValidationError
from app.features.schemas import FeatureExtractionResponse
from app.services.feature_service import FeatureService
from app.core.logging import get_logger

router = APIRouter(prefix="/features", tags=["Acoustic Feature Extraction"])
logger = get_logger(__name__)


@router.post(
    "/extract",
    response_model=FeatureExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Acoustic, Spectral & Prosodic Features",
    description="Ingests an audio file, converts it to 16kHz mono float32, and extracts comprehensive time-domain, spectral, 20-MFCC, pitch, voice quality (jitter/shimmer/HNR), and prosodic dynamics.",
)
async def extract_features(
    file: UploadFile = File(..., description="Audio file for acoustic feature extraction"),
) -> FeatureExtractionResponse:
    """Extract structured acoustic and spectral features from audio payload."""
    try:
        file_bytes = await file.read()
        return FeatureService.extract_features(file_bytes=file_bytes, filename=file.filename)
    except AudioValidationError as e:
        logger.warning("Audio validation error during feature extraction for %s: %s", file.filename, e.message)
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error extracting features from %s: %s", file.filename, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract acoustic features.",
        )
