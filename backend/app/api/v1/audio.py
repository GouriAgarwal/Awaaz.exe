"""Audio inspection and ingestion endpoints."""

from fastapi import APIRouter, File, UploadFile, status, HTTPException
from app.audio.schemas import AudioInspectResponse
from app.audio.validator import AudioValidationError
from app.services.audio_service import AudioService
from app.core.logging import get_logger

router = APIRouter(prefix="/audio", tags=["Audio Ingestion"])
logger = get_logger(__name__)


@router.post(
    "/inspect",
    response_model=AudioInspectResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect & Standardize Audio",
    description="Decodes, standardizes (16kHz mono float32), validates, and analyzes the signal quality of an uploaded audio file or microphone recording.",
)
async def inspect_audio(file: UploadFile = File(..., description="Audio file to inspect")) -> AudioInspectResponse:
    """Ingest and analyze audio file metadata, signal levels, and quality."""
    try:
        file_bytes = await file.read()
        return AudioService.inspect_audio(file_bytes=file_bytes, filename=file.filename)
    except AudioValidationError as e:
        logger.warning("Audio validation error for file %s: %s (code=%d)", file.filename, e.message, e.code)
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error processing audio file %s: %s", file.filename, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio payload.",
        )
