"""FastAPI endpoints for AI Synthetic / Cloned Voice Detection (Step 4)."""

from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.audio.validator import AudioValidationError
from app.core.config import settings
from app.core.logging import get_logger
from app.detection.schemas import (
    DetectionResponse,
    DetectionResult,
    DetectorMetadata,
)
from app.services.detection_service import detection_service

logger = get_logger("api.detection")

router = APIRouter(prefix="/detection", tags=["AI Voice Detection"])


@router.post(
    "/analyze",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze audio for AI synthetic/cloned voice evidence",
    description=(
        "Processes an audio recording through the 16 kHz standardized pipeline and runs "
        "AI-based synthetic voice detection. Returns classification, score, score type, "
        "confidence band, and temporal window breakdown."
    ),
)
async def analyze_synthetic_voice(
    file: UploadFile = File(..., description="Audio file to analyze (WAV, MP3, WebM, M4A, OGG, FLAC)"),
    detector: Optional[str] = Form(None, description="Optional detector backend (aasist, aasist_pretrained, spec_cnn, fallback)"),
    model_name: Optional[str] = None,
) -> DetectionResponse:
    """Analyze audio for synthetic speech artifacts."""
    target_detector = detector or model_name
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read uploaded audio file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty (0 bytes).",
        )

    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({len(file_bytes)} bytes) exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB.",
        )

    try:
        detection_result, _ = detection_service.analyze_raw_audio(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
            detector_name=target_detector,
        )

        return DetectionResponse(
            success=True,
            data=detection_result,
            message="Synthetic voice detection completed successfully.",
        )

    except AudioValidationError as exc:
        logger.warning(f"Audio validation error during detection: {exc.message}")
        raise HTTPException(
            status_code=exc.code,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error(f"Unexpected error in synthetic voice detection: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the audio for voice clone detection: {str(exc)}",
        )


@router.get(
    "/models",
    response_model=List[DetectorMetadata],
    summary="List available synthetic voice detection models and their metadata",
)
@router.get(
    "/detectors",
    response_model=List[DetectorMetadata],
    summary="Alias to list available synthetic voice detectors",
    include_in_schema=False,
)
async def list_models() -> List[DetectorMetadata]:
    """Retrieve metadata of all registered synthetic voice detectors."""
    return detection_service.list_available_detectors()


@router.get(
    "/active",
    response_model=DetectorMetadata,
    summary="Get active synthetic voice detector metadata",
)
async def get_active_model() -> DetectorMetadata:
    """Retrieve metadata of the currently active synthetic voice detector."""
    return detection_service.get_active_metadata()


# ---------------------------------------------------------------------------
# Step 5: Scientific Validation and Evaluation Endpoints
# ---------------------------------------------------------------------------

from app.detection.evaluation.reports import get_latest_evaluation_report, save_evaluation_report
from app.detection.evaluation.runner import EvaluationRunner
from app.detection.evaluation.dataset import DatasetManifestParser, DatasetValidationError
from app.detection.evaluation.schemas import EvaluationReport
from app.detection.provenance import get_all_provenance_records
from app.detection.provenance_schema import ModelProvenanceRecord
from app.detection.registry import detector_registry


@router.get(
    "/evaluation/status",
    response_model=EvaluationReport,
    summary="Get latest scientific evaluation report and benchmark status",
    description="Returns the latest evaluation report, metrics, latency benchmarks, or 'NOT_RUN' status.",
)
@router.get(
    "/evaluation/latest",
    response_model=EvaluationReport,
    summary="Retrieve the latest completed evaluation report",
    include_in_schema=True,
)
async def get_evaluation_status() -> EvaluationReport:
    """Retrieve latest evaluation report."""
    return get_latest_evaluation_report()


@router.get(
    "/provenance",
    response_model=dict[str, ModelProvenanceRecord],
    summary="Get machine-readable model provenance records for all detectors",
    description="Returns provenance, checkpoint, architecture, and pretrained status for all detectors.",
)
async def get_model_provenance() -> dict[str, ModelProvenanceRecord]:
    """Retrieve machine-readable model provenance records."""
    return get_all_provenance_records()


@router.post(
    "/evaluation/run",
    response_model=EvaluationReport,
    summary="Run benchmark evaluation against a dataset manifest",
    description="Executes reproducible benchmark evaluation on a specified manifest and saves the report.",
)
async def run_evaluation(
    manifest_path: str = Form(..., description="Path to dataset manifest CSV or JSON file"),
    detector: Optional[str] = Form(None, description="Optional detector backend (aasist, spec_cnn, fallback)"),
    threshold: float = Form(0.50, description="Operating threshold for classification"),
) -> EvaluationReport:
    """Run benchmark evaluation on a dataset manifest."""
    try:
        manifest = DatasetManifestParser.parse_manifest_file(manifest_path)
    except DatasetValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset manifest validation failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not load manifest: {str(e)}",
        )

    target_detector = detector_registry.get_detector(detector)
    report = EvaluationRunner.evaluate(
        detector=target_detector,
        manifest=manifest,
        operating_threshold=threshold,
    )
    save_evaluation_report(report)
    return report
