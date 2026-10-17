"""
REST API Endpoints for Speaker Identity Verification Subsystem.
"""
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from app.services.audio_service import AudioService
from app.audio.schemas import AudioData
from app.speaker_verification.registry import SpeakerVerificationService
from app.speaker_verification.provenance import get_speaker_model_provenance
from app.speaker_verification.schemas import (
    EnrollmentResponse,
    VerificationResponse,
    SpeakerProfileSummary,
    SpeakerModelProvenance,
)
from app.speaker_verification.evaluation.reports import load_latest_speaker_report
from app.speaker_verification.evaluation.evaluator import SpeakerVerificationEvaluator
from pathlib import Path

router = APIRouter(prefix="/speaker", tags=["Speaker Verification"])


class RunEvaluationRequest(BaseModel):
    manifest_path: str
    dataset_name: Optional[str] = "custom_speaker_benchmark"
    threshold: Optional[float] = 0.65


@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Enroll a speaker identity profile with one or more audio recordings."
)
async def enroll_speaker(
    profile_id: str = Form(..., description="Unique alphanumeric identifier for the speaker profile."),
    files: List[UploadFile] = File(..., description="One or more audio recordings (recommended 10-20s total).")
):
    """
    Decodes audio recordings into 16 kHz mono float32, extracts L2-normalized ECAPA-TDNN embeddings,
    computes an L2-normalized centroid, and stores the biometric profile in memory.
    Raw audio is immediately discarded.
    """
    if not profile_id or not profile_id.strip():
        raise HTTPException(status_code=400, detail="Profile ID must be non-empty.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file must be uploaded.")

    audio_samples: List[AudioData] = []
    for f in files:
        try:
            content = await f.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty.")
            audio_data = AudioService.get_processed_audio_data(content)
            audio_samples.append(audio_data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing '{f.filename}': {str(e)}")

    service = SpeakerVerificationService.get_instance()
    try:
        response = service.enrollment_manager.enroll(profile_id=profile_id.strip(), audio_samples=audio_samples)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failure: {str(e)}")


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify an audio recording against an enrolled speaker profile."
)
async def verify_speaker(
    profile_id: str = Form(..., description="Target enrolled speaker profile ID."),
    file: UploadFile = File(..., description="Verification audio recording.")
):
    """
    Extracts a 192-dim speaker embedding and computes cosine similarity against the enrolled profile centroid.
    Renders MATCH / NON_MATCH / UNCERTAIN decision with confidence band and privacy disclosures.
    """
    if not profile_id or not profile_id.strip():
        raise HTTPException(status_code=400, detail="Profile ID must be non-empty.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded verification file is empty.")
        audio_data = AudioService.get_processed_audio_data(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding verification audio: {str(e)}")

    service = SpeakerVerificationService.get_instance()
    try:
        response = service.verify(profile_id=profile_id.strip(), audio_data=audio_data)
        return response
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failure: {str(e)}")


@router.get(
    "/profiles",
    response_model=List[SpeakerProfileSummary],
    summary="List all enrolled speaker profile summaries (biometric embeddings omitted)."
)
async def list_profiles():
    """
    Returns sanitized profile metadata without exposing biometric embedding vectors.
    """
    service = SpeakerVerificationService.get_instance()
    return service.enrollment_manager.list_profiles()


@router.get(
    "/profiles/{profile_id}/status",
    response_model=SpeakerProfileSummary,
    summary="Get metadata and status for a specific enrolled profile."
)
async def get_profile_status(profile_id: str):
    """
    Returns sanitized metadata for an enrolled speaker profile.
    """
    service = SpeakerVerificationService.get_instance()
    profile = service.enrollment_manager.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")

    return SpeakerProfileSummary(
        profile_id=profile["profile_id"],
        model_id=profile["model_id"],
        embedding_dimension=profile["embedding_dimension"],
        sample_count=profile["sample_count"],
        total_audio_duration_seconds=profile["total_audio_duration_seconds"],
        created_at=profile["created_at"],
        updated_at=profile["updated_at"],
        in_memory_only=profile.get("in_memory_only", True),
    )


@router.delete(
    "/profiles/{profile_id}",
    summary="Delete an enrolled speaker profile."
)
async def delete_profile(profile_id: str):
    """
    Deletes an enrolled speaker profile from memory.
    """
    service = SpeakerVerificationService.get_instance()
    success = service.enrollment_manager.delete_profile(profile_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return {"success": True, "message": f"Profile '{profile_id}' deleted successfully."}


@router.get(
    "/provenance",
    response_model=SpeakerModelProvenance,
    summary="Get machine-readable cryptographic provenance for the speaker verification encoder."
)
async def get_provenance():
    """
    Returns cryptographic SHA-256 checksums, architecture, parameter count, and license for ECAPA-TDNN.
    """
    return get_speaker_model_provenance()


@router.get(
    "/models",
    summary="List available speaker verification models."
)
async def list_speaker_models():
    """
    Lists available speaker verification models.
    """
    provenance = get_speaker_model_provenance()
    return {
        "active_model": provenance.model_id,
        "models": [
            {
                "model_id": provenance.model_id,
                "model_name": provenance.model_name,
                "architecture": provenance.architecture,
                "embedding_dimension": provenance.embedding_dimension,
                "license": provenance.license,
                "scientific_status": provenance.scientific_status,
                "is_active": True,
            }
        ]
    }


@router.get(
    "/evaluation/status",
    summary="Get speaker verification benchmark and validation status."
)
async def get_speaker_evaluation_status():
    """
    Returns benchmark status and CSV trial manifest format guide.
    """
    provenance = get_speaker_model_provenance()
    latest_report = load_latest_speaker_report()

    return {
        "model_id": provenance.model_id,
        "scientific_status": provenance.scientific_status,
        "calibration_status": provenance.calibration_status,
        "has_latest_report": latest_report is not None,
        "latest_report_timestamp": latest_report.timestamp if latest_report else None,
        "dataset_manifest_instructions": {
            "format": "CSV or JSON Lines",
            "required_columns": [
                "trial_id",
                "enrollment_audio_path",
                "verification_audio_path",
                "enrollment_speaker_id",
                "verification_speaker_id",
                "trial_type"
            ],
            "optional_columns": ["generator", "language", "accent", "gender", "codec", "noise_condition"],
            "example_csv_row": "trial_001,/data/spk1_enr.wav,/data/spk1_ver.wav,spk_01,spk_01,target,elevenlabs,en,us,female,pcm,clean"
        }
    }


@router.post(
    "/evaluation/run",
    summary="Run speaker verification evaluation benchmark against a trial manifest."
)
async def run_speaker_evaluation(req: RunEvaluationRequest):
    """
    Executes speaker verification evaluation across target/non-target trials and detects speaker leakage.
    """
    path = Path(req.manifest_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest file '{req.manifest_path}' not found.")

    service = SpeakerVerificationService.get_instance()
    evaluator = SpeakerVerificationEvaluator(service.encoder)

    try:
        report = evaluator.run_evaluation(
            manifest_path=path,
            dataset_name=req.dataset_name or "custom_speaker_benchmark",
            threshold=req.threshold or 0.65,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
