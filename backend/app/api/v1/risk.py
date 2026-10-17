"""
REST API Endpoints for VoiceShield Controlled Fusion & Impersonation Risk Engine.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from app.risk.service import ImpersonationRiskService
from app.risk.provenance import get_risk_provenance
from app.risk.schemas import (
    RiskAnalysisResult,
    RiskSimulationRequest,
    RiskConfigResponse,
    RiskProvenanceResponse,
)
from app.risk.evaluation.reports import load_latest_fusion_report
from app.risk.evaluation.evaluator import FusionEvaluator

router = APIRouter(prefix="/risk", tags=["Impersonation Risk Engine"])


class RunFusionEvaluationRequest(BaseModel):
    manifest_path: str
    dataset_name: Optional[str] = "custom_fusion_benchmark"


@router.post(
    "/analyze",
    response_model=RiskAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-modal voice cloning and impersonation risk analysis."
)
async def analyze_impersonation_risk(
    profile_id: str = Form(..., description="Claimed enrolled speaker profile ID."),
    file: UploadFile = File(..., description="Incoming audio stream or recording.")
):
    """
    Coordinates 16 kHz audio standardization, AASIST synthetic voice detection, ECAPA-TDNN speaker verification,
    and transparent rule-based fusion into an actionable 0–100 impersonation risk score with structured evidence.
    """
    if not profile_id or not profile_id.strip():
        raise HTTPException(status_code=400, detail="Profile ID must be non-empty.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio payload: {str(e)}")

    service = ImpersonationRiskService.get_instance()
    try:
        result = service.analyze_audio(file_bytes=content, profile_id=profile_id.strip())
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")


@router.post(
    "/simulate",
    response_model=RiskAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Simulate fusion rule matrix with manual synthetic and speaker scores."
)
async def simulate_risk_fusion(req: RiskSimulationRequest):
    """
    Developer/demo endpoint: feeds controlled synthetic detection and speaker verification signals
    directly into the rule matrix without executing neural network forward passes.
    """
    service = ImpersonationRiskService.get_instance()
    try:
        result = service.simulate(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation error: {str(e)}")


@router.get(
    "/config",
    response_model=RiskConfigResponse,
    summary="Get fusion engine configuration, risk bands, and action policy mappings."
)
async def get_risk_config():
    """
    Returns transparent configuration metadata for the rule matrix, score bands, and operational action mappings.
    """
    return RiskConfigResponse(
        fusion_version="v1.0-rule-matrix",
        ruleset_version="2026.09.1",
        calibration_status="NOT_CALIBRATED",
        risk_bands={
            "LOW": {"min": 0, "max": 24, "description": "Minimal risk of voice-cloning or identity fraud."},
            "MEDIUM": {"min": 25, "max": 49, "description": "Moderate risk or identity mismatch (natural non-enrolled voice)."},
            "HIGH": {"min": 50, "max": 74, "description": "Elevated risk of synthetic voice manipulation or spoofing."},
            "CRITICAL": {"min": 75, "max": 100, "description": "High likelihood of targeted voice-cloning attack on enrolled identity."},
        },
        speaker_similarity_bands={
            "VERY_LOW": {"range": "< 0.25", "meaning": "Acoustic identity mismatch"},
            "LOW": {"range": "0.25 - 0.55", "meaning": "Low similarity / distinct speaker"},
            "MEDIUM": {"range": "0.55 - 0.65", "meaning": "Borderline acoustic similarity"},
            "HIGH": {"range": ">= 0.65", "meaning": "Acoustic match with enrolled centroid"},
        },
        synthetic_score_bands={
            "LOW": {"range": "< 0.35", "meaning": "Natural speech dynamics"},
            "MEDIUM": {"range": "0.35 - 0.65", "meaning": "Borderline / moderate synthetic artifacts"},
            "HIGH": {"range": ">= 0.65", "meaning": "Strong synthetic voice / vocoder artifacts"},
        },
        action_policies={
            "ALLOW": "Acoustically verified genuine caller. Standard call flow permitted.",
            "MONITOR": "Passive surveillance recommended. Log biometric audit records.",
            "STEP_UP_VERIFICATION": "Secondary authentication required (SMS OTP, knowledge-based auth).",
            "BLOCK_OR_ESCALATE": "Immediate security intervention / agent alert for potential clone attack.",
        }
    )


@router.get(
    "/provenance",
    response_model=RiskProvenanceResponse,
    summary="Get cryptographic provenance for the multi-modal fusion architecture."
)
async def get_provenance():
    """
    Returns provenance for AASIST synthetic detector, ECAPA-TDNN speaker verifier, and the fusion ruleset.
    """
    return get_risk_provenance()


@router.get(
    "/evaluation/status",
    summary="Get multi-modal fusion evaluation and benchmark status."
)
async def get_fusion_evaluation_status():
    """
    Returns 4-quadrant benchmark readiness and trial manifest formatting instructions.
    """
    latest_report = load_latest_fusion_report()
    return {
        "fusion_engine": "VoiceShield-Controlled-Rule-Fusion",
        "scientific_status": "FUSION_IMPLEMENTED_NOT_YET_VALIDATED",
        "calibration_status": "NOT_CALIBRATED",
        "has_latest_report": latest_report is not None,
        "latest_report_timestamp": latest_report.timestamp if latest_report else None,
        "four_quadrant_framework": {
            "Quadrant 1": "Authorized + Natural (Legitimate caller)",
            "Quadrant 2": "Authorized + Synthetic (Targeted voice-cloning attack)",
            "Quadrant 3": "Unauthorized + Natural (Identity mismatch / normal third-party)",
            "Quadrant 4": "Unauthorized + Synthetic (Untargeted synthetic spoof)"
        },
        "dataset_manifest_instructions": {
            "format": "CSV or JSON Lines",
            "required_columns": [
                "trial_id",
                "audio_path",
                "profile_id",
                "speaker_id",
                "speaker_label",
                "synthetic_label",
                "attack_label"
            ],
            "optional_columns": ["generator", "language", "accent", "codec", "noise_condition"],
            "example_csv_row": "trial_001,/data/clone_01.wav,alice,spk_01,authorized,synthetic,impersonation,elevenlabs,en,us,pcm,clean"
        }
    }


@router.post(
    "/evaluation/run",
    summary="Run multi-modal 4-quadrant fusion benchmark against a trial manifest."
)
async def run_fusion_evaluation(req: RunFusionEvaluationRequest):
    """
    Executes 4-quadrant benchmark evaluation on multi-modal trial manifests.
    """
    path = Path(req.manifest_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest file '{req.manifest_path}' not found.")

    evaluator = FusionEvaluator()
    try:
        report = evaluator.run_evaluation(
            manifest_path=path,
            dataset_name=req.dataset_name or "custom_fusion_benchmark",
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion evaluation failed: {str(e)}")
