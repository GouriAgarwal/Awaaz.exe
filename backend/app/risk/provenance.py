from app.risk.schemas import RiskProvenanceResponse
from app.detection.pretrained.provenance import get_pretrained_aasist_provenance
from app.speaker_verification.provenance import get_speaker_model_provenance


def get_risk_provenance() -> RiskProvenanceResponse:
    """
    Returns combined machine-readable provenance metadata for the fusion engine.
    """
    aasist_prov = get_pretrained_aasist_provenance()
    spk_prov = get_speaker_model_provenance()

    return RiskProvenanceResponse(
        fusion_engine="VoiceShield-Controlled-Rule-Fusion",
        fusion_version="v1.0",
        calibration_status="NOT_CALIBRATED",
        ruleset_version="2026.09.1",
        synthetic_detector={
            "model_name": aasist_prov.detector_name if aasist_prov else "VoiceShield-AASIST-Pretrained-v1",
            "architecture": aasist_prov.architecture if aasist_prov else "AASIST",
            "sha256": "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0",
            "license": "BSD-3-Clause",
            "scientific_status": "PRETRAINED_NOT_YET_VALIDATED",
        },
        speaker_verifier={
            "model_name": spk_prov.model_name,
            "architecture": spk_prov.architecture,
            "sha256": spk_prov.checkpoint_sha256,
            "license": spk_prov.license,
            "embedding_dimension": spk_prov.embedding_dimension,
            "scientific_status": spk_prov.scientific_status,
        },
        disclaimer=(
            "VoiceShield Impersonation Risk Engine computes a provisional heuristic fusion score. "
            "It combines orthogonal synthetic detection and biometric similarity signals. "
            "It is not a calibrated probability of fraud."
        )
    )
