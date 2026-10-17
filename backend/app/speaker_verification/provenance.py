"""
Cryptographic Provenance for Speaker Identity Verification Subsystem.
"""
from app.speaker_verification.schemas import SpeakerModelProvenance, CalibrationStatus
from app.speaker_verification.model_loader import EXPECTED_SHA256, HF_REPO_ID, HF_FILENAME


def get_speaker_model_provenance() -> SpeakerModelProvenance:
    """
    Returns machine-readable provenance metadata for the active speaker verification encoder.
    """
    return SpeakerModelProvenance(
        model_id="speechbrain_ecapa_tdnn_voxceleb",
        model_name="SpeechBrain ECAPA-TDNN (VoxCeleb)",
        version="1.0.0",
        architecture="ECAPA-TDNN (Conv1D + SE-Res2Net + MFA + Attentive Statistics Pooling)",
        source_repository=f"https://huggingface.co/{HF_REPO_ID}",
        checkpoint_url=f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{HF_FILENAME}",
        checkpoint_sha256=EXPECTED_SHA256,
        license="Apache-2.0",
        embedding_dimension=192,
        expected_sample_rate=16000,
        parameter_count=20767552,
        training_dataset="VoxCeleb 1 + VoxCeleb 2 (7,205 speakers)",
        is_l2_normalized=True,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        scientific_status="SPEAKER_MODEL_INTEGRATED_NOT_YET_VALIDATED",
        disclaimer=(
            "Speaker similarity scores evaluate acoustic-phonetic identity, not synthetic manipulation. "
            "High similarity indicates acoustic closeness to enrollment, but does NOT rule out AI-generated cloning."
        )
    )
