"""Provenance registries and audit records for all VoiceShield synthetic voice detectors."""

from typing import Dict, List
from app.detection.provenance_schema import (
    CalibrationStatus,
    ModelProvenanceRecord,
    PretrainedStatus,
)


def get_aasist_provenance() -> ModelProvenanceRecord:
    """Provenance record for VoiceShield-AASIST-v1."""
    return ModelProvenanceRecord(
        detector_name="VoiceShield-AASIST-v1",
        architecture="AASIST (Automated Anti-Spoofing Integration with SincConv Waveform Frontend)",
        checkpoint_identifier="none (weights/aasist_lite.pt not present)",
        source_repository="unknown",
        source_url="https://github.com/clovaai/aasist",
        revision="unknown",
        license="Apache 2.0",
        training_dataset="unknown",
        intended_task="Synthetic Voice Detection (Raw Waveform)",
        sample_rate=16000,
        input_format="16 kHz mono float32 PCM",
        output_classes=["NATURAL", "SYNTHETIC"],
        score_semantics="Uncalibrated sigmoid output in [0.0, 1.0]. Values near 0.5 indicate neutral baseline activation.",
        pretrained=False,
        validated=False,
        pretrained_status=PretrainedStatus.UNTRAINED_NEURAL_BASELINE,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        notes=(
            "AASIST-inspired raw waveform architecture. Checkpoint weights are not loaded; "
            "model currently executes with baseline random initialization. "
            "Formally audited and classified as UNTRAINED_NEURAL_BASELINE."
        ),
    )


def get_spec_cnn_provenance() -> ModelProvenanceRecord:
    """Provenance record for VoiceShield-SpecCNN-v1."""
    return ModelProvenanceRecord(
        detector_name="VoiceShield-SpecCNN-v1",
        architecture="ResNet-18 Spectral Anti-Spoofing Architecture with Log-Mel Spectrogram Front-End",
        checkpoint_identifier="none (weights/spectral_resnet.pt not present)",
        source_repository="unknown",
        source_url="unknown",
        revision="unknown",
        license="Apache 2.0",
        training_dataset="unknown",
        intended_task="Synthetic Voice Detection (Spectral)",
        sample_rate=16000,
        input_format="16 kHz mono float32 PCM",
        output_classes=["NATURAL", "SYNTHETIC"],
        score_semantics="Continuous spectral artifact score in [0.0, 1.0].",
        pretrained=False,
        validated=False,
        pretrained_status=PretrainedStatus.UNTRAINED_NEURAL_BASELINE,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        notes=(
            "Spectral ResNet model executing with baseline uncalibrated parameters. "
            "Classified as UNTRAINED_NEURAL_BASELINE."
        ),
    )


def get_fallback_provenance() -> ModelProvenanceRecord:
    """Provenance record for Fallback Acoustic Regularity Detector."""
    return ModelProvenanceRecord(
        detector_name="VoiceShield-AcousticConsistency-Fallback-v1",
        architecture="Multi-Feature Acoustic/Harmonic Regularity Estimator (Non-ML Heuristic)",
        checkpoint_identifier="builtin-deterministic-heuristic",
        source_repository="VoiceShield Builtin Feature Pipeline",
        source_url="unknown",
        revision="v1.0",
        license="MIT",
        training_dataset="none (deterministic acoustic formulas)",
        intended_task="Acoustic Unnaturalness Estimation (Diagnostics/Fallback)",
        sample_rate=16000,
        input_format="16 kHz mono float32 PCM",
        output_classes=["NATURAL", "SYNTHETIC"],
        score_semantics="Heuristic metric in [0.0, 1.0] derived from spectral flatness, HF energy, and centroid.",
        pretrained=False,
        validated=False,
        pretrained_status=PretrainedStatus.HEURISTIC_FALLBACK,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        notes=(
            "Heuristic fallback detector. Evaluates acoustic consistency and spectral unnaturalness. "
            "Explicitly classified as HEURISTIC_FALLBACK; not validated for production security enforcement."
        ),
    )


def get_all_provenance_records() -> Dict[str, ModelProvenanceRecord]:
    """Return dictionary of all registered detector provenance records."""
    from app.detection.pretrained.provenance import get_pretrained_aasist_provenance

    return {
        "aasist_pretrained": get_pretrained_aasist_provenance(),
        "aasist": get_aasist_provenance(),
        "aasist_untrained": get_aasist_provenance(),
        "spec_cnn": get_spec_cnn_provenance(),
        "fallback": get_fallback_provenance(),
    }
