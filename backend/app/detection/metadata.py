"""Metadata definitions and provenance information for synthetic voice detectors."""

from app.core.config import settings
from app.detection.schemas import DetectorMetadata, ScoreType


def get_deep_learning_metadata(device: str = "cpu") -> DetectorMetadata:
    """Provenance metadata for Deep Learning / Raw Waveform / Spectral Model."""
    return DetectorMetadata(
        model_name="VoiceShield-AASIST-v1",
        model_type="deep_learning_raw_waveform",
        architecture="AASIST (Automated Anti-Spoofing Integration with Spectral-Temporal Graph Attention)",
        checkpoint_or_source="VoiceShield Pretrained Benchmark Checkpoint (ASVspoof2019/2021 logical access topology)",
        license="Apache 2.0 / Academic Evaluation",
        expected_sample_rate=settings.TARGET_SAMPLE_RATE,
        window_size_sec=settings.DETECTION_WINDOW_SECONDS,
        window_hop_sec=settings.DETECTION_HOP_SECONDS,
        score_type=ScoreType.UNCALIBRATED_MODEL_SCORE,
        score_interpretation="Continuous score in [0.0, 1.0]. Higher values indicate higher probability of synthetic/cloned speech artifacts.",
        scientific_disclaimer=(
            "Model score is an uncalibrated inference metric. Detection performance varies across codec compression, "
            "acoustic reverberation, background noise, and novel voice conversion architectures. Not certified for sole "
            "automated critical security enforcement without secondary corroboration."
        ),
        device=device,
        is_fallback=False,
    )


def get_spectral_cnn_metadata(device: str = "cpu") -> DetectorMetadata:
    """Provenance metadata for Spectral Convolutional synthetic voice detector."""
    return DetectorMetadata(
        model_name="VoiceShield-SpecCNN-v1",
        model_type="spectral_cnn",
        architecture="ResNet-18 Spectral Anti-Spoofing Architecture with Log-Mel Spectrogram Front-End",
        checkpoint_or_source="VoiceShield Pretrained Spectral Classifier",
        license="Apache 2.0",
        expected_sample_rate=settings.TARGET_SAMPLE_RATE,
        window_size_sec=settings.DETECTION_WINDOW_SECONDS,
        window_hop_sec=settings.DETECTION_HOP_SECONDS,
        score_type=ScoreType.UNCALIBRATED_MODEL_SCORE,
        score_interpretation="Continuous synthetic artifact likelihood in [0.0, 1.0].",
        scientific_disclaimer=(
            "Spectral artifact detector. Scores represent uncalibrated likelihood estimates on 16 kHz audio windows. "
            "Subject to false positives in high-frequency noise environments."
        ),
        device=device,
        is_fallback=False,
    )


def get_fallback_metadata(device: str = "cpu") -> DetectorMetadata:
    """Provenance metadata for Fallback Acoustic Consistency detector."""
    return DetectorMetadata(
        model_name="VoiceShield-AcousticConsistency-Fallback-v1",
        model_type="acoustic_heuristic_fallback",
        architecture="Multi-Feature Acoustic/Harmonic Regularity Estimator (Non-ML Fallback)",
        checkpoint_or_source="Builtin Step 3 Acoustic Consistency Engine",
        license="MIT (Internal VoiceShield Engine)",
        expected_sample_rate=settings.TARGET_SAMPLE_RATE,
        window_size_sec=settings.DETECTION_WINDOW_SECONDS,
        window_hop_sec=settings.DETECTION_HOP_SECONDS,
        score_type=ScoreType.HEURISTIC_FALLBACK_SCORE,
        score_interpretation=(
            "Heuristic score based on unnatural pitch modulation, spectral flatness variance, and harmonic distortion. "
            "Scores represent acoustic unnaturalness estimates, NOT a neural network inference."
        ),
        scientific_disclaimer=(
            "FALLBACK ENGINE — NOT VALIDATED FOR PRODUCTION SECURITY DECISIONS. "
            "Active because deep learning weights were not loaded. For research, diagnostics, and test verification only."
        ),
        device=device,
        is_fallback=True,
    )
