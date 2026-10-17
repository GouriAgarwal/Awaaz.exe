"""Unit tests for synthetic voice detector components, preprocessors, calibrators, and registry."""

import numpy as np
import pytest

from app.audio.schemas import AudioData
from app.audio.validator import AudioSignalError
from app.detection.calibration import ScoreCalibrator
from app.detection.evaluation.metrics import compute_eer, evaluate_predictions
from app.detection.evaluation.schemas import EvaluationMetrics
from app.detection.fallback import FallbackSyntheticDetector
from app.detection.inference import DeepLearningSyntheticDetector
from app.detection.metadata import get_deep_learning_metadata, get_fallback_metadata
from app.detection.preprocessing import AudioWindowPreprocessor
from app.detection.registry import DetectorRegistry
from app.detection.schemas import ClassificationLabel, ScoreType


def create_mock_audio(duration_sec: float = 3.0, freq: float = 440.0, sr: int = 16000) -> AudioData:
    """Helper to generate mock AudioData."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioData(
        samples=samples,
        sample_rate=sr,
        channels=1,
        duration_seconds=duration_sec,
    )


class TestAudioWindowPreprocessor:
    """Tests for AudioWindowPreprocessor."""

    def test_short_audio_padding(self):
        """Audio shorter than window size (e.g. 1.0s vs 3.0s window) is padded to 1 full window."""
        preprocessor = AudioWindowPreprocessor(window_size_sec=3.0, window_hop_sec=1.5, sample_rate=16000)
        audio = create_mock_audio(duration_sec=1.0)
        windows = preprocessor.process(audio)

        assert len(windows) == 1
        w_idx, start, end, chunk = windows[0]
        assert w_idx == 0
        assert len(chunk) == 48000
        assert chunk.dtype == np.float32

    def test_exact_window_audio(self):
        """Audio exactly equal to window size produces 1 window."""
        preprocessor = AudioWindowPreprocessor(window_size_sec=3.0, window_hop_sec=1.5, sample_rate=16000)
        audio = create_mock_audio(duration_sec=3.0)
        windows = preprocessor.process(audio)

        assert len(windows) == 1
        assert len(windows[0][3]) == 48000

    def test_multi_window_audio(self):
        """Long audio (e.g. 6.0s) produces multiple overlapping windows."""
        preprocessor = AudioWindowPreprocessor(window_size_sec=3.0, window_hop_sec=1.5, sample_rate=16000)
        audio = create_mock_audio(duration_sec=6.0)
        windows = preprocessor.process(audio)

        # 0.0-3.0, 1.5-4.5, 3.0-6.0 -> 3 windows
        assert len(windows) == 3
        for w_idx, start, end, chunk in windows:
            assert len(chunk) == 48000
            assert start < end

    def test_nan_or_infinite_samples_raises_error(self):
        """Preprocessor rejects audio with NaN or infinite values."""
        preprocessor = AudioWindowPreprocessor()
        samples = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, channels=1, duration_seconds=3/16000)
        with pytest.raises(AudioSignalError):
            preprocessor.process(audio)

    def test_invalid_parameters_raise_value_error(self):
        """Invalid window parameters raise ValueError."""
        with pytest.raises(ValueError):
            AudioWindowPreprocessor(window_size_sec=-1.0)
        with pytest.raises(ValueError):
            AudioWindowPreprocessor(window_size_sec=2.0, window_hop_sec=3.0)


class TestScoreCalibrator:
    """Tests for ScoreCalibrator."""

    def test_natural_classification(self):
        calibrator = ScoreCalibrator(threshold_natural=0.35, threshold_synthetic=0.65)
        label, confidence = calibrator.classify_score(0.10)
        assert label == ClassificationLabel.NATURAL
        assert confidence == "HIGH"

        label, confidence = calibrator.classify_score(0.30)
        assert label == ClassificationLabel.NATURAL
        assert confidence == "MEDIUM"

    def test_synthetic_classification(self):
        calibrator = ScoreCalibrator(threshold_natural=0.35, threshold_synthetic=0.65)
        label, confidence = calibrator.classify_score(0.90)
        assert label == ClassificationLabel.SYNTHETIC
        assert confidence == "HIGH"

        label, confidence = calibrator.classify_score(0.70)
        assert label == ClassificationLabel.SYNTHETIC
        assert confidence == "MEDIUM"

    def test_uncertain_classification(self):
        calibrator = ScoreCalibrator(threshold_natural=0.35, threshold_synthetic=0.65)
        label, confidence = calibrator.classify_score(0.50)
        assert label == ClassificationLabel.UNCERTAIN
        assert confidence == "UNCERTAIN"

    def test_clamping_behavior(self):
        calibrator = ScoreCalibrator(threshold_natural=0.35, threshold_synthetic=0.65)
        label, _ = calibrator.classify_score(-0.5)
        assert label == ClassificationLabel.NATURAL

        label, _ = calibrator.classify_score(1.5)
        assert label == ClassificationLabel.SYNTHETIC


class TestFallbackSyntheticDetector:
    """Tests for FallbackSyntheticDetector."""

    def test_fallback_detector_prediction(self):
        detector = FallbackSyntheticDetector()
        audio = create_mock_audio(duration_sec=4.0)
        result = detector.predict(audio)

        assert result.detector_metadata.is_fallback is True
        assert result.score_type == ScoreType.HEURISTIC_FALLBACK_SCORE
        assert 0.0 <= result.score <= 1.0
        assert result.total_windows > 0
        assert len(result.window_scores) == result.total_windows
        assert len(result.warnings) > 0


class TestDeepLearningSyntheticDetector:
    """Tests for DeepLearningSyntheticDetector."""

    def test_deep_learning_detector_prediction(self):
        detector = DeepLearningSyntheticDetector(model_type="aasist")
        audio = create_mock_audio(duration_sec=5.0)
        result = detector.predict(audio)

        assert result.score_type == ScoreType.UNCALIBRATED_MODEL_SCORE
        assert 0.0 <= result.score <= 1.0
        assert result.inference_latency_ms >= 0.0
        assert result.total_windows >= 2
        assert result.classification in [
            ClassificationLabel.NATURAL,
            ClassificationLabel.SYNTHETIC,
            ClassificationLabel.UNCERTAIN,
        ]

    def test_metadata_completeness(self):
        detector = DeepLearningSyntheticDetector(model_type="spectral_cnn")
        meta = detector.metadata()
        assert meta.model_name == "VoiceShield-SpecCNN-v1"
        assert meta.expected_sample_rate == 16000
        assert len(meta.scientific_disclaimer) > 0


class TestDetectorRegistry:
    """Tests for DetectorRegistry."""

    def test_singleton_and_defaults(self):
        registry1 = DetectorRegistry()
        registry2 = DetectorRegistry()
        assert registry1 is registry2

        active_det = registry1.get_detector()
        assert active_det is not None

        models = registry1.list_detectors()
        assert len(models) >= 2

    def test_switching_active_detector(self):
        registry = DetectorRegistry()
        registry.set_active_detector("fallback")
        assert registry.get_active_detector_name() == "fallback"
        assert registry.get_detector().metadata().is_fallback is True

        # Reset back to aasist
        registry.set_active_detector("aasist")
        assert registry.get_active_detector_name() == "aasist"


class TestEvaluationMetrics:
    """Tests for evaluation metrics framework."""

    def test_compute_eer_perfect_separation(self):
        bonafide = np.array([0.1, 0.15, 0.2, 0.25])
        spoof = np.array([0.8, 0.85, 0.9, 0.95])
        eer, thresh = compute_eer(bonafide, spoof)
        assert eer == 0.0

    def test_evaluate_predictions_empty(self):
        metrics = evaluate_predictions(np.array([]), np.array([]))
        assert metrics.evaluation_status == "NOT_RUN"

    def test_evaluate_predictions_populated(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = evaluate_predictions(y_true, y_scores, dataset_name="UnitTestDataset")
        assert metrics.evaluation_status == "COMPLETED"
        assert metrics.equal_error_rate == 0.0
        assert metrics.accuracy == 1.0
        assert metrics.total_eval_samples == 6
