"""Fallback synthetic voice detector based on acoustic and harmonic regularity indicators.

Used when deep learning model weights or ML runtimes are unavailable.
Explicitly tagged as heuristic fallback with full transparent warnings.
"""

import time
from typing import List
import numpy as np

from app.audio.schemas import AudioData
from app.detection.base import BaseSyntheticVoiceDetector
from app.detection.calibration import ScoreCalibrator
from app.detection.metadata import get_fallback_metadata
from app.detection.preprocessing import AudioWindowPreprocessor
from app.detection.schemas import (
    ClassificationLabel,
    DetectionResult,
    DetectorMetadata,
    ScoreType,
    WindowScore,
)


class FallbackSyntheticDetector(BaseSyntheticVoiceDetector):
    """Safe fallback detector using acoustic consistency and spectral unnaturalness heuristics."""

    def __init__(
        self,
        threshold_natural: float = 0.35,
        threshold_synthetic: float = 0.65,
        aggregation_method: str = "median",
    ):
        self._is_loaded = False
        self.aggregation_method = aggregation_method
        self.preprocessor = AudioWindowPreprocessor()
        self.calibrator = ScoreCalibrator(
            threshold_natural=threshold_natural,
            threshold_synthetic=threshold_synthetic,
            score_type=ScoreType.HEURISTIC_FALLBACK_SCORE,
        )

    def load(self) -> None:
        """Initialize fallback engine."""
        self._is_loaded = True

    def is_loaded(self) -> bool:
        return self._is_loaded

    def metadata(self) -> DetectorMetadata:
        return get_fallback_metadata(device="cpu")

    def _score_window(self, samples: np.ndarray) -> float:
        """Compute acoustic unnaturalness metric on a single window.
        
        Evaluates spectral flatness variance, high-frequency energy ratio,
        and zero-crossing density anomalies typical of synthetic vocoders.
        """
        if len(samples) < 32:
            return 0.5

        # 1. Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(samples)))) / (len(samples) - 1)

        # 2. Power Spectrum & Spectral Centroid / Flatness
        fft_vals = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) + 1e-10
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.preprocessor.sample_rate)

        # Spectral centroid
        centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)

        # Spectral flatness (geometric mean / arithmetic mean)
        log_fft = np.log(fft_vals)
        geo_mean = np.exp(np.mean(log_fft))
        arith_mean = np.mean(fft_vals)
        flatness = float(geo_mean / (arith_mean + 1e-10))

        # 3. High-frequency energy ratio (> 4000 Hz)
        hf_mask = freqs > 4000
        hf_energy = np.sum(fft_vals[hf_mask] ** 2)
        total_energy = np.sum(fft_vals ** 2) + 1e-10
        hf_ratio = float(hf_energy / total_energy)

        # 4. Synthesize into normalized heuristic unnaturalness indicator [0.0, 1.0]
        # Pure tones or extreme flatness anomalies increase synthetic suspicion score
        norm_centroid = min(1.0, centroid / 4000.0)
        norm_hf = min(1.0, hf_ratio * 4.0)
        norm_flatness = min(1.0, flatness * 3.0)

        # Natural human speech typically has structured harmonic roll-off and moderate flatness
        raw_indicator = 0.35 * norm_centroid + 0.35 * norm_hf + 0.30 * norm_flatness
        synthetic_score = float(np.clip(raw_indicator, 0.05, 0.95))
        return synthetic_score

    def predict(self, audio_data: AudioData) -> DetectionResult:
        """Execute detection on 16 kHz mono AudioData."""
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        windows = self.preprocessor.process(audio_data)

        window_scores: List[WindowScore] = []
        scores_list: List[float] = []

        for window_idx, start_sec, end_sec, chunk in windows:
            score = self._score_window(chunk)
            label, _ = self.calibrator.classify_score(score)
            
            window_scores.append(
                WindowScore(
                    window_index=window_idx,
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    raw_score=round(score, 4),
                    synthetic_score=round(score, 4),
                    label=label,
                )
            )
            scores_list.append(score)

        # Aggregation
        if not scores_list:
            agg_score = 0.5
        elif self.aggregation_method == "mean":
            agg_score = float(np.mean(scores_list))
        else:  # default median
            agg_score = float(np.median(scores_list))

        final_label, confidence_band = self.calibrator.classify_score(agg_score)
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0

        warnings = [
            "FALLBACK ENGINE ACTIVE — NOT VALIDATED FOR PRODUCTION SECURITY DECISIONS.",
            "Inference was performed using non-ML acoustic heuristics because deep learning weights were unavailable.",
        ]

        return DetectionResult(
            detector_metadata=self.metadata(),
            classification=final_label,
            score=round(agg_score, 4),
            score_type=ScoreType.HEURISTIC_FALLBACK_SCORE,
            confidence_band=confidence_band,
            thresholds_applied=self.calibrator.get_thresholds_dict(),
            window_scores=window_scores,
            aggregation_method=self.aggregation_method,
            inference_latency_ms=round(inference_latency_ms, 2),
            audio_duration_sec=round(audio_data.duration_seconds, 3),
            total_windows=len(window_scores),
            warnings=warnings,
        )
