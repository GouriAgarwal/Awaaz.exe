"""Centralized deterministic audio preprocessing and quality analysis pipeline."""

from typing import Tuple, List
import numpy as np

from app.audio.decoder import AudioDecoder
from app.audio.normalizer import AudioNormalizer
from app.audio.schemas import AudioData, AudioQualityMetrics
from app.audio.validator import AudioValidator, AudioSignalError
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Production audio processing engine converting multi-format audio into 16kHz mono float32."""

    @classmethod
    def process(cls, file_bytes: bytes) -> Tuple[AudioData, AudioQualityMetrics]:
        """
        Execute the deterministic audio ingestion pipeline.

        Pipeline:
            1. Validate raw byte payload
            2. In-memory decoding & resampling (16kHz mono float32)
            3. Numeric sanity verification (NaN/Inf)
            4. Non-destructive amplitude normalization
            5. Duration & sample validation
            6. Signal quality & acoustic energy analysis

        Returns:
            Tuple of (AudioData, AudioQualityMetrics)
        """
        # 1. Validate raw bytes
        AudioValidator.validate_raw_bytes(file_bytes)

        # 2. Decode & resample to 16 kHz mono float32
        samples, original_format, original_sr, original_channels = AudioDecoder.decode(file_bytes)

        # 3. Verify numeric integrity
        if not np.all(np.isfinite(samples)):
            raise AudioSignalError("Decoded audio stream contains non-finite numerical values.")

        # 4. Safe amplitude normalization
        samples = AudioNormalizer.normalize(samples)

        # 5. Validate standardized samples
        AudioValidator.validate_samples(
            samples,
            sample_rate=settings.TARGET_SAMPLE_RATE,
            min_duration=settings.MIN_DURATION_SECONDS,
            max_duration=settings.MAX_DURATION_SECONDS,
        )

        # 6. Quality metrics & energy analysis
        quality_metrics = cls.analyze_quality(samples, settings.TARGET_SAMPLE_RATE)

        # 7. Construct internal AudioData object
        duration_seconds = round(len(samples) / settings.TARGET_SAMPLE_RATE, 3)
        audio_data = AudioData(
            samples=samples,
            sample_rate=settings.TARGET_SAMPLE_RATE,
            channels=settings.TARGET_CHANNELS,
            duration_seconds=duration_seconds,
            original_format=original_format,
            original_sample_rate=original_sr,
            original_channels=original_channels,
        )

        logger.info(
            "Audio processed: format=%s -> 16kHz mono, duration=%.2fs, rms=%.1fdB, quality=%s",
            original_format,
            duration_seconds,
            quality_metrics.rms_db,
            quality_metrics.quality,
        )

        return audio_data, quality_metrics

    @classmethod
    def analyze_quality(cls, samples: np.ndarray, sample_rate: int = 16000) -> AudioQualityMetrics:
        """Calculate signal energy, clipping, and silence ratios."""
        notes: List[str] = []

        # RMS Energy Calculation
        rms_val = float(np.sqrt(np.mean(samples**2) + 1e-12))
        rms_db = float(20.0 * np.log10(rms_val + 1e-9))
        rms_db = max(-100.0, min(0.0, rms_db))

        # Peak Amplitude
        peak_val = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        peak_db = float(20.0 * np.log10(peak_val + 1e-9))
        peak_db = max(-100.0, min(0.0, peak_db))

        # Clipping Ratio (|sample| >= CLIPPING_THRESHOLD)
        clipping_mask = np.abs(samples) >= settings.CLIPPING_THRESHOLD
        clipping_ratio = float(np.mean(clipping_mask)) if len(samples) > 0 else 0.0

        # Silence Ratio (Frame-based RMS analysis with 50ms frames)
        frame_size = int(0.05 * sample_rate)  # 800 samples = 50ms at 16kHz
        total_frames = max(1, len(samples) // frame_size)
        silent_frames = 0

        for i in range(total_frames):
            frame = samples[i * frame_size : (i + 1) * frame_size]
            frame_rms = np.sqrt(np.mean(frame**2) + 1e-12)
            frame_db = 20.0 * np.log10(frame_rms + 1e-9)
            if frame_db < settings.SILENCE_THRESHOLD_DB:
                silent_frames += 1

        silence_ratio = round(float(silent_frames / total_frames), 3)

        # Determine Quality Classification
        quality = "good"
        if rms_db < -60.0 or silence_ratio > 0.95:
            quality = "invalid"
            notes.append("Audio is near completely silent or below detectable signal floor.")
        elif rms_db < -35.0:
            quality = "warning"
            notes.append("Low signal energy; speech features may be quiet.")
        elif clipping_ratio > 0.02:
            quality = "warning"
            notes.append(f"Moderate signal clipping detected ({clipping_ratio*100:.1f}%).")
        elif silence_ratio > 0.70:
            quality = "warning"
            notes.append("Audio contains significant periods of silence.")

        if not notes:
            notes.append("Acoustic signal levels and dynamic range are optimal.")

        return AudioQualityMetrics(
            rms_db=round(rms_db, 1),
            peak_db=round(peak_db, 1),
            clipping_ratio=round(clipping_ratio, 4),
            silence_ratio=silence_ratio,
            quality=quality,
            notes=notes,
        )
