"""Fundamental Frequency (F0) tracking and speech prosody dynamics extraction."""

from typing import List, Tuple, Optional
import numpy as np
from app.core.config import settings
from app.features.schemas import PitchFeatures, ProsodyFeatures
from app.features.validation import safe_float


class ProsodyFeatureExtractor:
    """Extracts fundamental pitch (F0), voiced intervals, speaking rate, and prosody dynamics."""

    @classmethod
    def estimate_f0_frames(cls, samples: np.ndarray) -> Tuple[List[Optional[float]], List[bool]]:
        """
        Estimate frame-by-frame fundamental frequency using autocorrelation with parabolic interpolation.

        Returns:
            Tuple of (f0_values_per_frame, is_voiced_per_frame)
        """
        sample_rate = settings.TARGET_SAMPLE_RATE
        win_size = settings.STFT_WIN_LENGTH  # 400 samples (25ms)
        hop_size = settings.STFT_HOP_LENGTH  # 160 samples (10ms)

        lag_min = int(sample_rate / settings.PITCH_F0_MAX)  # ~32 samples (500 Hz)
        lag_max = int(sample_rate / settings.PITCH_F0_MIN)  # ~320 samples (50 Hz)

        total_frames = max(1, (len(samples) - win_size) // hop_size + 1)
        f0_list: List[Optional[float]] = []
        voiced_list: List[bool] = []

        for i in range(total_frames):
            frame = samples[i * hop_size : i * hop_size + win_size]
            if len(frame) < win_size:
                frame = np.pad(frame, (0, win_size - len(frame)))

            frame_rms = float(np.sqrt(np.mean(frame**2)))
            if frame_rms < 0.003:  # Near-silent frame
                f0_list.append(None)
                voiced_list.append(False)
                continue

            # Normalized Autocorrelation
            frame_centered = frame - np.mean(frame)
            autocorr = np.correlate(frame_centered, frame_centered, mode="full")
            center = len(autocorr) // 2
            r = autocorr[center:]

            r0 = r[0] + 1e-12
            if lag_max >= len(r):
                lag_max_bound = len(r) - 1
            else:
                lag_max_bound = lag_max

            if lag_min >= lag_max_bound:
                f0_list.append(None)
                voiced_list.append(False)
                continue

            search_region = r[lag_min : lag_max_bound + 1]
            peak_idx = int(np.argmax(search_region)) + lag_min
            peak_val = r[peak_idx]

            norm_peak = float(peak_val / r0)

            if norm_peak >= settings.VOICING_THRESHOLD:
                # Parabolic Interpolation for Sub-sample Peak Refinement
                if 0 < peak_idx < len(r) - 1:
                    alpha = r[peak_idx - 1]
                    beta = r[peak_idx]
                    gamma = r[peak_idx + 1]
                    denom = alpha - 2.0 * beta + gamma
                    if abs(denom) > 1e-12:
                        delta = 0.5 * (alpha - gamma) / denom
                        true_lag = peak_idx + delta
                    else:
                        true_lag = float(peak_idx)
                else:
                    true_lag = float(peak_idx)

                if true_lag > 0:
                    f0 = sample_rate / true_lag
                    if settings.PITCH_F0_MIN <= f0 <= settings.PITCH_F0_MAX:
                        f0_list.append(f0)
                        voiced_list.append(True)
                        continue

            f0_list.append(None)
            voiced_list.append(False)

        return f0_list, voiced_list

    @classmethod
    def extract_pitch_features(cls, samples: np.ndarray) -> PitchFeatures:
        """Extract voiced F0 statistics and voicing ratio."""
        if len(samples) == 0:
            return PitchFeatures(
                f0_mean_hz=None,
                f0_median_hz=None,
                f0_std_hz=None,
                f0_min_hz=None,
                f0_max_hz=None,
                f0_p10_hz=None,
                f0_p25_hz=None,
                f0_p75_hz=None,
                f0_p90_hz=None,
                voiced_ratio=0.0,
            )

        f0_list, voiced_list = cls.estimate_f0_frames(samples)
        voiced_f0 = [f for f in f0_list if f is not None]

        total_frames = len(voiced_list)
        voiced_ratio = float(len(voiced_f0) / total_frames) if total_frames > 0 else 0.0

        if not voiced_f0:
            return PitchFeatures(
                f0_mean_hz=None,
                f0_median_hz=None,
                f0_std_hz=None,
                f0_min_hz=None,
                f0_max_hz=None,
                f0_p10_hz=None,
                f0_p25_hz=None,
                f0_p75_hz=None,
                f0_p90_hz=None,
                voiced_ratio=safe_float(voiced_ratio, 0.0, 3),
            )

        f0_arr = np.array(voiced_f0, dtype=np.float64)

        return PitchFeatures(
            f0_mean_hz=safe_float(np.mean(f0_arr), None, 1),
            f0_median_hz=safe_float(np.median(f0_arr), None, 1),
            f0_std_hz=safe_float(np.std(f0_arr), None, 1),
            f0_min_hz=safe_float(np.min(f0_arr), None, 1),
            f0_max_hz=safe_float(np.max(f0_arr), None, 1),
            f0_p10_hz=safe_float(np.percentile(f0_arr, 10), None, 1),
            f0_p25_hz=safe_float(np.percentile(f0_arr, 25), None, 1),
            f0_p75_hz=safe_float(np.percentile(f0_arr, 75), None, 1),
            f0_p90_hz=safe_float(np.percentile(f0_arr, 90), None, 1),
            voiced_ratio=safe_float(voiced_ratio, 0.0, 3),
        )

    @classmethod
    def extract_prosody_features(cls, samples: np.ndarray) -> ProsodyFeatures:
        """Extract temporal dynamics, pause structure, and speaking activity."""
        if len(samples) == 0:
            return ProsodyFeatures(
                voiced_ratio=0.0,
                pause_ratio=1.0,
                speaking_ratio=0.0,
                voiced_segment_count=0,
                avg_voiced_duration_s=0.0,
                energy_variability=0.0,
                f0_variability=None,
            )

        f0_list, voiced_list = cls.estimate_f0_frames(samples)
        sample_rate = settings.TARGET_SAMPLE_RATE
        hop_s = settings.STFT_HOP_LENGTH / sample_rate  # 0.01s (10ms)

        # 1. Voiced Segments Analysis
        voiced_segments = []
        current_len = 0
        for is_v in voiced_list:
            if is_v:
                current_len += 1
            elif current_len > 0:
                voiced_segments.append(current_len * hop_s)
                current_len = 0
        if current_len > 0:
            voiced_segments.append(current_len * hop_s)

        segment_count = len(voiced_segments)
        avg_segment_dur = float(np.mean(voiced_segments)) if segment_count > 0 else 0.0

        # 2. Pause & Speaking Activity (Silence vs Active speech)
        win_size = settings.STFT_WIN_LENGTH
        hop_size = settings.STFT_HOP_LENGTH
        total_frames = max(1, (len(samples) - win_size) // hop_size + 1)

        frame_energies = []
        silent_count = 0
        for i in range(total_frames):
            frame = samples[i * hop_size : i * hop_size + win_size]
            rms = np.sqrt(np.mean(frame**2) + 1e-12)
            db = 20.0 * np.log10(rms + 1e-9)
            if db < settings.SILENCE_THRESHOLD_DB:
                silent_count += 1
            frame_energies.append(rms)

        pause_ratio = float(silent_count / total_frames)
        speaking_ratio = max(0.0, 1.0 - pause_ratio)

        # 3. Energy Variability (Coefficient of Variation)
        energy_mean = float(np.mean(frame_energies))
        energy_std = float(np.std(frame_energies))
        energy_var = energy_std / (energy_mean + 1e-9)

        # 4. F0 Variability
        voiced_f0 = [f for f in f0_list if f is not None]
        if len(voiced_f0) >= 3:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            f0_var = safe_float(f0_std / (f0_mean + 1e-9), None, 3)
        else:
            f0_var = None

        voiced_ratio = float(len(voiced_f0) / total_frames) if total_frames > 0 else 0.0

        return ProsodyFeatures(
            voiced_ratio=safe_float(voiced_ratio, 0.0, 3),
            pause_ratio=safe_float(pause_ratio, 0.0, 3),
            speaking_ratio=safe_float(speaking_ratio, 0.0, 3),
            voiced_segment_count=segment_count,
            avg_voiced_duration_s=safe_float(avg_segment_dur, 0.0, 3),
            energy_variability=safe_float(energy_var, 0.0, 3),
            f0_variability=f0_var,
        )
