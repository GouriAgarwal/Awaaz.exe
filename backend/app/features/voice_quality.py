"""Acoustic perturbation and voice quality feature extraction (Jitter, Shimmer, HNR)."""

from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.features.prosody import ProsodyFeatureExtractor
from app.features.schemas import VoiceQualityFeatures
from app.features.validation import safe_float


class VoiceQualityFeatureExtractor:
    """Extracts cycle-to-cycle perturbation (Jitter, Shimmer) and Harmonics-to-Noise Ratio (HNR)."""

    @classmethod
    def extract(cls, samples: np.ndarray) -> VoiceQualityFeatures:
        """Calculate local jitter, shimmer, HNR, and harmonic/noise energy ratios."""
        if len(samples) == 0:
            return VoiceQualityFeatures(
                jitter=None,
                shimmer=None,
                hnr_db=None,
                harmonic_energy_ratio=None,
                noise_energy_estimate=None,
            )

        f0_list, _ = ProsodyFeatureExtractor.estimate_f0_frames(samples)
        sample_rate = settings.TARGET_SAMPLE_RATE
        win_size = settings.STFT_WIN_LENGTH
        hop_size = settings.STFT_HOP_LENGTH

        voiced_periods: List[float] = []
        voiced_amplitudes: List[float] = []
        hnr_linear_list: List[float] = []
        harmonic_ratio_list: List[float] = []

        total_frames = len(f0_list)
        for i in range(total_frames):
            f0 = f0_list[i]
            if f0 is not None and f0 > 0:
                period_s = 1.0 / f0
                voiced_periods.append(period_s)

                frame = samples[i * hop_size : i * hop_size + win_size]
                if len(frame) > 0:
                    peak_amp = float(np.max(np.abs(frame)))
                    voiced_amplitudes.append(peak_amp)

                    # Autocorrelation for HNR calculation
                    frame_c = frame - np.mean(frame)
                    r = np.correlate(frame_c, frame_c, mode="full")
                    center = len(r) // 2
                    r_half = r[center:]
                    r0 = float(r_half[0]) + 1e-12

                    lag_min = int(sample_rate / settings.PITCH_F0_MAX)
                    lag_max = min(len(r_half) - 1, int(sample_rate / settings.PITCH_F0_MIN))

                    if lag_min < lag_max:
                        r_peak = float(np.max(r_half[lag_min : lag_max + 1]))
                        harmonic_ratio = max(0.0, min(1.0, r_peak / r0))
                        harmonic_ratio_list.append(harmonic_ratio)

                        noise_residual = max(1e-9, r0 - r_peak)
                        hnr_lin = r_peak / noise_residual
                        hnr_linear_list.append(hnr_lin)

        # 1. Jitter (Local relative cycle-to-cycle F0 period variation)
        # Jitter = (mean(|T_{i+1} - T_i|)) / mean(T)
        if len(voiced_periods) >= 3:
            periods_arr = np.array(voiced_periods, dtype=np.float64)
            period_diffs = np.abs(np.diff(periods_arr))
            mean_period = float(np.mean(periods_arr))
            jitter_val = float(np.mean(period_diffs) / (mean_period + 1e-9))
            jitter = safe_float(jitter_val, None, 4)
        else:
            jitter = None

        # 2. Shimmer (Local relative cycle-to-cycle peak amplitude variation)
        # Shimmer = (mean(|A_{i+1} - A_i|)) / mean(A)
        if len(voiced_amplitudes) >= 3:
            amp_arr = np.array(voiced_amplitudes, dtype=np.float64)
            amp_diffs = np.abs(np.diff(amp_arr))
            mean_amp = float(np.mean(amp_arr))
            shimmer_val = float(np.mean(amp_diffs) / (mean_amp + 1e-9))
            shimmer = safe_float(shimmer_val, None, 4)
        else:
            shimmer = None

        # 3. Harmonics-to-Noise Ratio (HNR in dB)
        if len(hnr_linear_list) > 0:
            mean_hnr_lin = float(np.mean(hnr_linear_list))
            hnr_db_val = float(10.0 * np.log10(mean_hnr_lin + 1e-9))
            hnr_db_val = max(-20.0, min(50.0, hnr_db_val))
            hnr_db = safe_float(hnr_db_val, None, 1)

            mean_harm_ratio = float(np.mean(harmonic_ratio_list))
            harmonic_energy_ratio = safe_float(mean_harm_ratio, None, 3)
            noise_energy_estimate = safe_float(max(0.0, 1.0 - mean_harm_ratio), None, 3)
        else:
            hnr_db = None
            harmonic_energy_ratio = None
            noise_energy_estimate = None

        return VoiceQualityFeatures(
            jitter=jitter,
            shimmer=shimmer,
            hnr_db=hnr_db,
            harmonic_energy_ratio=harmonic_energy_ratio,
            noise_energy_estimate=noise_energy_estimate,
        )
