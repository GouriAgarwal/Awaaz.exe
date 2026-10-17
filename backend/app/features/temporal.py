"""Time-domain acoustic feature extraction."""

import numpy as np
from app.features.schemas import TimeDomainFeatures
from app.features.validation import safe_float


class TemporalFeatureExtractor:
    """Extracts waveform energy, peak levels, zero-crossing rate, and short-time energy percentiles."""

    @classmethod
    def extract(cls, samples: np.ndarray, sample_rate: int = 16000) -> TimeDomainFeatures:
        """Extract time-domain statistics from 16kHz mono float32 samples."""
        if len(samples) == 0:
            return TimeDomainFeatures(
                rms_db=-100.0,
                peak_db=-100.0,
                zero_crossing_rate=0.0,
                energy_mean=0.0,
                energy_std=0.0,
                energy_p10=0.0,
                energy_p25=0.0,
                energy_p50=0.0,
                energy_p75=0.0,
                energy_p90=0.0,
            )

        # 1. Global RMS (dBFS)
        rms = float(np.sqrt(np.mean(samples**2) + 1e-12))
        rms_db = float(20.0 * np.log10(rms + 1e-9))
        rms_db = max(-100.0, min(0.0, rms_db))

        # 2. Peak Amplitude (dBFS)
        peak = float(np.max(np.abs(samples)))
        peak_db = float(20.0 * np.log10(peak + 1e-9))
        peak_db = max(-100.0, min(0.0, peak_db))

        # 3. Zero-Crossing Rate (ZCR)
        # Fraction of consecutive sign changes
        sign_changes = np.abs(np.diff(np.signbit(samples)))
        zcr = float(np.mean(sign_changes)) if len(sign_changes) > 0 else 0.0

        # 4. Short-Time Energy Statistics (25ms window, 10ms hop)
        win_size = int(0.025 * sample_rate)  # 400 samples
        hop_size = int(0.010 * sample_rate)  # 160 samples
        total_frames = max(1, (len(samples) - win_size) // hop_size + 1)

        frame_energies = []
        for i in range(total_frames):
            frame = samples[i * hop_size : i * hop_size + win_size]
            if len(frame) > 0:
                frame_energies.append(float(np.mean(frame**2)))

        if not frame_energies:
            frame_energies = [0.0]

        energies_arr = np.array(frame_energies, dtype=np.float64)

        return TimeDomainFeatures(
            rms_db=safe_float(rms_db, -100.0, 2),
            peak_db=safe_float(peak_db, -100.0, 2),
            zero_crossing_rate=safe_float(zcr, 0.0, 4),
            energy_mean=safe_float(np.mean(energies_arr), 0.0, 6),
            energy_std=safe_float(np.std(energies_arr), 0.0, 6),
            energy_p10=safe_float(np.percentile(energies_arr, 10), 0.0, 6),
            energy_p25=safe_float(np.percentile(energies_arr, 25), 0.0, 6),
            energy_p50=safe_float(np.percentile(energies_arr, 50), 0.0, 6),
            energy_p75=safe_float(np.percentile(energies_arr, 75), 0.0, 6),
            energy_p90=safe_float(np.percentile(energies_arr, 90), 0.0, 6),
        )
