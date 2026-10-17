"""Spectral feature extraction and 20-coefficient Mel-Frequency Cepstral Coefficients (MFCCs)."""

import numpy as np
from typing import Tuple
from app.core.config import settings
from app.features.schemas import SpectralFeatures, MFCCFeatures
from app.features.validation import safe_float


class SpectralFeatureExtractor:
    """Extracts STFT-based spectral metrics and MFCCs from 16kHz mono float32 audio."""

    @classmethod
    def compute_stft(cls, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT).

        Returns:
            Tuple of:
                - magnitudes: 2D array of shape (num_frames, n_fft // 2 + 1)
                - powers: 2D array of shape (num_frames, n_fft // 2 + 1)
                - freqs: 1D array of frequency bin centers in Hz
        """
        n_fft = settings.STFT_N_FFT
        hop_length = settings.STFT_HOP_LENGTH
        win_length = settings.STFT_WIN_LENGTH
        sample_rate = settings.TARGET_SAMPLE_RATE

        if len(samples) < win_length:
            samples = np.pad(samples, (0, win_length - len(samples)))

        # Periodic Hann window
        window = np.hanning(win_length)
        total_frames = max(1, (len(samples) - win_length) // hop_length + 1)

        frames = []
        for i in range(total_frames):
            frame = samples[i * hop_length : i * hop_length + win_length]
            if len(frame) < win_length:
                frame = np.pad(frame, (0, win_length - len(frame)))
            windowed = frame * window
            # Pad windowed frame to n_fft if win_length < n_fft
            if win_length < n_fft:
                windowed = np.pad(windowed, (0, n_fft - win_length))
            frames.append(windowed)

        frames_arr = np.array(frames, dtype=np.float32)

        # Real FFT
        stft_complex = np.fft.rfft(frames_arr, n=n_fft, axis=1)
        magnitudes = np.abs(stft_complex)
        powers = magnitudes**2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        return magnitudes, powers, freqs

    @classmethod
    def extract_spectral_features(cls, samples: np.ndarray) -> SpectralFeatures:
        """Extract spectral centroid, bandwidth, rolloff, flatness, entropy, flux, and sub-band ratios."""
        if len(samples) == 0:
            return SpectralFeatures(
                centroid_hz=0.0,
                bandwidth_hz=0.0,
                rolloff_hz=0.0,
                flatness=0.0,
                entropy=0.0,
                flux=0.0,
                low_energy_ratio=0.0,
                mid_energy_ratio=0.0,
                high_energy_ratio=0.0,
            )

        magnitudes, powers, freqs = cls.compute_stft(samples)

        # 1. Spectral Centroid per frame
        mag_sum = np.sum(magnitudes, axis=1, keepdims=True) + 1e-12
        centroids = np.sum(magnitudes * freqs, axis=1, keepdims=True) / mag_sum
        mean_centroid = float(np.mean(centroids))

        # 2. Spectral Bandwidth per frame
        deviations = (freqs - centroids) ** 2
        bandwidths = np.sqrt(np.sum(deviations * magnitudes, axis=1, keepdims=True) / mag_sum)
        mean_bandwidth = float(np.mean(bandwidths))

        # 3. Spectral Rolloff (85% energy)
        power_sum = np.sum(powers, axis=1) + 1e-12
        cum_power = np.cumsum(powers, axis=1)
        rolloff_indices = np.argmax(cum_power >= 0.85 * power_sum[:, None], axis=1)
        rolloff_freqs = freqs[rolloff_indices]
        mean_rolloff = float(np.mean(rolloff_freqs))

        # 4. Spectral Flatness (Geometric Mean / Arithmetic Mean of power)
        log_power = np.log(powers + 1e-12)
        geom_mean = np.exp(np.mean(log_power, axis=1))
        arith_mean = np.mean(powers, axis=1) + 1e-12
        flatness_per_frame = geom_mean / arith_mean
        mean_flatness = float(np.mean(flatness_per_frame))

        # 5. Spectral Entropy (Normalized Shannon Entropy)
        norm_powers = powers / power_sum[:, None]
        entropy_per_frame = -np.sum(norm_powers * np.log2(norm_powers + 1e-12), axis=1) / np.log2(len(freqs))
        mean_entropy = float(np.mean(entropy_per_frame))

        # 6. Spectral Flux (Euclidean distance between adjacent normalized frames)
        if magnitudes.shape[0] > 1:
            norm_mags = magnitudes / mag_sum
            flux_per_frame = np.sqrt(np.sum((norm_mags[1:] - norm_mags[:-1]) ** 2, axis=1))
            mean_flux = float(np.mean(flux_per_frame))
        else:
            mean_flux = 0.0

        # 7. Sub-band Energy Ratios
        low_mask = freqs <= 1000.0
        mid_mask = (freqs > 1000.0) & (freqs <= 4000.0)
        high_mask = freqs > 4000.0

        total_energy = np.sum(powers) + 1e-12
        low_ratio = float(np.sum(powers[:, low_mask]) / total_energy)
        mid_ratio = float(np.sum(powers[:, mid_mask]) / total_energy)
        high_ratio = float(np.sum(powers[:, high_mask]) / total_energy)

        return SpectralFeatures(
            centroid_hz=safe_float(mean_centroid, 0.0, 1),
            bandwidth_hz=safe_float(mean_bandwidth, 0.0, 1),
            rolloff_hz=safe_float(mean_rolloff, 0.0, 1),
            flatness=safe_float(mean_flatness, 0.0, 4),
            entropy=safe_float(mean_entropy, 0.0, 4),
            flux=safe_float(mean_flux, 0.0, 4),
            low_energy_ratio=safe_float(low_ratio, 0.0, 3),
            mid_energy_ratio=safe_float(mid_ratio, 0.0, 3),
            high_energy_ratio=safe_float(high_ratio, 0.0, 3),
        )

    @classmethod
    def extract_mfcc(cls, samples: np.ndarray, num_coeffs: int = settings.MFCC_N_COEFFS) -> MFCCFeatures:
        """Extract 20 MFCC coefficient means and standard deviations."""
        if len(samples) == 0:
            return MFCCFeatures(
                coefficients=num_coeffs,
                means=[0.0] * num_coeffs,
                stds=[0.0] * num_coeffs,
            )

        _, powers, freqs = cls.compute_stft(samples)
        sample_rate = settings.TARGET_SAMPLE_RATE

        # Construct Mel Filterbank
        mel_min = 0.0
        mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(mel_min, mel_max, num_coeffs + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((settings.STFT_N_FFT + 1) * hz_points / sample_rate).astype(int)

        n_bins = len(freqs)
        filterbank = np.zeros((num_coeffs, n_bins), dtype=np.float32)

        for m in range(1, num_coeffs + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if k < n_bins and f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if k < n_bins and f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        # Mel spectrum energies
        mel_energies = np.dot(powers, filterbank.T) + 1e-12
        log_mel_energies = np.log(mel_energies)

        # Type-II DCT Matrix
        num_filters = num_coeffs
        dct_matrix = np.empty((num_coeffs, num_filters), dtype=np.float32)
        for k in range(num_coeffs):
            factor = np.sqrt(1.0 / num_filters) if k == 0 else np.sqrt(2.0 / num_filters)
            dct_matrix[k, :] = factor * np.cos(np.pi * k * (np.arange(num_filters) + 0.5) / num_filters)

        mfcc_matrix = np.dot(log_mel_energies, dct_matrix.T)  # (num_frames, num_coeffs)

        means = np.mean(mfcc_matrix, axis=0)
        stds = np.std(mfcc_matrix, axis=0)

        return MFCCFeatures(
            coefficients=num_coeffs,
            means=[safe_float(m, 0.0, 3) for m in means],
            stds=[safe_float(s, 0.0, 3) for s in stds],
        )
