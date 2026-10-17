"""Pydantic schemas and dataclasses for acoustic, spectral, harmonic, and prosodic features."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TimeDomainFeatures(BaseModel):
    """Time-domain waveform statistics."""

    rms_db: float = Field(..., description="Root Mean Square energy in dBFS", examples=[-18.4])
    peak_db: float = Field(..., description="Peak amplitude in dBFS", examples=[-1.2])
    zero_crossing_rate: float = Field(..., description="Fraction of zero-crossings per sample", examples=[0.082])
    energy_mean: float = Field(..., description="Mean short-time frame energy", examples=[0.015])
    energy_std: float = Field(..., description="Standard deviation of frame energy", examples=[0.012])
    energy_p10: float = Field(..., description="10th percentile of frame energy", examples=[0.001])
    energy_p25: float = Field(..., description="25th percentile of frame energy", examples=[0.004])
    energy_p50: float = Field(..., description="Median (50th percentile) frame energy", examples=[0.011])
    energy_p75: float = Field(..., description="75th percentile of frame energy", examples=[0.024])
    energy_p90: float = Field(..., description="90th percentile of frame energy", examples=[0.038])


class SpectralFeatures(BaseModel):
    """Short-Time Fourier Transform (STFT) spectral distribution statistics."""

    centroid_hz: float = Field(..., description="Spectral centroid (center of mass in Hz)", examples=[1650.4])
    bandwidth_hz: float = Field(..., description="Spectral spread/bandwidth around centroid (Hz)", examples=[1220.8])
    rolloff_hz: float = Field(..., description="85% spectral energy rolloff frequency in Hz", examples=[3400.0])
    flatness: float = Field(..., description="Spectral flatness measure (0 = tonal, 1 = white noise)", examples=[0.042])
    entropy: float = Field(..., description="Spectral Shannon entropy (information dispersion)", examples=[0.685])
    flux: float = Field(..., description="Frame-to-frame spectral magnitude variation rate", examples=[0.128])
    low_energy_ratio: float = Field(..., description="Energy proportion in 0 - 1000 Hz band", examples=[0.58])
    mid_energy_ratio: float = Field(..., description="Energy proportion in 1000 - 4000 Hz band", examples=[0.34])
    high_energy_ratio: float = Field(..., description="Energy proportion in 4000 - 8000 Hz band", examples=[0.08])


class MFCCFeatures(BaseModel):
    """Mel-Frequency Cepstral Coefficients (20 bands)."""

    coefficients: int = Field(20, description="Number of computed MFCC coefficients")
    means: List[float] = Field(..., description="Mean coefficient values across frames (C0-C19)")
    stds: List[float] = Field(..., description="Standard deviation across frames (C0-C19)")


class PitchFeatures(BaseModel):
    """Fundamental frequency (F0) tracking and pitch statistics."""

    f0_mean_hz: Optional[float] = Field(None, description="Mean voiced pitch in Hz (null if unvoiced)", examples=[142.5])
    f0_median_hz: Optional[float] = Field(None, description="Median voiced pitch in Hz", examples=[139.8])
    f0_std_hz: Optional[float] = Field(None, description="Standard deviation of pitch in Hz", examples=[24.2])
    f0_min_hz: Optional[float] = Field(None, description="Minimum voiced pitch in Hz", examples=[95.0])
    f0_max_hz: Optional[float] = Field(None, description="Maximum voiced pitch in Hz", examples=[210.0])
    f0_p10_hz: Optional[float] = Field(None, description="10th percentile voiced pitch in Hz", examples=[110.0])
    f0_p25_hz: Optional[float] = Field(None, description="25th percentile voiced pitch in Hz", examples=[125.0])
    f0_p75_hz: Optional[float] = Field(None, description="75th percentile voiced pitch in Hz", examples=[160.0])
    f0_p90_hz: Optional[float] = Field(None, description="90th percentile voiced pitch in Hz", examples=[182.0])
    voiced_ratio: float = Field(..., description="Ratio of voiced frames to total speech frames", examples=[0.68])


class VoiceQualityFeatures(BaseModel):
    """Acoustic perturbation and voice quality metrics."""

    jitter: Optional[float] = Field(None, description="Local relative cycle-to-cycle F0 period variation", examples=[0.012])
    shimmer: Optional[float] = Field(None, description="Local relative cycle-to-cycle peak amplitude variation", examples=[0.038])
    hnr_db: Optional[float] = Field(None, description="Harmonics-to-Noise Ratio in dB", examples=[18.5])
    harmonic_energy_ratio: Optional[float] = Field(None, description="Proportion of periodic harmonic energy", examples=[0.82])
    noise_energy_estimate: Optional[float] = Field(None, description="Normalized estimate of aperiodic noise floor", examples=[0.05])


class ProsodyFeatures(BaseModel):
    """Temporal dynamics and speech rate prosodic features."""

    voiced_ratio: float = Field(..., description="Proportion of active speech frames that are voiced", examples=[0.68])
    pause_ratio: float = Field(..., description="Proportion of total duration identified as pause/silence", examples=[0.18])
    speaking_ratio: float = Field(..., description="Proportion of total duration with active speech", examples=[0.82])
    voiced_segment_count: int = Field(..., description="Count of contiguous voiced speech segments", examples=[7])
    avg_voiced_duration_s: float = Field(..., description="Average duration of voiced speech segments in seconds", examples=[0.38])
    energy_variability: float = Field(..., description="Coefficient of variation of short-time energy (std/mean)", examples=[0.62])
    f0_variability: Optional[float] = Field(None, description="Coefficient of variation of pitch (std/mean)", examples=[0.17])


class AcousticFeatures(BaseModel):
    """Combined structured acoustic feature representation."""

    audio_duration_seconds: float = Field(..., description="Audio duration in seconds", examples=[4.82])
    time_domain: TimeDomainFeatures
    spectral: SpectralFeatures
    mfcc: MFCCFeatures
    pitch: PitchFeatures
    voice_quality: VoiceQualityFeatures
    prosody: ProsodyFeatures


class FeatureExtractionResponse(BaseModel):
    """Complete API response schema for POST /api/features/extract."""

    success: bool = Field(True, description="Whether feature extraction succeeded")
    processing_time_ms: float = Field(..., description="Server execution time in milliseconds", examples=[32.4])
    features: AcousticFeatures
    explainability: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable acoustic parameter descriptions explaining physical properties.",
    )


# Standard Explainability Dictionary for API metadata
FEATURE_EXPLAINABILITY_METADATA: Dict[str, str] = {
    "rms_db": "Root Mean Square energy measuring overall acoustic signal power in decibels relative to full scale.",
    "peak_db": "Maximum absolute sample amplitude measured in decibels relative to full scale.",
    "zero_crossing_rate": "Rate at which signal changes sign; higher values indicate fricatives, high-frequency content, or noise.",
    "spectral_centroid": "Center of mass of the audio spectrum in Hz, reflecting perceived brightness or sharpness of sound.",
    "spectral_bandwidth": "Spectral spread around the centroid in Hz, describing the width of the frequency distribution.",
    "spectral_rolloff": "Frequency below which 85% of total spectral power is contained.",
    "spectral_flatness": "Ratio of geometric to arithmetic mean of power; 0 indicates pure tones, while 1 indicates white noise.",
    "spectral_entropy": "Shannon entropy of the normalized power spectrum; measures complexity and distribution spread.",
    "spectral_flux": "Rate of spectral change between successive frames, measuring acoustic transition dynamics.",
    "low_energy_ratio": "Proportion of total spectral power in the 0 - 1,000 Hz frequency band (fundamental and low harmonics).",
    "mid_energy_ratio": "Proportion of total spectral power in the 1,000 - 4,000 Hz band (critical speech vowel formant zone).",
    "high_energy_ratio": "Proportion of total spectral power in the 4,000 - 8,000 Hz band (fricatives and high-frequency harmonics).",
    "mfcc": "20 Mel-Frequency Cepstral Coefficients capturing the compact acoustic timbre and spectral envelope.",
    "f0_pitch": "Fundamental frequency of vocal fold vibration in Hz, representing speech melody and intonation.",
    "jitter": "Relative cycle-to-cycle perturbation in fundamental period, reflecting vocal fold stability.",
    "shimmer": "Relative cycle-to-cycle perturbation in peak amplitude, reflecting vocal intensity stability.",
    "hnr_db": "Harmonics-to-Noise Ratio in dB; quantifies relative energy of periodic harmonics versus aperiodic noise.",
    "voiced_ratio": "Proportion of speech frames exhibiting periodic vocal fold vibration.",
    "speaking_ratio": "Proportion of recording containing active speech energy above the background silence floor.",
}
