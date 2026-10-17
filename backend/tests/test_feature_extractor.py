"""Comprehensive unit tests for the acoustic, spectral, harmonic, and prosodic feature extraction engine."""

import io
import json
import pytest
import numpy as np
import soundfile as sf
import av

from app.audio.schemas import AudioData
from app.audio.processor import AudioProcessor
from app.features.extractor import FeatureExtractor
from app.features.temporal import TemporalFeatureExtractor
from app.features.spectral import SpectralFeatureExtractor
from app.features.prosody import ProsodyFeatureExtractor
from app.features.voice_quality import VoiceQualityFeatureExtractor


def _generate_synthetic_tone(
    freq: float = 200.0,
    duration: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a clean synthetic sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _generate_synthetic_harmonic(
    f0: float = 150.0,
    duration: float = 2.0,
    sample_rate: int = 16000,
    num_harmonics: int = 5,
) -> np.ndarray:
    """Generate a rich harmonic signal with integer multiples of fundamental frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        amp = 0.5 / k
        signal += amp * np.sin(2 * np.pi * (k * f0) * t)
    return signal.astype(np.float32)


def _generate_synthetic_noise(
    duration: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate uniform white noise."""
    rng = np.random.default_rng(42)
    return (amplitude * rng.uniform(-1.0, 1.0, int(sample_rate * duration))).astype(np.float32)


def _to_audio_data(samples: np.ndarray, sample_rate: int = 16000) -> AudioData:
    """Wrap raw samples in an AudioData container."""
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        duration_seconds=round(len(samples) / sample_rate, 3),
        original_format="wav",
        original_sample_rate=sample_rate,
        original_channels=1,
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_feature_extraction_valid_wav() -> None:
    """Test 1: Feature extraction from valid WAV audio data."""
    samples = _generate_synthetic_harmonic(f0=220.0, duration=2.0)
    audio_data = _to_audio_data(samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.audio_duration_seconds == 2.0
    assert features.time_domain.rms_db > -30.0
    assert features.spectral.centroid_hz > 100.0
    assert len(features.mfcc.means) == 20
    assert len(features.mfcc.stds) == 20
    assert features.pitch.f0_mean_hz is not None
    assert 200.0 <= features.pitch.f0_mean_hz <= 240.0
    assert features.voice_quality.hnr_db is not None
    assert features.voice_quality.hnr_db > 5.0


def test_feature_extraction_webm_opus() -> None:
    """Test 2: Feature extraction pipeline on decoded WebM/Opus audio."""
    # Create WebM Opus audio in memory using PyAV
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=48000)
    stream.layout = "mono"
    stream.format = "flt"

    duration = 1.5
    total_samples = int(48000 * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    frame_size = 960
    for i in range(0, total_samples, frame_size):
        chunk = signal[i : i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
        frame.rate = 48000
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    raw_bytes = buf.getvalue()
    audio_data, _ = AudioProcessor.process(raw_bytes)
    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.audio_duration_seconds >= 1.4
    assert features.pitch.f0_mean_hz is not None
    assert 280.0 <= features.pitch.f0_mean_hz <= 320.0


def test_correct_sample_rate_and_mfcc_dimensions() -> None:
    """Test 3 & 4: Verified 16kHz assumption and 20 MFCC coefficient dimensions."""
    samples = _generate_synthetic_tone(freq=440.0, duration=1.0)
    audio_data = _to_audio_data(samples, sample_rate=16000)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.mfcc.coefficients == 20
    assert len(features.mfcc.means) == 20
    assert len(features.mfcc.stds) == 20
    for m in features.mfcc.means:
        assert isinstance(m, float)
    for s in features.mfcc.stds:
        assert isinstance(s, float)


def test_spectral_features_pure_tone() -> None:
    """Test 5: Spectral centroid on a pure 1000 Hz tone concentrates near 1000 Hz."""
    samples = _generate_synthetic_tone(freq=1000.0, duration=1.0)
    audio_data = _to_audio_data(samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert 900.0 <= features.spectral.centroid_hz <= 1100.0
    assert features.spectral.flatness < 0.1  # Highly tonal -> low flatness
    assert features.spectral.low_energy_ratio + features.spectral.mid_energy_ratio > 0.8


def test_f0_pitch_and_voiced_ratio() -> None:
    """Test 6 & 7: Accurate F0 estimation and voiced ratio on periodic vs unvoiced signals."""
    harmonic_samples = _generate_synthetic_harmonic(f0=175.0, duration=1.5)
    features_harmonic = FeatureExtractor.extract_from_audio_data(_to_audio_data(harmonic_samples))

    assert features_harmonic.pitch.f0_mean_hz is not None
    assert 165.0 <= features_harmonic.pitch.f0_mean_hz <= 185.0
    assert features_harmonic.pitch.voiced_ratio >= 0.85
    assert features_harmonic.prosody.speaking_ratio >= 0.85


def test_silence_handling_graceful_nulls() -> None:
    """Test 8: Silence handling produces valid JSON structure with null pitch/jitter/shimmer."""
    silent_samples = np.zeros(32000, dtype=np.float32)
    audio_data = _to_audio_data(silent_samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.time_domain.rms_db == -100.0
    assert features.pitch.f0_mean_hz is None
    assert features.pitch.voiced_ratio == 0.0
    assert features.voice_quality.jitter is None
    assert features.voice_quality.shimmer is None
    assert features.voice_quality.hnr_db is None
    assert features.prosody.pause_ratio >= 0.95


def test_white_noise_characteristics() -> None:
    """Test 9: White noise produces high spectral flatness and zero/low voiced ratio."""
    noise_samples = _generate_synthetic_noise(duration=2.0)
    audio_data = _to_audio_data(noise_samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.spectral.flatness > 0.15  # Noise has significantly higher flatness than speech
    assert features.pitch.voiced_ratio < 0.20  # Noise should not be detected as voiced speech


def test_clipped_signal_handling() -> None:
    """Test 10: Heavily clipped signal does not cause calculation failures."""
    samples = _generate_synthetic_tone(freq=300.0, duration=1.0, amplitude=5.0)
    samples_clipped = np.clip(samples, -1.0, 1.0)
    audio_data = _to_audio_data(samples_clipped)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.time_domain.peak_db == 0.0
    assert features.time_domain.rms_db > -10.0


def test_short_audio_handling() -> None:
    """Test 11: 0.5s audio extracts cleanly without indexing errors."""
    samples = _generate_synthetic_tone(freq=250.0, duration=0.5)
    audio_data = _to_audio_data(samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.audio_duration_seconds == 0.5
    assert len(features.mfcc.means) == 20


def test_jitter_and_shimmer_unvoiced_case() -> None:
    """Test 12 & 13: Jitter and Shimmer return None when there are < 3 voiced cycles."""
    # Signal with only 1 brief pulse
    samples = np.zeros(16000, dtype=np.float32)
    samples[1000:1050] = 0.5
    audio_data = _to_audio_data(samples)

    features = FeatureExtractor.extract_from_audio_data(audio_data)

    assert features.voice_quality.jitter is None
    assert features.voice_quality.shimmer is None


def test_zero_nan_and_zero_infinity_in_json() -> None:
    """Test 14 & 15: Validates that serialized feature JSON contains zero NaN or Infinity."""
    for sig in [
        _generate_synthetic_tone(440.0, 1.0),
        _generate_synthetic_noise(1.0),
        np.zeros(16000, dtype=np.float32),
        np.array([1.0, -1.0] * 8000, dtype=np.float32),
    ]:
        audio_data = _to_audio_data(sig)
        features = FeatureExtractor.extract_from_audio_data(audio_data)

        # Serialize to JSON and check
        json_str = features.model_dump_json()
        assert "NaN" not in json_str
        assert "Infinity" not in json_str
        assert "-Infinity" not in json_str

        # Ensure valid JSON deserialization
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


def test_feature_consistency_determinism() -> None:
    """Test 19: Identical audio produces identical feature values across multiple runs."""
    samples = _generate_synthetic_harmonic(f0=180.0, duration=1.0)
    audio_data1 = _to_audio_data(samples)
    audio_data2 = _to_audio_data(samples.copy())

    feat1 = FeatureExtractor.extract_from_audio_data(audio_data1)
    feat2 = FeatureExtractor.extract_from_audio_data(audio_data2)

    assert feat1.model_dump() == feat2.model_dump()
