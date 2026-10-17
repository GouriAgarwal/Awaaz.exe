"""Comprehensive unit tests for the audio decoding, preprocessing, and validation pipeline."""

import io
import pytest
import numpy as np
import soundfile as sf
import av

from app.audio.decoder import AudioDecoder
from app.audio.normalizer import AudioNormalizer
from app.audio.processor import AudioProcessor
from app.audio.validator import (
    AudioValidator,
    AudioEmptyError,
    AudioFormatError,
    AudioSizeError,
    AudioDurationError,
    AudioSignalError,
)


def _create_synthetic_wav_bytes(
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory WAV audio bytes with a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    if channels > 1:
        signal = np.column_stack([signal] * channels)

    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _create_synthetic_flac_bytes(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory FLAC audio bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.6 * np.sin(2 * np.pi * 520.0 * t).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="FLAC")
    return buf.getvalue()


def _create_synthetic_ogg_bytes(duration: float = 1.0, sample_rate: int = 48000) -> bytes:
    """Generate in-memory OGG audio bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 330.0 * t).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="OGG")
    return buf.getvalue()


def _create_synthetic_webm_opus_bytes(duration: float = 1.0, sample_rate: int = 48000, channels: int = 2) -> bytes:
    """Encode synthetic multi-channel sine wave into WebM/Opus container using PyAV."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=sample_rate)
    stream.layout = "stereo" if channels == 2 else "mono"
    stream.format = "flt"

    total_samples = int(sample_rate * duration)
    frame_size = 960  # 20ms at 48kHz
    t_full = np.linspace(0, duration, total_samples, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440.0 * t_full)).astype(np.float32)

    for i in range(0, total_samples, frame_size):
        chunk = signal[i : i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))

        if channels == 2:
            # Interleaved stereo samples in (1, frame_size * 2)
            interleaved = np.empty((frame_size * 2,), dtype=np.float32)
            interleaved[0::2] = chunk
            interleaved[1::2] = chunk
            frame_data = np.ascontiguousarray(interleaved.reshape(1, -1))
        else:
            frame_data = np.ascontiguousarray(chunk.reshape(1, -1))

        frame = av.AudioFrame.from_ndarray(frame_data, format="flt", layout="stereo" if channels == 2 else "mono")
        frame.rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return buf.getvalue()


def _create_synthetic_mp3_bytes(duration: float = 1.0, sample_rate: int = 44100) -> bytes:
    """Encode synthetic sine wave into MP3 container using PyAV."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp3")
    stream = container.add_stream("libmp3lame", rate=sample_rate)
    stream.layout = "mono"
    stream.format = "fltp"

    total_samples = int(sample_rate * duration)
    frame_size = 1152  # MP3 frame size
    t_full = np.linspace(0, duration, total_samples, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 880.0 * t_full)).astype(np.float32)

    for i in range(0, total_samples, frame_size):
        chunk = signal[i : i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))

        frame_data = np.ascontiguousarray(chunk.reshape(1, -1))
        frame = av.AudioFrame.from_ndarray(frame_data, format="flt", layout="mono")
        frame.rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_valid_wav_audio() -> None:
    """Test A: Valid WAV audio is decoded and standardized to 16kHz mono float32."""
    raw_bytes = _create_synthetic_wav_bytes(duration=2.0, sample_rate=16000, channels=1)
    audio_data, metrics = AudioProcessor.process(raw_bytes)

    assert audio_data.sample_rate == 16000
    assert audio_data.channels == 1
    assert audio_data.samples.dtype == np.float32
    assert 1.9 <= audio_data.duration_seconds <= 2.1
    assert metrics.quality in ["good", "warning"]
    assert metrics.rms_db > -40.0


def test_valid_webm_opus_audio() -> None:
    """Test B: Valid WebM/Opus (browser microphone format) decodes cleanly."""
    raw_bytes = _create_synthetic_webm_opus_bytes(duration=1.5, sample_rate=48000, channels=2)
    audio_data, metrics = AudioProcessor.process(raw_bytes)

    assert audio_data.sample_rate == 16000
    assert audio_data.channels == 1
    assert audio_data.original_format == "webm"
    assert audio_data.original_sample_rate == 48000
    assert 1.4 <= audio_data.duration_seconds <= 1.6
    assert metrics.rms_db > -40.0


def test_valid_mp3_audio() -> None:
    """Test C: Valid MP3 audio decodes properly."""
    raw_bytes = _create_synthetic_mp3_bytes(duration=1.2, sample_rate=44100)
    audio_data, metrics = AudioProcessor.process(raw_bytes)

    assert audio_data.sample_rate == 16000
    assert audio_data.channels == 1
    assert audio_data.original_format == "mp3"
    assert 1.0 <= audio_data.duration_seconds <= 1.4


def test_valid_flac_and_ogg_audio() -> None:
    """Test D: Valid FLAC and OGG audio formats decode properly."""
    flac_bytes = _create_synthetic_flac_bytes(duration=1.0, sample_rate=16000)
    audio_flac, _ = AudioProcessor.process(flac_bytes)
    assert audio_flac.sample_rate == 16000
    assert 0.9 <= audio_flac.duration_seconds <= 1.1

    ogg_bytes = _create_synthetic_ogg_bytes(duration=1.0, sample_rate=48000)
    audio_ogg, _ = AudioProcessor.process(ogg_bytes)
    assert audio_ogg.sample_rate == 16000
    assert 0.9 <= audio_ogg.duration_seconds <= 1.1


def test_stereo_to_mono_conversion() -> None:
    """Test E: Stereo input is converted to single-channel mono."""
    raw_bytes = _create_synthetic_wav_bytes(duration=1.0, sample_rate=16000, channels=2)
    audio_data, _ = AudioProcessor.process(raw_bytes)

    assert audio_data.channels == 1
    assert audio_data.original_channels == 2
    assert audio_data.samples.ndim == 1


def test_non_16khz_resampling() -> None:
    """Test F: Audio sampled at non-16kHz rates (e.g. 44.1kHz, 48kHz) is accurately resampled to 16kHz."""
    for sr in [8000, 44100, 48000]:
        raw_bytes = _create_synthetic_wav_bytes(duration=1.0, sample_rate=sr, channels=1)
        audio_data, _ = AudioProcessor.process(raw_bytes)

        assert audio_data.sample_rate == 16000
        assert audio_data.original_sample_rate == sr
        assert 0.95 <= audio_data.duration_seconds <= 1.05


def test_silent_audio_quality_detection() -> None:
    """Test G: Silent audio produces low RMS energy and high silence ratio."""
    raw_bytes = _create_synthetic_wav_bytes(duration=1.0, sample_rate=16000, amplitude=0.0)
    audio_data, metrics = AudioProcessor.process(raw_bytes)

    assert metrics.silence_ratio > 0.9
    assert metrics.quality in ["invalid", "warning"]


def test_corrupted_audio_rejection() -> None:
    """Test H: Corrupted or truncated audio files raise AudioFormatError."""
    corrupted_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\xFF" * 50
    with pytest.raises(AudioFormatError):
        AudioProcessor.process(corrupted_bytes)


def test_empty_audio_upload() -> None:
    """Test I: Empty byte payloads raise AudioEmptyError."""
    with pytest.raises(AudioEmptyError):
        AudioProcessor.process(b"")


def test_unsupported_file_format() -> None:
    """Test J: Arbitrary non-audio binary and text raise AudioFormatError."""
    non_audio = b"{\x22title\x22: \x22Not an audio file\x22, \x22version\x22: 1}"
    with pytest.raises(AudioFormatError):
        AudioProcessor.process(non_audio)


def test_oversized_file_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test K: Files exceeding configured MAX_FILE_SIZE_MB raise AudioSizeError."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)  # 1MB limit for test

    huge_payload = b"0" * (2 * 1024 * 1024)  # 2MB
    with pytest.raises(AudioSizeError):
        AudioValidator.validate_raw_bytes(huge_payload)


def test_very_short_audio_rejection() -> None:
    """Test L: Audio shorter than MIN_DURATION_SECONDS is rejected."""
    raw_bytes = _create_synthetic_wav_bytes(duration=0.2, sample_rate=16000)
    with pytest.raises(AudioDurationError):
        AudioProcessor.process(raw_bytes)


def test_very_long_audio_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test M: Audio exceeding MAX_DURATION_SECONDS is rejected."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "MAX_DURATION_SECONDS", 2.0)

    raw_bytes = _create_synthetic_wav_bytes(duration=3.0, sample_rate=16000)
    with pytest.raises(AudioDurationError):
        AudioProcessor.process(raw_bytes)


def test_nan_or_inf_signal_rejection() -> None:
    """Test N: Samples containing NaN or infinite values raise AudioSignalError."""
    bad_samples = np.array([0.1, np.nan, 0.5, np.inf], dtype=np.float32)
    with pytest.raises(AudioSignalError):
        AudioValidator.validate_samples(bad_samples, sample_rate=16000, min_duration=0.0)
