"""Multi-format in-memory audio decoding using PyAV and FFmpeg."""

import io
from typing import Tuple
import av
import numpy as np
import soundfile as sf

from app.audio.validator import AudioFormatError, AudioEmptyError, AudioValidationError
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioDecoder:
    """In-memory audio decoder supporting WAV, MP3, M4A, WebM/Opus, OGG, FLAC."""

    @classmethod
    def decode(cls, file_bytes: bytes) -> Tuple[np.ndarray, str, int, int]:
        """
        Decode raw audio bytes into 16kHz mono float32 PCM numpy array.

        Returns:
            Tuple of:
                - samples: 1D np.ndarray (float32)
                - original_format: str (e.g., 'webm', 'wav', 'mp3', 'flac')
                - original_sample_rate: int
                - original_channels: int
        """
        if not file_bytes:
            raise AudioEmptyError("Cannot decode empty audio payload.")

        # Attempt decoding with PyAV first (primary multi-format & WebM/Opus decoder)
        try:
            return cls._decode_with_pyav(file_bytes)
        except AudioValidationError:
            raise
        except Exception as av_err:
            logger.debug("PyAV decode failed: %s. Attempting soundfile fallback.", av_err)

            # Fallback to soundfile (for WAV/FLAC/OGG standard PCM streams)
            try:
                return cls._decode_with_soundfile(file_bytes)
            except Exception as sf_err:
                logger.warning("All audio decoders failed. PyAV: %s | Soundfile: %s", av_err, sf_err)
                raise AudioFormatError(
                    f"Failed to decode audio file. Format is unrecognized, unsupported, or corrupted."
                )

    @classmethod
    def _decode_with_pyav(cls, file_bytes: bytes) -> Tuple[np.ndarray, str, int, int]:
        """Decode using PyAV with libswresample resampling to 16kHz mono float32."""
        stream = io.BytesIO(file_bytes)

        try:
            container = av.open(stream, mode="r")
        except Exception as e:
            raise AudioFormatError(f"Corrupted or invalid audio container: {str(e)}")

        # Find first audio stream
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise AudioFormatError("Container does not contain an audio stream.")

        audio_stream = audio_streams[0]
        original_sample_rate = audio_stream.rate or audio_stream.codec_context.sample_rate or 16000
        original_channels = audio_stream.channels or audio_stream.codec_context.channels or 1

        # Determine clean format name from container format
        raw_format = container.format.name.lower()
        if "webm" in raw_format or "matroska" in raw_format:
            detected_format = "webm"
        elif "wav" in raw_format:
            detected_format = "wav"
        elif "mp3" in raw_format:
            detected_format = "mp3"
        elif "ogg" in raw_format:
            detected_format = "ogg"
        elif "flac" in raw_format:
            detected_format = "flac"
        elif "mp4" in raw_format or "mov" in raw_format or "m4a" in raw_format:
            detected_format = "m4a"
        else:
            detected_format = raw_format.split(",")[0].strip()

        # Configure PyAV AudioResampler for target rate (16000) and layout (mono)
        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=settings.TARGET_SAMPLE_RATE,
        )

        pcm_chunks = []
        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    arr = rf.to_ndarray()  # shape (1, N)
                    pcm_chunks.append(arr.flatten())

        # Flush resampler buffer
        flushed_frames = resampler.resample(None)
        if flushed_frames:
            for rf in flushed_frames:
                arr = rf.to_ndarray()
                pcm_chunks.append(arr.flatten())

        if not pcm_chunks:
            raise AudioEmptyError("Audio stream contains no decodable audio frames.")

        samples = np.concatenate(pcm_chunks).astype(np.float32)

        return samples, detected_format, original_sample_rate, original_channels

    @classmethod
    def _decode_with_soundfile(cls, file_bytes: bytes) -> Tuple[np.ndarray, str, int, int]:
        """Decode using SoundFile and resample if necessary."""
        stream = io.BytesIO(file_bytes)
        data, sample_rate = sf.read(stream, dtype="float32", always_2d=True)

        if data.size == 0:
            raise AudioEmptyError("Soundfile decoded 0 samples.")

        num_channels = data.shape[1]
        # Mix down to mono by averaging channels
        if num_channels > 1:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data[:, 0]

        # Resample if sample_rate != target
        if sample_rate != settings.TARGET_SAMPLE_RATE:
            # Linear interpolation resampling for fallback
            duration = len(mono_data) / sample_rate
            target_num_samples = int(round(duration * settings.TARGET_SAMPLE_RATE))
            mono_data = np.interp(
                np.linspace(0.0, 1.0, target_num_samples, endpoint=False),
                np.linspace(0.0, 1.0, len(mono_data), endpoint=False),
                mono_data,
            ).astype(np.float32)

        return mono_data.astype(np.float32), "wav", sample_rate, num_channels
