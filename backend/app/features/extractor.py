"""Central acoustic feature extraction coordinator."""

from app.audio.schemas import AudioData
from app.features.temporal import TemporalFeatureExtractor
from app.features.spectral import SpectralFeatureExtractor
from app.features.prosody import ProsodyFeatureExtractor
from app.features.voice_quality import VoiceQualityFeatureExtractor
from app.features.schemas import AcousticFeatures
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Coordinates time-domain, spectral, harmonic, and prosodic feature extraction from standardized AudioData."""

    @classmethod
    def extract_from_audio_data(cls, audio_data: AudioData) -> AcousticFeatures:
        """
        Extract acoustic feature set from standardized 16kHz mono float32 AudioData.

        Args:
            audio_data: Validated 16kHz AudioData instance.

        Returns:
            AcousticFeatures structured model.
        """
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate

        # 1. Time-Domain Features
        time_domain = TemporalFeatureExtractor.extract(samples, sample_rate)

        # 2. Spectral STFT Features
        spectral = SpectralFeatureExtractor.extract_spectral_features(samples)

        # 3. 20-Coefficient MFCC Features
        mfcc = SpectralFeatureExtractor.extract_mfcc(samples)

        # 4. Pitch / F0 Statistics
        pitch = ProsodyFeatureExtractor.extract_pitch_features(samples)

        # 5. Voice Quality: Jitter, Shimmer & HNR
        voice_quality = VoiceQualityFeatureExtractor.extract(samples)

        # 6. Temporal Speech Prosody Dynamics
        prosody = ProsodyFeatureExtractor.extract_prosody_features(samples)

        logger.debug(
            "Feature extraction completed for duration=%.2fs | RMS=%.1fdB | F0=%s Hz | HNR=%s dB",
            audio_data.duration_seconds,
            time_domain.rms_db,
            str(pitch.f0_mean_hz),
            str(voice_quality.hnr_db),
        )

        return AcousticFeatures(
            audio_duration_seconds=audio_data.duration_seconds,
            time_domain=time_domain,
            spectral=spectral,
            mfcc=mfcc,
            pitch=pitch,
            voice_quality=voice_quality,
            prosody=prosody,
        )
