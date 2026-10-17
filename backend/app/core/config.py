"""Centralized application configuration using Pydantic Settings."""

from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    PROJECT_NAME: str = "VoiceShield API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    API_V1_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    LOG_LEVEL: str = "INFO"

    # Audio Pipeline Configuration
    MAX_FILE_SIZE_MB: int = 25
    MIN_DURATION_SECONDS: float = 0.5
    MAX_DURATION_SECONDS: float = 300.0
    TARGET_SAMPLE_RATE: int = 16000
    TARGET_CHANNELS: int = 1
    SILENCE_THRESHOLD_DB: float = -50.0
    CLIPPING_THRESHOLD: float = 0.999

    # Feature Extraction & STFT Parameters
    STFT_N_FFT: int = 512
    STFT_HOP_LENGTH: int = 160      # 10 ms at 16 kHz
    STFT_WIN_LENGTH: int = 400      # 25 ms at 16 kHz
    MFCC_N_COEFFS: int = 20
    PITCH_F0_MIN: float = 50.0      # Hz
    PITCH_F0_MAX: float = 500.0     # Hz
    VOICING_THRESHOLD: float = 0.35 # Autocorrelation peak threshold for voiced decision

    # Synthetic Voice Detection Parameters (Step 4)
    DETECTION_WINDOW_SECONDS: float = 3.0
    DETECTION_HOP_SECONDS: float = 1.5
    DETECTION_THRESHOLD_NATURAL: float = 0.35
    DETECTION_THRESHOLD_SYNTHETIC: float = 0.65
    DETECTION_AGGREGATION_METHOD: str = "median"
    DETECTION_DEVICE: str = "cpu"

    # Streaming & Security Parameters (Step 10 / Step 11 / Step 12)
    MAX_ACTIVE_SESSIONS: int = 20
    MAX_STREAM_CHUNK_BYTES: int = 1048576  # 1 MB
    MAX_STREAM_MESSAGE_BYTES: int = 2097152  # 2 MB
    STREAM_RATE_LIMIT_CHUNKS_PER_SEC: int = 50
    SESSION_IDLE_TTL_SEC: float = 120.0
    STRICT_MODEL_INTEGRITY_CHECK: bool = True

    # Model Artifact Paths and Cryptographic Checksums
    AASIST_MODEL_PATH: Union[str, None] = None
    AASIST_MODEL_SHA256: str = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
    ECAPA_MODEL_PATH: Union[str, None] = None
    ECAPA_MODEL_SHA256: str = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"

    # Observability & Operational Settings (Step 12)
    METRICS_ENABLED: bool = True
    STRUCTURED_LOGGING_ENABLED: bool = True
    CORS_ALLOWED_ORIGINS: Union[List[str], str, None] = None


    @property
    def max_file_size_bytes(self) -> int:
        """Return maximum file size converted to bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


settings = Settings()
