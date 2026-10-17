"""Audio schemas and internal data representations."""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from pydantic import BaseModel, Field


@dataclass
class AudioData:
    """Standardized internal representation of decoded audio data."""

    samples: np.ndarray  # 1D float32 numpy array, normalized [-1.0, 1.0]
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float = 0.0
    original_format: str = "unknown"
    original_sample_rate: int = 16000
    original_channels: int = 1

    def __post_init__(self) -> None:
        if self.duration_seconds == 0.0 and len(self.samples) > 0 and self.sample_rate > 0:
            self.duration_seconds = round(len(self.samples) / self.sample_rate, 4)


@dataclass
class AudioQualityMetrics:
    """Acoustic signal quality and integrity analysis metrics."""

    rms_db: float
    peak_db: float
    clipping_ratio: float
    silence_ratio: float
    quality: str  # "good" | "warning" | "invalid"
    notes: List[str] = field(default_factory=list)


# Pydantic schemas for API serialization


class AudioMetadataSchema(BaseModel):
    """Processed audio file metadata."""

    duration_seconds: float = Field(..., description="Duration in seconds", examples=[4.82])
    sample_rate: int = Field(..., description="Standardized sample rate (Hz)", examples=[16000])
    channels: int = Field(..., description="Standardized channel count", examples=[1])
    original_format: str = Field(..., description="Detected original container/format", examples=["webm"])
    original_sample_rate: int = Field(..., description="Original input sample rate (Hz)", examples=[48000])
    original_channels: int = Field(..., description="Original input channel count", examples=[2])


class AudioQualitySchema(BaseModel):
    """Acoustic signal quality analysis schema."""

    rms_db: float = Field(..., description="Root Mean Square energy in dBFS", examples=[-18.4])
    peak_db: float = Field(..., description="Peak amplitude in dBFS", examples=[-1.2])
    clipping_ratio: float = Field(..., description="Proportion of clipped samples [0.0 - 1.0]", examples=[0.001])
    silence_ratio: float = Field(..., description="Proportion of silent frames [0.0 - 1.0]", examples=[0.03])
    quality: str = Field(..., description="Signal quality status: 'good' | 'warning' | 'invalid'", examples=["good"])
    notes: List[str] = Field(default_factory=list, description="Diagnostic observations regarding signal quality")


class AudioInspectResponse(BaseModel):
    """Response schema for POST /api/audio/inspect."""

    success: bool = Field(True, description="Whether audio inspection succeeded")
    audio: AudioMetadataSchema = Field(..., description="Standardized audio properties")
    quality: AudioQualitySchema = Field(..., description="Acoustic quality metrics")
