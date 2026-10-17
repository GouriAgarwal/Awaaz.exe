"""Pydantic schemas for AI synthetic / cloned voice detection (Step 4)."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ScoreType(str, Enum):
    """Type of score returned by the detection engine."""
    UNCALIBRATED_MODEL_SCORE = "uncalibrated_model_score"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    HEURISTIC_FALLBACK_SCORE = "heuristic_fallback_score"


class ClassificationLabel(str, Enum):
    """Voice classification result."""
    NATURAL = "NATURAL"
    SYNTHETIC = "SYNTHETIC"
    UNCERTAIN = "UNCERTAIN"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class DetectorMetadata(BaseModel):
    """Provenance and architectural metadata for the active voice detector."""
    model_name: str = Field(..., description="Human-readable model name or identifier")
    model_type: str = Field(..., description="Category of model (e.g. 'deep_learning_acoustic', 'spectral_cnn', 'fallback')")
    architecture: str = Field(..., description="Underlying neural or acoustic architecture")
    checkpoint_or_source: str = Field(..., description="Model repository, checkpoint hash, or builtin descriptor")
    license: str = Field(..., description="Model license (e.g. 'Apache 2.0', 'MIT', 'CC-BY-NC')")
    expected_sample_rate: int = Field(default=16000, description="Target sample rate in Hz")
    window_size_sec: float = Field(default=3.0, description="Temporal window duration in seconds")
    window_hop_sec: float = Field(default=1.5, description="Temporal window hop/stride in seconds")
    score_type: ScoreType = Field(default=ScoreType.UNCALIBRATED_MODEL_SCORE, description="Raw score vs calibrated probability")
    score_interpretation: str = Field(..., description="Explanation of score directionality and range")
    scientific_disclaimer: str = Field(..., description="Explicit scientific limitation and evaluation notice")
    device: str = Field(default="cpu", description="Compute device utilized (cpu or cuda)")
    is_fallback: bool = Field(default=False, description="Whether fallback engine is currently handling inference")


class WindowScore(BaseModel):
    """Detection score and classification for an individual temporal window."""
    window_index: int = Field(..., description="0-indexed temporal window index")
    start_sec: float = Field(..., description="Window start time in seconds")
    end_sec: float = Field(..., description="Window end time in seconds")
    raw_score: float = Field(..., description="Raw model activation score")
    synthetic_score: float = Field(..., description="Normalized synthetic likelihood indicator in [0.0, 1.0]")
    label: ClassificationLabel = Field(..., description="Decision label for this temporal window")


class DetectionResult(BaseModel):
    """Complete synthetic voice detection analysis result."""
    detector_metadata: DetectorMetadata = Field(..., description="Active detector provenance and configuration")
    classification: ClassificationLabel = Field(..., description="Final aggregated classification")
    score: float = Field(..., description="Aggregated synthetic score in [0.0, 1.0]")
    score_type: ScoreType = Field(..., description="Score provenance (uncalibrated_model_score, calibrated_probability, heuristic_fallback_score)")
    confidence_band: str = Field(..., description="Qualitative confidence band: HIGH, MEDIUM, LOW, or UNCERTAIN")
    thresholds_applied: Dict[str, float] = Field(..., description="Threshold boundaries applied for classification")
    window_scores: List[WindowScore] = Field(default_factory=list, description="Per-window granular scores")
    aggregation_method: str = Field(..., description="Method used to aggregate window scores (e.g. 'median', 'mean')")
    inference_latency_ms: float = Field(..., description="Total inference time in milliseconds")
    audio_duration_sec: float = Field(..., description="Total input audio duration in seconds")
    total_windows: int = Field(..., description="Number of temporal windows processed")
    warnings: List[str] = Field(default_factory=list, description="Quality, calibration, or runtime warnings")


class DetectionResponse(BaseModel):
    """API response envelope for detection analysis."""
    success: bool = True
    data: DetectionResult
    message: str = "Audio synthetic voice detection completed successfully"
