"""Machine-readable Model Provenance Record definitions for synthetic voice detectors."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PretrainedStatus(str, Enum):
    """Scientific verification status of detector model weights."""
    UNTRAINED_NEURAL_BASELINE = "UNTRAINED_NEURAL_BASELINE"
    PRETRAINED_NOT_YET_VALIDATED = "PRETRAINED_NOT_YET_VALIDATED"
    VALIDATED_PRETRAINED = "VALIDATED_PRETRAINED"
    HEURISTIC_FALLBACK = "HEURISTIC_FALLBACK"


class CalibrationStatus(str, Enum):
    """Scientific score calibration status."""
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATED_ISOTONIC = "CALIBRATED_ISOTONIC"
    CALIBRATED_PLATT = "CALIBRATED_PLATT"


class ModelProvenanceRecord(BaseModel):
    """Standardized machine-readable provenance metadata for voice clone detectors."""

    detector_name: str = Field(..., description="Unique detector identifier")
    architecture: str = Field(..., description="Underlying model architecture")
    checkpoint_identifier: str = Field(..., description="Model checkpoint identifier or 'none'")
    source_repository: str = Field(default="unknown", description="Source code or weights repository")
    source_url: str = Field(default="unknown", description="URL to model repository or paper")
    revision: str = Field(default="unknown", description="Git commit hash or checkpoint version")
    license: str = Field(..., description="Software/weights license")
    training_dataset: str = Field(default="unknown", description="Dataset on which weights were trained")
    intended_task: str = Field(default="Synthetic Voice Detection", description="Target classification domain")
    sample_rate: int = Field(default=16000, description="Expected input audio sample rate in Hz")
    input_format: str = Field(default="16 kHz mono float32 PCM", description="Standardized audio tensor format")
    output_classes: List[str] = Field(
        default_factory=lambda: ["NATURAL", "SYNTHETIC"],
        description="Binary or multi-class target labels"
    )
    score_semantics: str = Field(..., description="Interpretation of continuous raw score output")
    pretrained: bool = Field(default=False, description="Whether genuine pretrained weights are loaded")
    validated: bool = Field(default=False, description="Whether scientifically validated against benchmark datasets")
    pretrained_status: PretrainedStatus = Field(..., description="Pretrained classification verdict")
    calibration_status: CalibrationStatus = Field(default=CalibrationStatus.NOT_CALIBRATED, description="Calibration state")
    notes: str = Field(default="", description="Audit notes and scientific limitations")
