"""
Schemas for Impersonation Risk Fusion Evaluation and 4-Quadrant Benchmarks.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SpeakerLabel(str, Enum):
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"


class SyntheticLabel(str, Enum):
    NATURAL = "natural"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class AttackLabel(str, Enum):
    LEGITIMATE = "legitimate"
    IMPERSONATION = "impersonation"


class FusionTrial(BaseModel):
    trial_id: str
    audio_path: str
    profile_id: str
    speaker_id: str
    speaker_label: SpeakerLabel
    synthetic_label: SyntheticLabel
    attack_label: AttackLabel
    generator: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = "standard"
    codec: Optional[str] = "pcm"
    noise_condition: Optional[str] = "clean"


class FourQuadrantMatrix(BaseModel):
    authorized_natural: int = Field(0, description="Quadrant 1: Authorized + Natural (Legitimate)")
    authorized_synthetic: int = Field(0, description="Quadrant 2: Authorized + Synthetic (Targeted Clone)")
    unauthorized_natural: int = Field(0, description="Quadrant 3: Unauthorized + Natural (Identity Mismatch)")
    unauthorized_synthetic: int = Field(0, description="Quadrant 4: Unauthorized + Synthetic (Untargeted Spoof)")


class FusionEvaluationMetrics(BaseModel):
    total_trials: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    fpr: float
    fnr: float
    four_quadrant_matrix: FourQuadrantMatrix
    subgroups: Dict[str, Any] = {}


class FusionEvaluationReport(BaseModel):
    evaluation_id: str
    timestamp: str
    status: str = "COMPLETED"
    dataset_name: str
    metrics: Optional[FusionEvaluationMetrics] = None
    speaker_leakage_detected: bool = False
    speaker_leakage_warning: Optional[str] = None
    limitations: List[str] = []
