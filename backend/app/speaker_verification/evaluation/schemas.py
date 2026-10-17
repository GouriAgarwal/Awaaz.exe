"""
Schemas for Speaker Verification Scientific Evaluation and Benchmarking.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TrialType(str, Enum):
    TARGET = "target"
    NON_TARGET = "non_target"


class VerificationTrial(BaseModel):
    trial_id: str
    enrollment_audio_path: str
    verification_audio_path: str
    enrollment_speaker_id: str
    verification_speaker_id: str
    trial_type: TrialType
    generator: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = "standard"
    gender: Optional[str] = "unknown"
    codec: Optional[str] = "pcm"
    noise_condition: Optional[str] = "clean"


class ThresholdOperatingPoint(BaseModel):
    threshold: float
    far: float
    frr: float
    tar: float
    trr: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int


class SubgroupVerificationMetrics(BaseModel):
    subgroup_key: str
    subgroup_value: str
    sample_count: int
    status: str = "valid"
    target_trials: int = 0
    non_target_trials: int = 0
    far: Optional[float] = None
    frr: Optional[float] = None
    eer: Optional[float] = None


class VerificationMetrics(BaseModel):
    total_trials: int
    target_trials: int
    non_target_trials: int
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    far: float
    frr: float
    tar: float
    trr: float
    eer: Optional[float] = None
    eer_threshold: Optional[float] = None
    roc_auc: Optional[float] = None
    threshold_sweep: List[ThresholdOperatingPoint] = []
    subgroups: Dict[str, List[SubgroupVerificationMetrics]] = {}


class SpeakerVerificationEvaluationReport(BaseModel):
    evaluation_id: str
    timestamp: str
    model_id: str
    checkpoint_sha256: str
    dataset_name: str
    status: str
    evaluation_metrics: Optional[VerificationMetrics] = None
    speaker_leakage_detected: bool = False
    speaker_leakage_warning: Optional[str] = None
    limitations: List[str] = []
