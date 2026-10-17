"""Unified schemas for Step 11: Production Scientific Validation, Calibration & Security Hardening."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetAvailabilityStatus(str, Enum):
    AVAILABLE_VALIDATED = "AVAILABLE_VALIDATED"
    AVAILABLE_WITH_WARNINGS = "AVAILABLE_WITH_WARNINGS"
    NOT_AVAILABLE_LOCALLY = "NOT_AVAILABLE_LOCALLY"
    CORRUPTED_OR_INVALID = "CORRUPTED_OR_INVALID"


class CalibrationStatus(str, Enum):
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATED = "CALIBRATED"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"


class CalibrationMethod(str, Enum):
    NONE = "NONE"
    TEMPERATURE_SCALING = "TEMPERATURE_SCALING"
    PLATT_SCALING = "PLATT_SCALING"
    ISOTONIC_REGRESSION = "ISOTONIC_REGRESSION"


class ScientificEvaluationStatus(str, Enum):
    DATASET_NOT_AVAILABLE = "DATASET_NOT_AVAILABLE"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"
    PRETRAINED_NOT_YET_VALIDATED = "PRETRAINED_NOT_YET_VALIDATED"
    VALIDATED_ON_DEFINED_BENCHMARK = "VALIDATED_ON_DEFINED_BENCHMARK"
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATED_ON_DEFINED_VALIDATION_DISTRIBUTION = "CALIBRATED_ON_DEFINED_VALIDATION_DISTRIBUTION"
    FUSION_IMPLEMENTED_NOT_YET_VALIDATED = "FUSION_IMPLEMENTED_NOT_YET_VALIDATED"
    FUSION_VALIDATED_ON_DEFINED_DATASET = "FUSION_VALIDATED_ON_DEFINED_DATASET"
    STREAMING_NOT_VALIDATED = "STREAMING_NOT_VALIDATED"
    STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL = "STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SubgroupStatus(str, Enum):
    VALIDATED = "VALIDATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ConfidenceInterval(BaseModel):
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    method: str = "bootstrap"
    bootstrap_iterations: int = 1000


class ThresholdSweepPoint(BaseModel):
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    fpr: float
    fnr: float
    f1: float


class ThresholdSweepReport(BaseModel):
    points: List[ThresholdSweepPoint]
    default_threshold: float = 0.50
    best_f1_threshold: float
    best_f1_value: float
    eer_threshold: float
    eer_value: float
    low_fpr_threshold: float  # threshold achieving FPR <= 1% (or closest minimum)
    low_fpr_value: float


class SubgroupMetric(BaseModel):
    subgroup_dimension: str  # e.g., "generator", "language", "codec", "noise"
    subgroup_value: str
    sample_count: int
    status: SubgroupStatus
    accuracy: Optional[float] = None
    eer: Optional[float] = None
    f1: Optional[float] = None
    note: Optional[str] = None


class DatasetAuditReport(BaseModel):
    dataset_name: str
    status: DatasetAvailabilityStatus
    manifest_path: Optional[str] = None
    total_samples: int = 0
    real_count: int = 0
    synthetic_count: int = 0
    split_counts: Dict[str, int] = Field(default_factory=dict)
    speaker_count: int = 0
    generator_counts: Dict[str, int] = Field(default_factory=dict)
    has_speaker_leakage: bool = False
    speaker_leakage_notes: List[str] = Field(default_factory=list)
    has_generator_leakage: bool = False
    generator_leakage_notes: List[str] = Field(default_factory=list)
    duplicate_count: int = 0
    duplicate_files: List[str] = Field(default_factory=list)
    missing_files: List[str] = Field(default_factory=list)
    integrity_notes: List[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    model_name: str
    calibration_status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED
    calibration_method: CalibrationMethod = CalibrationMethod.NONE
    calibration_dataset: Optional[str] = None
    calibration_split: Optional[str] = None  # MUST be validation split only
    calibration_timestamp: Optional[str] = None
    model_checksum: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_calibration_error_before: Optional[float] = None
    expected_calibration_error_after: Optional[float] = None
    scientific_notice: str = "Raw scores remain available. Calibrated probabilities apply only to defined validation distribution."


class AASISTBenchmarkReport(BaseModel):
    evaluation_status: ScientificEvaluationStatus = ScientificEvaluationStatus.DATASET_NOT_AVAILABLE
    dataset_name: Optional[str] = None
    score_type: str = "uncalibrated_model_score"
    total_evaluated: int = 0
    confusion_matrix: Optional[Dict[str, int]] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    eer: Optional[float] = None
    threshold_sweep: Optional[ThresholdSweepReport] = None
    confidence_intervals: List[ConfidenceInterval] = Field(default_factory=list)
    subgroups: List[SubgroupMetric] = Field(default_factory=list)
    model_checksum: str
    disclaimer: str = "AASIST output is an uncalibrated model score unless explicit post-hoc calibration is fitted on a verified validation set."


class SpeakerVerificationBenchmarkReport(BaseModel):
    evaluation_status: ScientificEvaluationStatus = ScientificEvaluationStatus.DATASET_NOT_AVAILABLE
    dataset_name: Optional[str] = None
    total_trials: int = 0
    genuine_trials: int = 0
    impostor_trials: int = 0
    provisional_threshold: float = 0.65
    threshold_status: str = "PROVISIONAL"
    far: Optional[float] = None
    frr: Optional[float] = None
    eer: Optional[float] = None
    roc_auc: Optional[float] = None
    genuine_mean_similarity: Optional[float] = None
    impostor_mean_similarity: Optional[float] = None
    threshold_sweep: Optional[List[Dict[str, float]]] = None
    confidence_intervals: List[ConfidenceInterval] = Field(default_factory=list)
    subgroups: List[SubgroupMetric] = Field(default_factory=list)
    model_checksum: str
    semantic_notice: str = "Cosine similarity score != probability of speaker identity."


class MultiModalScenarioResult(BaseModel):
    scenario_id: str  # A, B, C, D, E, F, G, H
    scenario_name: str
    description: str
    sample_count: int
    avg_synthetic_score: Optional[float] = None
    avg_speaker_similarity: Optional[float] = None
    avg_fusion_risk: Optional[float] = None
    dominant_risk_level: Optional[str] = None
    dominant_action: Optional[str] = None
    correct_operational_rate: Optional[float] = None


class FusionBenchmarkReport(BaseModel):
    evaluation_status: ScientificEvaluationStatus = ScientificEvaluationStatus.FUSION_IMPLEMENTED_NOT_YET_VALIDATED
    risk_score_type: str = "heuristic_fusion_score"
    total_scenarios_evaluated: int = 0
    scenarios: List[MultiModalScenarioResult] = Field(default_factory=list)
    clone_attack_detection_rate: Optional[float] = None
    false_block_rate_genuine: Optional[float] = None
    false_escalation_rate: Optional[float] = None
    natural_impostor_handling_rate: Optional[float] = None
    synthetic_non_target_handling_rate: Optional[float] = None
    fusion_rule_matrix_version: str = "1.0.0"
    calibration_notice: str = "Fusion risk score (0-100) is a bounded heuristic aggregation, not an empirical probability."


class StreamingStressTestCondition(BaseModel):
    test_id: str
    condition_name: str
    description: str
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class StreamingValidationReport(BaseModel):
    streaming_evaluation_status: ScientificEvaluationStatus = ScientificEvaluationStatus.STREAMING_NOT_VALIDATED
    total_conditions_tested: int = 0
    passed_conditions_count: int = 0
    all_passed: bool = False
    conditions: List[StreamingStressTestCondition] = Field(default_factory=list)
    avg_processing_lag_ms: float = 0.0
    max_memory_allocated_mb: float = 0.0
    ephemeral_privacy_verified: bool = True
    session_stability_score: float = 0.0
    timestamp: str


class ModelIntegrityAudit(BaseModel):
    model_name: str
    weights_path: str
    expected_sha256: str
    actual_sha256: str
    integrity_verified: bool
    parameter_count: int
    device: str
    status: str


class SecurityAuditReport(BaseModel):
    timestamp: str
    model_integrity: List[ModelIntegrityAudit] = Field(default_factory=list)
    websocket_security: Dict[str, Any] = Field(default_factory=dict)
    api_security: Dict[str, Any] = Field(default_factory=dict)
    privacy_guarantees: Dict[str, Any] = Field(default_factory=dict)
    overall_hardened: bool = True


class SystemScientificStatusSummary(BaseModel):
    timestamp: str
    datasets_status: DatasetAvailabilityStatus
    aasist_evaluation_status: ScientificEvaluationStatus
    aasist_calibration_status: CalibrationStatus
    speaker_evaluation_status: ScientificEvaluationStatus
    speaker_threshold_status: str
    fusion_evaluation_status: ScientificEvaluationStatus
    fusion_score_type: str
    streaming_evaluation_status: ScientificEvaluationStatus
    security_hardening_status: str
    model_integrity_verified: bool
    scientific_disclaimer: str
