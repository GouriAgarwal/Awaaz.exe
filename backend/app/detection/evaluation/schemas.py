"""Pydantic schemas for scientific validation and dataset evaluation (Step 5)."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.detection.provenance_schema import ModelProvenanceRecord


class DatasetSample(BaseModel):
    """Metadata record for an individual audio sample in an evaluation dataset."""
    file: str = Field(..., description="Relative or absolute path to audio file")
    label: str = Field(..., description="Ground truth label: 'REAL' or 'SYNTHETIC'")
    split: Optional[str] = Field(default="test", description="Dataset partition ('train', 'dev', 'test', 'eval')")
    speaker_id: Optional[str] = Field(default=None, description="Unique speaker identity if known")
    generator: Optional[str] = Field(default=None, description="AI voice synthesis generator/model if known")
    language: Optional[str] = Field(default=None, description="Spoken language ISO code or label")
    accent: Optional[str] = Field(default=None, description="Speaker accent description")
    gender: Optional[str] = Field(default=None, description="Speaker gender if documented")
    codec: Optional[str] = Field(default=None, description="Original audio container/codec (e.g. 'wav', 'mp3', 'opus')")
    noise_condition: Optional[str] = Field(default=None, description="Acoustic environment (e.g. 'clean', 'office_noise')")


class DatasetManifest(BaseModel):
    """Validated dataset manifest container."""
    dataset_name: str = Field(..., description="Unique dataset identifier or benchmark name")
    samples: List[DatasetSample] = Field(default_factory=list, description="List of validated dataset audio entries")
    total_samples: int = Field(default=0, description="Total number of valid samples")
    real_count: int = Field(default=0, description="Number of REAL (bonafide) human speech samples")
    synthetic_count: int = Field(default=0, description="Number of SYNTHETIC (spoofed/cloned) samples")
    split_counts: Dict[str, int] = Field(default_factory=dict, description="Counts per split partition")
    speaker_count: int = Field(default=0, description="Number of distinct speaker IDs identified")
    generator_counts: Dict[str, int] = Field(default_factory=dict, description="Counts per synthetic generator")


class ConfusionMatrix(BaseModel):
    """Confusion matrix counts for binary voice detection classification."""
    tp: int = Field(..., description="True Positives (SYNTHETIC correctly classified as SYNTHETIC)")
    tn: int = Field(..., description="True Negatives (REAL correctly classified as REAL)")
    fp: int = Field(..., description="False Positives / False Alarms (REAL erroneously flagged as SYNTHETIC)")
    fn: int = Field(..., description="False Negatives / Misses (SYNTHETIC erroneously classified as REAL)")


class ThresholdPoint(BaseModel):
    """Performance evaluation at an explicit classification threshold boundary."""
    threshold: float = Field(..., description="Decision threshold boundary in [0.0, 1.0]")
    tp: int = Field(..., description="True Positives")
    tn: int = Field(..., description="True Negatives")
    fp: int = Field(..., description="False Positives")
    fn: int = Field(..., description="False Negatives")
    precision: float = Field(..., description="Precision at this threshold")
    recall: float = Field(..., description="Recall / True Positive Rate at this threshold")
    specificity: float = Field(..., description="Specificity / True Negative Rate at this threshold")
    fpr: float = Field(..., description="False Positive Rate at this threshold")
    fnr: float = Field(..., description="False Negative Rate at this threshold")
    f1: float = Field(..., description="F1-Score at this threshold")


class EvaluationMetricsSummary(BaseModel):
    """Comprehensive mathematical evaluation metrics."""
    accuracy: float = Field(..., description="Overall classification accuracy in [0.0, 1.0]")
    precision: float = Field(..., description="Precision in [0.0, 1.0]")
    recall: float = Field(..., description="Recall / Sensitivity in [0.0, 1.0]")
    specificity: float = Field(..., description="Specificity / Selectivity in [0.0, 1.0]")
    f1_score: float = Field(..., description="F1-Score in [0.0, 1.0]")
    fpr: float = Field(..., description="False Positive Rate in [0.0, 1.0]")
    fnr: float = Field(..., description="False Negative Rate in [0.0, 1.0]")
    roc_auc: Optional[float] = Field(default=None, description="Area Under the Receiver Operating Characteristic Curve")
    eer: Optional[float] = Field(default=None, description="Equal Error Rate in % (where FAR == FRR)")
    eer_threshold: Optional[float] = Field(default=None, description="Operating threshold corresponding to EER")
    confusion_matrix: ConfusionMatrix = Field(..., description="Confusion matrix at default operating threshold")
    threshold_sweep: List[ThresholdPoint] = Field(
        default_factory=list,
        description="Comprehensive threshold sweep points across [0.0, 1.0]"
    )


class LatencySummary(BaseModel):
    """Inference execution latency and real-time efficiency benchmarks."""
    cold_start_ms: float = Field(..., description="First cold model initialization and forward pass time (ms)")
    warm_avg_ms: float = Field(..., description="Average warm sample inference latency (ms)")
    p50_ms: float = Field(..., description="50th percentile (median) inference latency (ms)")
    p95_ms: float = Field(..., description="95th percentile inference latency (ms)")
    total_audio_duration_sec: float = Field(default=0.0, description="Total duration of all evaluated audio in seconds")
    total_compute_sec: float = Field(default=0.0, description="Total CPU/GPU execution time in seconds")
    real_time_factor: float = Field(default=0.0, description="Real-Time Factor (RTF = compute_time / audio_duration, <1.0 is faster than real-time)")
    speedup_factor: float = Field(default=0.0, description="Speedup factor (audio_duration / compute_time)")
    device: str = Field(default="cpu", description="Hardware compute device used during benchmark")
    model_parameters: int = Field(default=297866, description="Model parameter count")


class EvaluationReport(BaseModel):
    """Complete scientific evaluation report (Step 7)."""
    evaluation_status: str = Field(
        default="NOT_RUN",
        description="Evaluation status: 'NOT_RUN', 'COMPLETED', 'NOT_VALIDATED', 'BLOCKED', or 'ERROR'"
    )
    detector_name: str = Field(..., description="Identifier of evaluated detector")
    model_id: Optional[str] = Field(default=None, description="Canonical model ID")
    checkpoint_sha256: Optional[str] = Field(default=None, description="Cryptographic SHA-256 hash of checkpoint")
    provenance: ModelProvenanceRecord = Field(..., description="Machine-readable provenance of the evaluated model")
    dataset_name: str = Field(..., description="Evaluation benchmark dataset name")
    sample_count: int = Field(default=0, description="Total evaluated samples")
    real_count: int = Field(default=0, description="Number of REAL speech samples evaluated")
    synthetic_count: int = Field(default=0, description="Number of SYNTHETIC speech samples evaluated")
    dataset: Optional[Dict] = Field(default=None, description="Summary statistics of the evaluation dataset")
    windowing: Optional[Dict] = Field(default=None, description="Temporal windowing configuration and sample counts")
    metrics: Optional[EvaluationMetricsSummary] = Field(default=None, description="Populated evaluation metrics")
    threshold_analysis: Optional[Dict] = Field(default=None, description="Identified optimal threshold operating points")
    calibration: Optional[Dict] = Field(default=None, description="Calibration analysis, score semantics, and reliability")
    subgroups: Optional[Dict] = Field(default=None, description="Subgroup metrics across language, accent, generator, etc.")
    robustness: Optional[Dict] = Field(default=None, description="Performance across codecs and acoustic noise conditions")
    speaker_leakage: Optional[Dict] = Field(default=None, description="Speaker partition analysis and leakage assessment")
    speaker_leakage_detected: bool = Field(default=False, description="Whether speaker identity leakage was detected")
    latency: Optional[LatencySummary] = Field(default=None, description="Measured latency and RTF statistics")
    calibration_status: str = Field(default="NOT_CALIBRATED", description="Calibration state of the evaluated model")
    reproducibility: Optional[Dict] = Field(default=None, description="Deterministic seed and environment configuration")
    environment_info: Dict[str, str] = Field(default_factory=dict, description="Hardware and software runtime versions")
    limitations: List[str] = Field(default_factory=list, description="Scientific limitations and known distribution shifts")
    scientific_disclaimer: str = Field(..., description="Scientific limitations notice")
    notes_or_reason: str = Field(default="", description="Detailed audit or evaluation notes")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")


class EvaluationMetrics(BaseModel):
    """Backwards-compatible scientific evaluation benchmark metrics."""
    evaluation_status: str = Field(default="NOT_RUN")
    dataset_name: Optional[str] = Field(default=None)
    equal_error_rate: Optional[float] = Field(default=None)
    min_dcf: Optional[float] = Field(default=None)
    accuracy: Optional[float] = Field(default=None)
    f1_score: Optional[float] = Field(default=None)
    roc_auc: Optional[float] = Field(default=None)
    threshold_at_eer: Optional[float] = Field(default=None)
    total_eval_samples: int = Field(default=0)
    disclaimer: str = Field(default="Standard empirical evaluation.")
