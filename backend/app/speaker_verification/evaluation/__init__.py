"""
Speaker Verification Evaluation Subsystem.
"""
from app.speaker_verification.evaluation.schemas import (
    TrialType,
    VerificationTrial,
    ThresholdOperatingPoint,
    SubgroupVerificationMetrics,
    VerificationMetrics,
    SpeakerVerificationEvaluationReport,
)
from app.speaker_verification.evaluation.metrics import (
    compute_rates_at_threshold,
    compute_roc_auc,
    compute_eer,
    compute_threshold_sweep,
    compute_comprehensive_metrics,
)
from app.speaker_verification.evaluation.evaluator import SpeakerVerificationEvaluator
from app.speaker_verification.evaluation.reports import (
    save_speaker_evaluation_report,
    load_latest_speaker_report,
)

__all__ = [
    "TrialType",
    "VerificationTrial",
    "ThresholdOperatingPoint",
    "SubgroupVerificationMetrics",
    "VerificationMetrics",
    "SpeakerVerificationEvaluationReport",
    "compute_rates_at_threshold",
    "compute_roc_auc",
    "compute_eer",
    "compute_threshold_sweep",
    "compute_comprehensive_metrics",
    "SpeakerVerificationEvaluator",
    "save_speaker_evaluation_report",
    "load_latest_speaker_report",
]
