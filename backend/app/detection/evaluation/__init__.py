"""Scientific evaluation and validation package for voice clone detectors (Step 5)."""

from app.detection.evaluation.dataset import DatasetManifest, DatasetManifestParser, DatasetValidationError
from app.detection.evaluation.fixtures import generate_mock_test_dataset
from app.detection.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_confusion_matrix,
    compute_eer,
    compute_rates_from_counts,
    compute_roc_auc,
    compute_threshold_sweep,
)
from app.detection.evaluation.reports import get_latest_evaluation_report, save_evaluation_report
from app.detection.evaluation.runner import EvaluationRunner
from app.detection.evaluation.schemas import (
    ConfusionMatrix,
    DatasetSample,
    EvaluationMetricsSummary,
    EvaluationReport,
    LatencySummary,
    ThresholdPoint,
)

__all__ = [
    "DatasetManifest",
    "DatasetSample",
    "DatasetManifestParser",
    "DatasetValidationError",
    "EvaluationRunner",
    "EvaluationReport",
    "EvaluationMetricsSummary",
    "ConfusionMatrix",
    "ThresholdPoint",
    "LatencySummary",
    "compute_comprehensive_metrics",
    "compute_confusion_matrix",
    "compute_eer",
    "compute_rates_from_counts",
    "compute_roc_auc",
    "compute_threshold_sweep",
    "get_latest_evaluation_report",
    "save_evaluation_report",
    "generate_mock_test_dataset",
]
