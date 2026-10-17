"""Comprehensive unit and integration tests for Step 5 Scientific Validation & Evaluation."""

import csv
import json
from pathlib import Path
from typing import Tuple
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.detection.evaluation.dataset import (
    DatasetManifest,
    DatasetManifestParser,
    DatasetValidationError,
)
from app.detection.evaluation.fixtures import create_mock_test_wav, generate_mock_test_dataset
from app.detection.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_confusion_matrix,
    compute_eer,
    compute_rates_from_counts,
    compute_roc_auc,
    compute_threshold_sweep,
)
from app.detection.evaluation.reports import (
    get_latest_evaluation_report,
    save_evaluation_report,
)
from app.detection.evaluation.runner import EvaluationRunner
from app.detection.evaluation.schemas import EvaluationReport
from app.detection.fallback import FallbackSyntheticDetector
from app.detection.inference import DeepLearningSyntheticDetector
from app.detection.provenance import (
    get_aasist_provenance,
    get_all_provenance_records,
    get_fallback_provenance,
    get_spec_cnn_provenance,
)
from app.detection.provenance_schema import CalibrationStatus, PretrainedStatus


@pytest.fixture
def mock_dataset_env(tmp_path: Path) -> Tuple[Path, DatasetManifest]:
    """Fixture providing a valid mock dataset and audio files."""
    return generate_mock_test_dataset(tmp_path, num_real=3, num_synthetic=3)


# ---------------------------------------------------------------------------
# 1. Dataset Manifest Parsing & Validation Tests
# ---------------------------------------------------------------------------

def test_manifest_csv_parsing(mock_dataset_env: Tuple[Path, DatasetManifest]):
    """Test 1: Manifest CSV parser correctly parses valid dataset with REAL and SYNTHETIC entries."""
    csv_path, manifest = mock_dataset_env
    assert manifest.total_samples == 6
    assert manifest.real_count == 3
    assert manifest.synthetic_count == 3
    assert manifest.speaker_count == 6


def test_manifest_json_parsing(tmp_path: Path):
    """Test 2: Manifest JSON parser correctly parses valid JSON manifest."""
    audio_dir = tmp_path / "json_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    f1 = audio_dir / "audio1.wav"
    f2 = audio_dir / "audio2.wav"
    create_mock_test_wav(f1, duration_sec=1.0, freq=250.0)
    create_mock_test_wav(f2, duration_sec=1.0, freq=450.0)

    json_data = [
        {"file": str(f1.resolve()), "label": "REAL", "split": "test", "speaker_id": "spk1"},
        {"file": str(f2.resolve()), "label": "SYNTHETIC", "split": "test", "speaker_id": "spk2"},
    ]
    json_path = tmp_path / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    manifest = DatasetManifestParser.parse_manifest_file(json_path)
    assert manifest.total_samples == 2
    assert manifest.real_count == 1
    assert manifest.synthetic_count == 1


def test_manifest_invalid_label_raises_error(tmp_path: Path):
    """Test 3: Manifest parser rejects invalid ground-truth labels."""
    f1 = tmp_path / "audio.wav"
    create_mock_test_wav(f1)
    csv_path = tmp_path / "bad_label.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerow([str(f1), "UNKNOWN_LABEL"])

    with pytest.raises(DatasetValidationError) as exc:
        DatasetManifestParser.parse_manifest_file(csv_path)
    assert "Invalid label" in str(exc.value)


def test_manifest_missing_audio_file_raises_error(tmp_path: Path):
    """Test 4: Manifest parser rejects references to nonexistent audio files."""
    csv_path = tmp_path / "missing_file.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerow(["nonexistent_sample.wav", "REAL"])

    with pytest.raises(DatasetValidationError) as exc:
        DatasetManifestParser.parse_manifest_file(csv_path)
    assert "does not exist" in str(exc.value)


def test_manifest_duplicate_file_paths_raises_error(tmp_path: Path):
    """Test 5: Manifest parser detects duplicate file paths."""
    f1 = tmp_path / "unique.wav"
    create_mock_test_wav(f1)
    csv_path = tmp_path / "dup_paths.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerow([str(f1), "REAL"])
        writer.writerow([str(f1), "SYNTHETIC"])

    with pytest.raises(DatasetValidationError) as exc:
        DatasetManifestParser.parse_manifest_file(csv_path)
    assert "Duplicate audio file path" in str(exc.value)


def test_manifest_duplicate_content_hash_raises_error(tmp_path: Path):
    """Test 6: Manifest parser detects duplicate content even with different filenames."""
    f1 = tmp_path / "sample_a.wav"
    f2 = tmp_path / "sample_b.wav"
    # Write identical bytes to both files
    create_mock_test_wav(f1, freq=300.0)
    with open(f1, "rb") as src, open(f2, "wb") as dst:
        dst.write(src.read())

    csv_path = tmp_path / "dup_content.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerow([str(f1), "REAL"])
        writer.writerow([str(f2), "SYNTHETIC"])

    with pytest.raises(DatasetValidationError) as exc:
        DatasetManifestParser.parse_manifest_file(csv_path)
    assert "Duplicate audio content detected" in str(exc.value)


def test_manifest_empty_file_raises_error(tmp_path: Path):
    """Test 7: Empty manifest file raises DatasetValidationError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(DatasetValidationError):
        DatasetManifestParser.parse_manifest_file(empty_csv)


def test_speaker_leakage_detection(tmp_path: Path):
    """Test 8: Speaker leakage across train/test splits is correctly identified."""
    f1 = tmp_path / "s1.wav"
    f2 = tmp_path / "s2.wav"
    create_mock_test_wav(f1, freq=200.0)
    create_mock_test_wav(f2, freq=400.0)

    csv_path = tmp_path / "leakage.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "split", "speaker_id"])
        writer.writerow([str(f1), "REAL", "train", "speaker_101"])
        writer.writerow([str(f2), "SYNTHETIC", "test", "speaker_101"])  # Same speaker in train and test!

    manifest = DatasetManifestParser.parse_manifest_file(csv_path)
    has_leakage, notes = DatasetManifestParser.check_speaker_leakage(manifest)
    assert has_leakage is True
    assert len(notes) > 0
    assert "speaker_101" in notes[0] or "Speaker leakage across splits" in notes[0]


def test_speaker_disjoint_split_passes(tmp_path: Path):
    """Test 9: Disjoint speakers across splits pass without leakage."""
    f1 = tmp_path / "s1.wav"
    f2 = tmp_path / "s2.wav"
    create_mock_test_wav(f1, freq=220.0)
    create_mock_test_wav(f2, freq=440.0)

    csv_path = tmp_path / "disjoint.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "split", "speaker_id"])
        writer.writerow([str(f1), "REAL", "train", "speaker_A"])
        writer.writerow([str(f2), "SYNTHETIC", "test", "speaker_B"])

    manifest = DatasetManifestParser.parse_manifest_file(csv_path)
    has_leakage, _ = DatasetManifestParser.check_speaker_leakage(manifest)
    assert has_leakage is False


# ---------------------------------------------------------------------------
# 2. Mathematical Metrics & Confusion Matrix Tests
# ---------------------------------------------------------------------------

def test_confusion_matrix_perfect():
    """Test 10: Confusion matrix with perfect classification."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.tp == 2
    assert cm.tn == 2
    assert cm.fp == 0
    assert cm.fn == 0


def test_confusion_matrix_mixed():
    """Test 11: Confusion matrix with mixed decisions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])  # 1 FP, 1 FN, 2 TN, 2 TP
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.tp == 2
    assert cm.tn == 2
    assert cm.fp == 1
    assert cm.fn == 1


def test_accuracy_and_f1_rates():
    """Test 12: Rate calculations (Accuracy, Precision, Recall, Specificity, FPR, FNR, F1)."""
    # TP=80, TN=70, FP=30, FN=20 (Total=200)
    acc, prec, rec, spec, fpr, fnr, f1 = compute_rates_from_counts(tp=80, tn=70, fp=30, fn=20)
    assert acc == (80 + 70) / 200  # 0.75
    assert prec == round(80 / (80 + 30), 4)  # 80/110 ≈ 0.7273
    assert rec == round(80 / (80 + 20), 4)   # 80/100 = 0.80
    assert spec == round(70 / (70 + 30), 4)  # 70/100 = 0.70
    assert fpr == round(1.0 - spec, 4)       # 0.30
    assert fnr == round(1.0 - rec, 4)        # 0.20
    assert f1 == round(2 * (80/110) * 0.8 / ((80/110) + 0.8), 4)


def test_rates_zero_division_safety():
    """Test 13: Zero division edge cases return 0.0 safely without exception."""
    acc, prec, rec, spec, fpr, fnr, f1 = compute_rates_from_counts(tp=0, tn=0, fp=0, fn=0)
    assert acc == 0.0
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_roc_auc_perfect_separation():
    """Test 14: ROC-AUC == 1.0 when scores perfectly separate REAL and SYNTHETIC."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = compute_roc_auc(y_true, y_scores)
    assert auc == 1.0


def test_roc_auc_inverted_separation():
    """Test 15: ROC-AUC == 0.0 when scores are completely inverted."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])
    auc = compute_roc_auc(y_true, y_scores)
    assert auc == 0.0


def test_roc_auc_single_class_returns_none():
    """Test 16: ROC-AUC returns None when only one class is present."""
    y_true = np.array([1, 1, 1])
    y_scores = np.array([0.5, 0.6, 0.7])
    auc = compute_roc_auc(y_true, y_scores)
    assert auc is None


def test_eer_perfect_separation():
    """Test 17: EER == 0.0% when scores perfectly separate classes."""
    bonafide = np.array([0.1, 0.2, 0.3])
    spoof = np.array([0.7, 0.8, 0.9])
    eer, thresh = compute_eer(bonafide, spoof)
    assert eer == 0.0
    assert 0.3 <= thresh <= 0.7


def test_eer_overlapping_distributions():
    """Test 18: EER calculation with overlapping score distributions."""
    bonafide = np.array([0.2, 0.4, 0.6, 0.8])
    spoof = np.array([0.2, 0.4, 0.6, 0.8])
    eer, _ = compute_eer(bonafide, spoof)
    assert eer == 50.0


def test_threshold_sweep_monotonicity():
    """Test 19: Threshold sweep correctly evaluates precision/recall across range."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.2, 0.4, 0.6, 0.8])
    sweep = compute_threshold_sweep(y_true, y_scores, thresholds=[0.1, 0.5, 0.9])

    assert len(sweep) == 3
    # At threshold 0.1: everything predicted as SYNTHETIC -> TP=2, FP=2, FN=0, TN=0
    assert sweep[0].tp == 2 and sweep[0].fp == 2
    # At threshold 0.9: nothing predicted as SYNTHETIC -> TP=0, FP=0, FN=2, TN=2
    assert sweep[2].tp == 0 and sweep[2].tn == 2


def test_compute_comprehensive_metrics_structure():
    """Test 20: compute_comprehensive_metrics produces populated summary."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])
    summary = compute_comprehensive_metrics(y_true, y_scores, default_threshold=0.5)

    assert summary.accuracy == 1.0
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.f1_score == 1.0
    assert summary.roc_auc == 1.0
    assert summary.eer == 0.0
    assert len(summary.threshold_sweep) > 10


# ---------------------------------------------------------------------------
# 3. Model Provenance & Audit Tests
# ---------------------------------------------------------------------------

def test_model_provenance_audit_records():
    """Test 21: Model provenance returns valid machine-readable records."""
    records = get_all_provenance_records()
    assert "aasist" in records
    assert "spec_cnn" in records
    assert "fallback" in records

    aasist = records["aasist"]
    assert aasist.detector_name == "VoiceShield-AASIST-v1"
    assert aasist.sample_rate == 16000
    assert aasist.calibration_status == CalibrationStatus.NOT_CALIBRATED


def test_provenance_pretrained_status():
    """Test 22: Unaudited / unweighted neural models are strictly classified as UNTRAINED_NEURAL_BASELINE."""
    aasist = get_aasist_provenance()
    spec = get_spec_cnn_provenance()
    fallback = get_fallback_provenance()

    assert aasist.pretrained_status == PretrainedStatus.UNTRAINED_NEURAL_BASELINE
    assert aasist.pretrained is False
    assert aasist.validated is False

    assert spec.pretrained_status == PretrainedStatus.UNTRAINED_NEURAL_BASELINE
    assert spec.pretrained is False

    assert fallback.pretrained_status == PretrainedStatus.HEURISTIC_FALLBACK
    assert fallback.pretrained is False


# ---------------------------------------------------------------------------
# 4. Evaluation Runner & Reports Tests
# ---------------------------------------------------------------------------

def test_evaluation_runner_on_mock_dataset(mock_dataset_env: Tuple[Path, DatasetManifest]):
    """Test 23: EvaluationRunner evaluates detector on dataset and produces structured EvaluationReport."""
    _, manifest = mock_dataset_env
    detector = DeepLearningSyntheticDetector(model_type="aasist")

    report = EvaluationRunner.evaluate(detector=detector, manifest=manifest, operating_threshold=0.50)

    assert report.sample_count == 6
    assert report.real_count == 3
    assert report.synthetic_count == 3
    assert report.metrics is not None
    assert report.latency is not None
    assert report.latency.cold_start_ms > 0
    assert report.latency.warm_avg_ms > 0
    # Neural model with baseline weights must be marked NOT_VALIDATED
    assert report.evaluation_status == "NOT_VALIDATED"
    assert "UNTRAINED_NEURAL_BASELINE" in report.notes_or_reason


def test_evaluation_runner_fallback_separation(mock_dataset_env: Tuple[Path, DatasetManifest]):
    """Test 24: Heuristic fallback evaluation is isolated and explicitly marked."""
    _, manifest = mock_dataset_env
    fallback = FallbackSyntheticDetector()

    report = EvaluationRunner.evaluate(detector=fallback, manifest=manifest)
    assert report.evaluation_status == "COMPLETED"
    assert "Heuristic Fallback Engine" in report.notes_or_reason


def test_reports_persistence_and_loading(mock_dataset_env: Tuple[Path, DatasetManifest], tmp_path: Path):
    """Test 25: Save and load evaluation reports."""
    _, manifest = mock_dataset_env
    detector = DeepLearningSyntheticDetector(model_type="aasist")
    report = EvaluationRunner.evaluate(detector=detector, manifest=manifest)

    saved_path = save_evaluation_report(report, filename="test_run_report.json")
    assert saved_path.exists()

    latest = get_latest_evaluation_report()
    assert latest.sample_count == 6
    assert latest.detector_name == "VoiceShield-AASIST-v1"


# ---------------------------------------------------------------------------
# 5. API Endpoints Integration Tests
# ---------------------------------------------------------------------------

def test_api_get_evaluation_status(client: TestClient):
    """Test 26: GET /api/detection/evaluation/status returns EvaluationReport."""
    response = client.get("/api/detection/evaluation/status")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_status" in data
    assert "detector_name" in data
    assert "scientific_disclaimer" in data


def test_api_get_model_provenance(client: TestClient):
    """Test 27: GET /api/detection/provenance returns dictionary of model provenance records."""
    response = client.get("/api/detection/provenance")
    assert response.status_code == 200
    data = response.json()
    assert "aasist" in data
    assert "spec_cnn" in data
    assert "fallback" in data
    assert data["aasist"]["pretrained_status"] == "UNTRAINED_NEURAL_BASELINE"


def test_api_post_evaluation_run(client: TestClient, mock_dataset_env: Tuple[Path, DatasetManifest]):
    """Test 28: POST /api/detection/evaluation/run executes benchmark evaluation."""
    csv_path, _ = mock_dataset_env
    form_data = {
        "manifest_path": str(csv_path.resolve()),
        "detector": "aasist",
        "threshold": 0.50,
    }

    response = client.post("/api/detection/evaluation/run", data=form_data)
    assert response.status_code == 200
    report = response.json()
    assert report["sample_count"] == 6
    assert report["metrics"] is not None
    assert report["latency"] is not None
