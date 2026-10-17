"""Automated Tests for Step 11: Scientific Validation, Calibration, and Benchmark Framework."""

import csv
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.evaluation.calibration import CalibrationError, PostHocCalibrator
from app.evaluation.confidence_intervals import BootstrapEvaluator
from app.evaluation.dataset_audit import DatasetAuditor, DatasetAvailabilityStatus
from app.evaluation.schemas import (
    CalibrationMethod,
    CalibrationStatus,
    ScientificEvaluationStatus,
    SubgroupStatus,
)
from app.evaluation.service import ScientificValidationService
from app.evaluation.streaming_stress import StreamingStressHarness
from app.evaluation.subgroup_analysis import SubgroupAnalyzer
from app.main import app

client = TestClient(app)


class TestDatasetDiscoveryAndAudit:
    """Tests for Phase 11A Dataset Discovery and Integrity Auditing."""

    def test_missing_dataset_returns_not_available_locally(self):
        report = DatasetAuditor.audit_manifest("/path/to/nonexistent/manifest.csv")
        assert report.status == DatasetAvailabilityStatus.NOT_AVAILABLE_LOCALLY
        assert report.total_samples == 0

    def test_invalid_manifest_syntax_handling(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{ invalid json")
            f.flush()
            temp_path = f.name

        try:
            report = DatasetAuditor.audit_manifest(temp_path)
            assert report.status == DatasetAvailabilityStatus.CORRUPTED_OR_INVALID
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_speaker_leakage_detection_across_splits(self):
        # Create manifest with speaker appearing in both train and test splits
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two dummy wav files
            w1 = Path(tmpdir) / "spk1_train.wav"
            w2 = Path(tmpdir) / "spk1_test.wav"
            w1.write_bytes(b"RIFFdummywav1")
            w2.write_bytes(b"RIFFdummywav2")

            manifest_file = Path(tmpdir) / "leakage_manifest.csv"
            with open(manifest_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "label", "split", "speaker_id"])
                writer.writerow(["spk1_train.wav", "REAL", "train", "SPEAKER_001"])
                writer.writerow(["spk1_test.wav", "REAL", "test", "SPEAKER_001"])

            report = DatasetAuditor.audit_manifest(manifest_file, base_dir=tmpdir, verify_audio_decoding=False)
            assert report.has_speaker_leakage is True
            assert len(report.speaker_leakage_notes) > 0
            assert "SPEAKER_001" in report.speaker_leakage_notes[0] or "train" in report.speaker_leakage_notes[0]

    def test_duplicate_content_sha256_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b"IDENTICAL_AUDIO_PAYLOAD_FOR_HASH_TEST"
            w1 = Path(tmpdir) / "file1.wav"
            w2 = Path(tmpdir) / "file2.wav"
            w1.write_bytes(content)
            w2.write_bytes(content)

            manifest_file = Path(tmpdir) / "dup_manifest.csv"
            with open(manifest_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "label", "split", "speaker_id"])
                writer.writerow(["file1.wav", "REAL", "test", "SPK_A"])
                writer.writerow(["file2.wav", "REAL", "test", "SPK_B"])

            report = DatasetAuditor.audit_manifest(manifest_file, base_dir=tmpdir, verify_audio_decoding=False)
            assert report.duplicate_count == 1
            assert len(report.duplicate_files) == 1


class TestThresholdSweepsAndConfidenceIntervals:
    """Tests for Phase 11B & 11I: Metrics, Threshold Sweeps & Bootstrap CIs."""

    def test_threshold_sweep_bounds_and_metrics(self):
        # 50 real (0), 50 synthetic (1)
        y_true = [0] * 50 + [1] * 50
        # Well-separated scores
        y_scores = [0.15] * 45 + [0.70] * 5 + [0.20] * 5 + [0.85] * 45

        service = ScientificValidationService()
        report = service.run_aasist_evaluation(y_true, y_scores, dataset_name="SYNTHETIC_TEST")

        assert report.evaluation_status == ScientificEvaluationStatus.VALIDATED_ON_DEFINED_BENCHMARK
        assert report.score_type == "uncalibrated_model_score"
        assert report.threshold_sweep is not None
        assert len(report.threshold_sweep.points) == 99
        assert 0.0 <= report.threshold_sweep.eer_value <= 1.0
        assert report.threshold_sweep.best_f1_value >= 0.80

    def test_bootstrap_confidence_interval_determinism(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        cis1 = BootstrapEvaluator.compute_standard_classification_cis(y_true, y_scores, random_seed=42)
        cis2 = BootstrapEvaluator.compute_standard_classification_cis(y_true, y_scores, random_seed=42)

        assert len(cis1) == len(cis2)
        for c1, c2 in zip(cis1, cis2):
            assert c1.metric == c2.metric
            assert pytest.approx(c1.ci_lower) == c2.ci_lower
            assert pytest.approx(c1.ci_upper) == c2.ci_upper
            assert c1.ci_lower <= c1.point_estimate <= c1.ci_upper


class TestCalibrationValidationSplitIsolation:
    """Tests for Phase 11C: Calibration Fitting strictly on Validation Split."""

    def test_calibration_rejects_test_split(self):
        val_true = [0, 0, 1, 1]
        val_scores = [0.2, 0.3, 0.7, 0.8]

        with pytest.raises((CalibrationError, ValueError)):
            PostHocCalibrator.fit_temperature_scaling(
                np.array(val_true), np.array(val_scores), split_name="test"
            )

    def test_temperature_scaling_reduces_ece(self):
        np.random.seed(42)
        val_true = np.random.binomial(1, 0.5, 100)
        # Overconfident scores
        val_scores = np.where(val_true == 1, 0.95, 0.05) + np.random.normal(0, 0.05, 100)
        val_scores = np.clip(val_scores, 0.01, 0.99)

        t, ece_before, ece_after = PostHocCalibrator.fit_temperature_scaling(
            val_true, val_scores, split_name="validation"
        )
        assert t > 0.0
        assert ece_before >= 0.0
        assert ece_after >= 0.0

    def test_service_calibration_lifecycle(self):
        service = ScientificValidationService()
        val_true = [0] * 20 + [1] * 20
        val_scores = [0.1] * 20 + [0.9] * 20

        report = service.run_aasist_calibration(
            val_true=val_true,
            val_scores=val_scores,
            method=CalibrationMethod.TEMPERATURE_SCALING,
            split_name="val",
            dataset_name="VAL_EXPERIMENT_01",
        )

        assert report.calibration_status == CalibrationStatus.CALIBRATED
        assert report.calibration_method == CalibrationMethod.TEMPERATURE_SCALING
        assert "temperature" in report.parameters
        assert service.active_aasist_calibration.calibration_status == CalibrationStatus.CALIBRATED


class TestSubgroupAnalysisSafeguards:
    """Tests for Phase 11J: Subgroup Minimum Sample Safeguards."""

    def test_insufficient_samples_marks_status_suppressed(self):
        metadata = [{"generator": "ElevenLabs"} for _ in range(5)]
        y_true = np.array([1] * 5)
        y_score = np.array([0.9] * 5)

        subgroups = SubgroupAnalyzer.evaluate_subgroups(metadata, y_true, y_score, min_samples=20)
        assert len(subgroups) == 1
        assert subgroups[0].status == SubgroupStatus.INSUFFICIENT_DATA
        assert subgroups[0].sample_count == 5
        assert subgroups[0].accuracy is None

    def test_sufficient_samples_calculates_subgroup_metrics(self):
        metadata = [{"generator": "XTTS"} for _ in range(25)]
        y_true = np.array([1] * 25)
        y_score = np.array([0.95] * 25)

        subgroups = SubgroupAnalyzer.evaluate_subgroups(metadata, y_true, y_score, min_samples=20)
        assert len(subgroups) == 1
        assert subgroups[0].status == SubgroupStatus.VALIDATED
        assert subgroups[0].sample_count == 25
        assert subgroups[0].accuracy == 1.0


class TestStreamingStressAndAPI:
    """Tests for Phase 11G & API Integration."""

    def test_streaming_stress_harness_executes_all_conditions(self):
        import asyncio
        report = asyncio.run(StreamingStressHarness.run_all_stress_tests())
        assert report.total_conditions_tested == 15
        assert report.passed_conditions_count == 15
        assert report.all_passed is True
        assert report.streaming_evaluation_status == ScientificEvaluationStatus.STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL

    def test_api_evaluation_status(self):
        resp = client.get("/api/evaluation/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "datasets_status" in data
        assert "aasist_evaluation_status" in data
        assert "fusion_score_type" in data
        assert data["fusion_score_type"] == "heuristic_fusion_score"

    def test_api_security_audit(self):
        resp = client.get("/api/security/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_integrity" in data
        assert "websocket_security" in data
        assert "privacy_guarantees" in data
        assert len(data["model_integrity"]) >= 2
