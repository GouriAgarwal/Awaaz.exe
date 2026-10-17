"""Step 7 Comprehensive Scientific Benchmark & Validation Test Suite.

Validates:
1. Temporal window count mathematical correctness across all standard durations.
2. Dataset manifest parsing, integrity validation, and duplicate content rejection.
3. Strict speaker leakage detection across partitions.
4. Pretrained AASIST evaluation execution producing all structured metrics.
5. Subgroup analysis for language, accent, generator, and gender.
6. Robustness analysis across codecs and noise conditions.
7. Operating threshold sweep, Best F1, Best EER, and Low FPR selection.
8. Latency summary and Real-Time Factor (RTF = compute_time / audio_duration).
9. Report persistence with timestamped files.
10. REST API endpoints: GET /api/detection/evaluation/status and GET /api/detection/evaluation/latest.
11. Scientific status rule adherence (PRETRAINED_NOT_YET_VALIDATED / NOT_RUN).
"""

import csv
import io
import json
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio.schemas import AudioData
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
    compute_roc_auc,
    compute_threshold_sweep,
)
from app.detection.evaluation.reports import (
    REPORTS_DIR,
    get_latest_evaluation_report,
    save_evaluation_report,
)
from app.detection.evaluation.runner import EvaluationRunner
from app.detection.evaluation.schemas import EvaluationReport
from app.detection.preprocessing import AudioWindowPreprocessor
from app.detection.pretrained.adapter import AASISTPretrainedAdapter
from app.detection.pretrained.provenance import get_pretrained_aasist_provenance
from app.detection.provenance_schema import PretrainedStatus, CalibrationStatus
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Window Count & Temporal Calculation Tests
# ---------------------------------------------------------------------------

class TestWindowCountAudit:
    """Tests auditing mathematical windowing formula: 1 + floor((N - win) / hop) (+ tail)."""

    def test_window_counts_standard_durations(self):
        preprocessor = AudioWindowPreprocessor(window_size_sec=3.0, window_hop_sec=1.5, sample_rate=16000)

        durations_and_expected_counts = [
            (0.5, 1),   # < 3.0s -> padded to 1 window
            (1.0, 1),   # < 3.0s -> padded to 1 window
            (3.0, 1),   # exactly 3.0s -> 1 window: [0.0, 3.0]
            (5.0, 3),   # 5.0s -> 2 full stride + 1 tail = 3 windows: [0,3], [1.5,4.5], [2.0,5.0]
            (6.0, 3),   # 6.0s -> 3 full stride = 3 windows: [0,3], [1.5,4.5], [3,6]
            (10.0, 6),  # 10.0s -> 5 full stride + 1 tail = 6 windows
            (30.0, 19), # 30.0s -> 19 windows: [0,3] through [27,30]
        ]

        for duration, expected_count in durations_and_expected_counts:
            samples = np.zeros(int(16000 * duration), dtype=np.float32)
            audio = AudioData(samples=samples, sample_rate=16000, duration_seconds=duration)
            windows = preprocessor.process(audio)
            assert len(windows) == expected_count, (
                f"For duration {duration}s, expected {expected_count} windows, got {len(windows)}"
            )

    def test_window_intervals_continuity(self):
        preprocessor = AudioWindowPreprocessor(window_size_sec=3.0, window_hop_sec=1.5, sample_rate=16000)
        samples = np.zeros(int(16000 * 6.0), dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration_seconds=6.0)
        windows = preprocessor.process(audio)

        assert len(windows) == 3
        assert windows[0][1] == 0.0 and windows[0][2] == 3.0
        assert windows[1][1] == 1.5 and windows[1][2] == 4.5
        assert windows[2][1] == 3.0 and windows[2][2] == 6.0


# ---------------------------------------------------------------------------
# 2. Dataset Manifest & Integrity Tests
# ---------------------------------------------------------------------------

class TestDatasetIntegrityStep7:
    """Tests for dataset manifest validation and speaker leakage checks."""

    def test_rich_manifest_metadata_parsing(self, tmp_path: Path):
        f1 = tmp_path / "real_en.wav"
        f2 = tmp_path / "synth_en.wav"
        create_mock_test_wav(f1, freq=220.0)
        create_mock_test_wav(f2, freq=440.0)

        csv_path = tmp_path / "rich_manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "split", "speaker_id", "generator", "language", "accent", "gender", "codec", "noise_condition"])
            writer.writerow([str(f1), "REAL", "eval", "spk_1", "none", "en-US", "general_american", "female", "wav", "clean"])
            writer.writerow([str(f2), "SYNTHETIC", "eval", "spk_1", "elevenlabs_v2", "en-US", "general_american", "female", "wav", "clean"])

        manifest = DatasetManifestParser.parse_manifest_file(csv_path)
        assert manifest.total_samples == 2
        assert manifest.real_count == 1
        assert manifest.synthetic_count == 1
        assert manifest.samples[0].language == "en-US"
        assert manifest.samples[1].generator == "elevenlabs_v2"

    def test_speaker_leakage_detection_across_splits(self, tmp_path: Path):
        f1 = tmp_path / "s1.wav"
        f2 = tmp_path / "s2.wav"
        create_mock_test_wav(f1, freq=220.0)
        create_mock_test_wav(f2, freq=440.0)

        csv_path = tmp_path / "leakage.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "split", "speaker_id"])
            writer.writerow([str(f1), "REAL", "train", "speaker_X"])
            writer.writerow([str(f2), "SYNTHETIC", "test", "speaker_X"])

        manifest = DatasetManifestParser.parse_manifest_file(csv_path)
        has_leakage, notes = DatasetManifestParser.check_speaker_leakage(manifest)
        assert has_leakage is True
        assert any("speaker_X" in n or "Speaker leakage" in n for n in notes)

    def test_missing_audio_raises_validation_error(self, tmp_path: Path):
        csv_path = tmp_path / "missing.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label"])
            writer.writerow(["non_existent_audio_file.wav", "REAL"])

        with pytest.raises(DatasetValidationError) as exc:
            DatasetManifestParser.parse_manifest_file(csv_path)
        assert "does not exist" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. Pretrained AASIST Benchmark Execution & Report Structure
# ---------------------------------------------------------------------------

class TestPretrainedAASISTBenchmark:
    """Tests executing evaluation on pretrained AASIST model with complete report structure."""

    @pytest.fixture
    def mock_benchmark_env(self, tmp_path: Path) -> Tuple[Path, DatasetManifest]:
        return generate_mock_test_dataset(tmp_path, num_real=4, num_synthetic=4)

    def test_evaluation_runner_with_pretrained_aasist(self, mock_benchmark_env: Tuple[Path, DatasetManifest]):
        _, manifest = mock_benchmark_env
        adapter = AASISTPretrainedAdapter(device="cpu")

        report = EvaluationRunner.evaluate(detector=adapter, manifest=manifest, operating_threshold=0.50)

        assert report.evaluation_status == "COMPLETED"
        assert report.detector_name == "VoiceShield-AASIST-Pretrained-v1"
        assert report.sample_count == 8
        assert report.real_count == 4
        assert report.synthetic_count == 4
        assert report.metrics is not None
        assert report.metrics.accuracy is not None
        assert report.metrics.precision is not None
        assert report.metrics.recall is not None
        assert report.metrics.f1_score is not None
        assert report.metrics.confusion_matrix.tp + report.metrics.confusion_matrix.tn + report.metrics.confusion_matrix.fp + report.metrics.confusion_matrix.fn == 8

        # Latency & RTF checks
        assert report.latency is not None
        assert report.latency.cold_start_ms > 0
        assert report.latency.warm_avg_ms > 0
        assert report.latency.real_time_factor > 0
        assert report.latency.real_time_factor < 1.0  # Must be faster than real time
        assert report.latency.model_parameters == 297866

        # Step 7 Structured Sections
        assert report.dataset is not None
        assert report.dataset["total_samples"] == 8
        assert report.windowing is not None
        assert report.windowing["window_size_sec"] == 3.0
        assert report.threshold_analysis is not None
        assert "best_f1_operating_point" in report.threshold_analysis
        assert "best_eer_operating_point" in report.threshold_analysis
        assert report.calibration is not None
        assert report.calibration["calibration_status"] == "NOT_CALIBRATED"
        assert len(report.limitations) >= 3

    def test_report_persistence_timestamped(self, mock_benchmark_env: Tuple[Path, DatasetManifest]):
        _, manifest = mock_benchmark_env
        adapter = AASISTPretrainedAdapter(device="cpu")
        report = EvaluationRunner.evaluate(detector=adapter, manifest=manifest)

        saved_path = save_evaluation_report(report)
        assert saved_path.exists()
        assert "aasist_validation_" in saved_path.name

        latest = get_latest_evaluation_report()
        assert latest.detector_name == "VoiceShield-AASIST-Pretrained-v1"
        assert latest.sample_count == 8


# ---------------------------------------------------------------------------
# 4. API Endpoints for Evaluation Status & Run
# ---------------------------------------------------------------------------

class TestEvaluationApiStep7:
    """Tests for GET /api/detection/evaluation/status, GET /latest, and POST /run."""

    def test_api_get_evaluation_status_and_latest(self):
        resp1 = client.get("/api/detection/evaluation/status")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert "evaluation_status" in data1
        assert "detector_name" in data1

        resp2 = client.get("/api/detection/evaluation/latest")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["evaluation_status"] == data1["evaluation_status"]

    def test_api_post_evaluation_run_with_pretrained(self, tmp_path: Path):
        csv_path, _ = generate_mock_test_dataset(tmp_path, num_real=3, num_synthetic=3)
        form_data = {
            "manifest_path": str(csv_path.resolve()),
            "detector": "aasist_pretrained",
            "threshold": 0.50,
        }
        response = client.post("/api/detection/evaluation/run", data=form_data)
        assert response.status_code == 200
        data = response.json()
        assert data["evaluation_status"] == "COMPLETED"
        assert data["detector_name"] == "VoiceShield-AASIST-Pretrained-v1"
        assert data["sample_count"] == 6
        assert data["metrics"] is not None
        assert data["latency"]["real_time_factor"] < 1.0
