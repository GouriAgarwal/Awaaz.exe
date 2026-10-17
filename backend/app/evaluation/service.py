"""Unified Scientific Validation and Calibration Service (Step 11)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.logging import get_logger
from app.detection.pretrained.loader import (
    CHECKPOINT_PATH as AASIST_PATH,
    EXPECTED_SHA256 as AASIST_SHA256,
    PretrainedModelLoader,
)
from app.detection.pretrained.adapter import AASISTPretrainedAdapter
from app.evaluation.calibration import PostHocCalibrator
from app.evaluation.confidence_intervals import BootstrapEvaluator
from app.evaluation.dataset_audit import DatasetAuditor
from app.evaluation.schemas import (
    AASISTBenchmarkReport,
    CalibrationMethod,
    CalibrationReport,
    CalibrationStatus,
    DatasetAvailabilityStatus,
    FusionBenchmarkReport,
    MultiModalScenarioResult,
    ScientificEvaluationStatus,
    SecurityAuditReport,
    SpeakerVerificationBenchmarkReport,
    StreamingValidationReport,
    SubgroupMetric,
    SubgroupStatus,
    SystemScientificStatusSummary,
    ThresholdSweepPoint,
    ThresholdSweepReport,
)
from app.evaluation.security_audit import SecurityAuditor
from app.evaluation.streaming_stress import StreamingStressHarness
from app.evaluation.subgroup_analysis import SubgroupAnalyzer
from app.speaker_verification.model_loader import (
    DEFAULT_WEIGHTS_PATH as ECAPA_PATH,
    EXPECTED_SHA256 as ECAPA_SHA256,
    PretrainedSpeakerLoader,
)

logger = get_logger("evaluation.service")


class ScientificValidationService:
    """Singleton coordinator for empirical benchmark evaluation, calibration, and security audits."""

    _instance: Optional["ScientificValidationService"] = None

    def __new__(cls) -> "ScientificValidationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self) -> None:
        self.active_aasist_calibration = CalibrationReport(
            model_name="AASIST Graph Attention Synthetic Voice Detector",
            calibration_status=CalibrationStatus.NOT_CALIBRATED,
            calibration_method=CalibrationMethod.NONE,
            model_checksum=AASIST_SHA256,
            parameters={},
            scientific_notice="AASIST outputs raw softmax scores ('uncalibrated_model_score'). Calibration requires empirical validation data.",
        )
        self.active_speaker_calibration = CalibrationReport(
            model_name="SpeechBrain ECAPA-TDNN Speaker Identity Verifier",
            calibration_status=CalibrationStatus.NOT_CALIBRATED,
            calibration_method=CalibrationMethod.NONE,
            model_checksum=ECAPA_SHA256,
            parameters={"operating_threshold": 0.65, "threshold_status": "PROVISIONAL"},
            scientific_notice="Cosine similarity is an uncalibrated geometric distance, not an identity probability.",
        )

    def get_system_scientific_status(self) -> SystemScientificStatusSummary:
        """Aggregate current empirical validation and calibration statuses across all subsystems."""
        local_manifests = DatasetAuditor.discover_local_datasets()
        datasets_status = (
            DatasetAvailabilityStatus.AVAILABLE_VALIDATED
            if local_manifests
            else DatasetAvailabilityStatus.NOT_AVAILABLE_LOCALLY
        )

        model_audits = SecurityAuditor.audit_model_integrity()
        all_models_valid = all(m.integrity_verified for m in model_audits)

        return SystemScientificStatusSummary(
            timestamp=datetime.now(timezone.utc).isoformat(),
            datasets_status=datasets_status,
            aasist_evaluation_status=ScientificEvaluationStatus.PRETRAINED_NOT_YET_VALIDATED,
            aasist_calibration_status=self.active_aasist_calibration.calibration_status,
            speaker_evaluation_status=ScientificEvaluationStatus.PRETRAINED_NOT_YET_VALIDATED,
            speaker_threshold_status="PROVISIONAL",
            fusion_evaluation_status=ScientificEvaluationStatus.FUSION_IMPLEMENTED_NOT_YET_VALIDATED,
            fusion_score_type="heuristic_fusion_score",
            streaming_evaluation_status=ScientificEvaluationStatus.STREAMING_NOT_VALIDATED,
            security_hardening_status="HARDENED" if all_models_valid else "DEGRADED",
            model_integrity_verified=all_models_valid,
            scientific_disclaimer=(
                "Pretrained weights do not constitute empirical validation. Scores remain uncalibrated operational "
                "heuristics until verified against genuine labeled benchmark datasets."
            ),
        )

    def discover_datasets(self) -> Dict[str, Any]:
        """Search local directory tree for benchmark manifests."""
        found = DatasetAuditor.discover_local_datasets()
        return {
            "found_count": len(found),
            "manifest_paths": [str(p) for p in found],
            "status": DatasetAvailabilityStatus.AVAILABLE_VALIDATED if found else DatasetAvailabilityStatus.NOT_AVAILABLE_LOCALLY,
            "message": "No local benchmark datasets found." if not found else f"Found {len(found)} candidate manifests.",
        }

    def audit_dataset_manifest(self, manifest_path: str, base_dir: Optional[str] = None) -> Any:
        """Run complete integrity audit on a dataset manifest."""
        return DatasetAuditor.audit_manifest(manifest_path, base_dir=base_dir)

    def run_aasist_evaluation(
        self,
        y_true: Optional[List[int]] = None,
        y_scores: Optional[List[float]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        dataset_name: Optional[str] = None,
    ) -> AASISTBenchmarkReport:
        """Run AASIST evaluation with threshold sweep, bootstrap CIs, and subgroup breakdown."""
        if y_true is None or len(y_true) == 0:
            return AASISTBenchmarkReport(
                evaluation_status=ScientificEvaluationStatus.DATASET_NOT_AVAILABLE,
                dataset_name=dataset_name or "NONE",
                score_type="uncalibrated_model_score",
                total_evaluated=0,
                model_checksum=AASIST_SHA256,
                disclaimer="No labeled test samples provided. Evaluation not run.",
            )

        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_scores, dtype=float)
        n = len(yt)

        # Threshold sweep 0.01 to 0.99
        points: List[ThresholdSweepPoint] = []
        best_f1 = -1.0
        best_f1_thresh = 0.50
        min_eer_diff = float("inf")
        eer_thresh = 0.50
        eer_val = 0.50
        low_fpr_thresh = 0.99
        low_fpr_val = 1.0

        for t in np.linspace(0.01, 0.99, 99):
            t_val = round(float(t), 2)
            yp = (ys >= t_val).astype(int)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            fn = int(np.sum((yt == 1) & (yp == 0)))

            acc = (tp + tn) / n if n > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            points.append(
                ThresholdSweepPoint(
                    threshold=t_val,
                    accuracy=round(acc, 4),
                    precision=round(prec, 4),
                    recall=round(rec, 4),
                    specificity=round(spec, 4),
                    fpr=round(fpr, 4),
                    fnr=round(fnr, 4),
                    f1=round(f1, 4),
                )
            )

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = t_val

            if abs(fpr - fnr) < min_eer_diff:
                min_eer_diff = abs(fpr - fnr)
                eer_thresh = t_val
                eer_val = (fpr + fnr) / 2.0

            if fpr <= 0.01 and t_val < low_fpr_thresh:
                low_fpr_thresh = t_val
                low_fpr_val = fpr

        sweep_report = ThresholdSweepReport(
            points=points,
            default_threshold=0.50,
            best_f1_threshold=best_f1_thresh,
            best_f1_value=round(best_f1, 4),
            eer_threshold=eer_thresh,
            eer_value=round(eer_val, 4),
            low_fpr_threshold=low_fpr_thresh,
            low_fpr_value=round(low_fpr_val, 4),
        )

        # Default metrics at 0.50
        y_pred_def = (ys >= 0.50).astype(int)
        tp_def = int(np.sum((yt == 1) & (y_pred_def == 1)))
        tn_def = int(np.sum((yt == 0) & (y_pred_def == 0)))
        fp_def = int(np.sum((yt == 0) & (y_pred_def == 1)))
        fn_def = int(np.sum((yt == 1) & (y_pred_def == 0)))

        acc_def = (tp_def + tn_def) / n
        prec_def = tp_def / (tp_def + fp_def) if (tp_def + fp_def) > 0 else 0.0
        rec_def = tp_def / (tp_def + fn_def) if (tp_def + fn_def) > 0 else 0.0
        spec_def = tn_def / (tn_def + fp_def) if (tn_def + fp_def) > 0 else 0.0
        f1_def = 2 * prec_def * rec_def / (prec_def + rec_def) if (prec_def + rec_def) > 0 else 0.0

        # Bootstrap CIs
        cis = BootstrapEvaluator.compute_standard_classification_cis(yt, ys, threshold=0.50, n_iterations=200)

        # Subgroups
        subgroups = []
        if metadata_list:
            subgroups = SubgroupAnalyzer.evaluate_subgroups(metadata_list, yt, ys, threshold=0.50)

        return AASISTBenchmarkReport(
            evaluation_status=ScientificEvaluationStatus.VALIDATED_ON_DEFINED_BENCHMARK,
            dataset_name=dataset_name or "EMPIRICAL_RUN",
            score_type="uncalibrated_model_score",
            total_evaluated=n,
            confusion_matrix={"tp": tp_def, "tn": tn_def, "fp": fp_def, "fn": fn_def},
            accuracy=round(acc_def, 4),
            precision=round(prec_def, 4),
            recall=round(rec_def, 4),
            specificity=round(spec_def, 4),
            fpr=round(fp_def / (fp_def + tn_def) if (fp_def + tn_def) > 0 else 0.0, 4),
            fnr=round(fn_def / (fn_def + tp_def) if (fn_def + tp_def) > 0 else 0.0, 4),
            f1_score=round(f1_def, 4),
            eer=round(eer_val, 4),
            threshold_sweep=sweep_report,
            confidence_intervals=cis,
            subgroups=subgroups,
            model_checksum=AASIST_SHA256,
        )

    def run_aasist_calibration(
        self,
        val_true: List[int],
        val_scores: List[float],
        method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING,
        split_name: str = "val",
        dataset_name: str = "VALIDATION_SPLIT",
    ) -> CalibrationReport:
        """Fit post-hoc calibration on validation data strictly."""
        if split_name.lower() not in ["val", "validation", "dev"]:
            raise ValueError(f"Calibration must ONLY be fitted on validation split. Got: {split_name}")

        vt = np.asarray(val_true, dtype=int)
        vs = np.asarray(val_scores, dtype=float)

        if method == CalibrationMethod.TEMPERATURE_SCALING:
            best_t, ece_before, ece_after = PostHocCalibrator.fit_temperature_scaling(vt, vs, split_name)
            params = {"temperature": round(best_t, 4)}
        elif method == CalibrationMethod.PLATT_SCALING:
            a, b, ece_before, ece_after = PostHocCalibrator.fit_platt_scaling(vt, vs, split_name)
            params = {"slope_a": round(a, 4), "intercept_b": round(b, 4)}
        else:
            params = {}
            ece_before, ece_after = None, None

        report = CalibrationReport(
            model_name="AASIST Graph Attention Synthetic Voice Detector",
            calibration_status=CalibrationStatus.CALIBRATED,
            calibration_method=method,
            calibration_dataset=dataset_name,
            calibration_split=split_name,
            calibration_timestamp=datetime.now(timezone.utc).isoformat(),
            model_checksum=AASIST_SHA256,
            parameters=params,
            expected_calibration_error_before=round(ece_before, 4) if ece_before is not None else None,
            expected_calibration_error_after=round(ece_after, 4) if ece_after is not None else None,
            scientific_notice="Calibrated probabilities are valid strictly under matching acoustic validation distributions.",
        )

        self.active_aasist_calibration = report
        return report

    async def run_streaming_stress_tests(self) -> StreamingValidationReport:
        """Run all 15 real-time streaming stress conditions."""
        return await StreamingStressHarness.run_all_stress_tests()

    def run_security_audit(self) -> SecurityAuditReport:
        """Run full security hardening and model integrity audit."""
        return SecurityAuditor.audit_system_security()
