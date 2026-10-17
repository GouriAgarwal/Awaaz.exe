"""Persistence, retrieval, and serialization of scientific evaluation reports."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.detection.evaluation.schemas import EvaluationReport
from app.detection.pretrained.provenance import get_pretrained_aasist_provenance

logger = get_logger("detection.evaluation.reports")

REPORTS_DIR = Path(os.path.dirname(__file__)) / "reports_data"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_REPORT_FILE = REPORTS_DIR / "latest_evaluation_report.json"


def save_evaluation_report(report: EvaluationReport, filename: Optional[str] = None) -> Path:
    """Save an EvaluationReport to disk as formatted JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_slug = "aasist" if "aasist" in report.detector_name.lower() else "detector"
        filename = f"{model_slug}_validation_{ts_str}.json"

    target_file = REPORTS_DIR / filename

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Also update latest pointer file
    with open(LATEST_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    logger.info(f"Saved evaluation report to {target_file} (and updated latest_evaluation_report.json)")
    return target_file


def get_latest_evaluation_report() -> EvaluationReport:
    """Retrieve latest saved evaluation report, or return a default NOT_RUN report if none exists."""
    if LATEST_REPORT_FILE.exists():
        try:
            with open(LATEST_REPORT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationReport.model_validate(data)
        except Exception as e:
            logger.warning(f"Could not load saved evaluation report: {e}")

    # Return clean NOT_RUN state
    return EvaluationReport(
        evaluation_status="NOT_RUN",
        detector_name="VoiceShield-AASIST-Pretrained-v1",
        model_id="VoiceShield-AASIST-Pretrained-v1",
        checkpoint_sha256="51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0",
        provenance=get_pretrained_aasist_provenance(),
        dataset_name="None (Pending Standardized Benchmark)",
        sample_count=0,
        real_count=0,
        synthetic_count=0,
        metrics=None,
        latency=None,
        calibration_status="NOT_CALIBRATED",
        speaker_leakage_detected=False,
        environment_info={},
        scientific_disclaimer="Independent scientific evaluation has not yet been executed on an external benchmark dataset.",
        notes_or_reason="No external benchmark dataset manifest has been executed. Status remains NOT_RUN.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
