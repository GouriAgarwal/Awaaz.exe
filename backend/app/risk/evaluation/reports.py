"""
Reports Persistence for Impersonation Risk Evaluation.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from app.risk.evaluation.schemas import FusionEvaluationReport

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports_data"
LATEST_REPORT_PATH = REPORTS_DIR / "latest_fusion_evaluation_report.json"


def save_fusion_evaluation_report(report: FusionEvaluationReport) -> Path:
    """
    Saves a timestamped fusion evaluation report and updates the latest report pointer.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_ts = report.timestamp.replace(":", "-").replace(".", "-")
    filename = f"fusion_evaluation_{clean_ts}.json"
    filepath = REPORTS_DIR / filename

    data = report.model_dump()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved fusion evaluation report to {filepath}")
    return filepath


def load_latest_fusion_report() -> Optional[FusionEvaluationReport]:
    """
    Loads the latest fusion evaluation report.
    """
    if not LATEST_REPORT_PATH.exists():
        return None
    try:
        with open(LATEST_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FusionEvaluationReport(**data)
    except Exception as e:
        logger.error(f"Failed to load latest fusion report: {e}")
        return None
