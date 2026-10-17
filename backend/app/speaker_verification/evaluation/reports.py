"""
Evaluation Reports Storage for Speaker Verification.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from app.speaker_verification.evaluation.schemas import SpeakerVerificationEvaluationReport

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports_data"
LATEST_REPORT_PATH = REPORTS_DIR / "latest_speaker_evaluation_report.json"


def save_speaker_evaluation_report(report: SpeakerVerificationEvaluationReport) -> Path:
    """
    Saves a timestamped speaker evaluation report and updates the latest pointer.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_ts = report.timestamp.replace(":", "-").replace(".", "-")
    filename = f"speaker_evaluation_{clean_ts}.json"
    filepath = REPORTS_DIR / filename

    data = report.model_dump()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved speaker evaluation report to {filepath}")
    return filepath


def load_latest_speaker_report() -> Optional[SpeakerVerificationEvaluationReport]:
    """
    Loads the latest speaker evaluation report if available.
    """
    if not LATEST_REPORT_PATH.exists():
        return None
    try:
        with open(LATEST_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SpeakerVerificationEvaluationReport(**data)
    except Exception as e:
        logger.error(f"Failed to load latest speaker report: {e}")
        return None
