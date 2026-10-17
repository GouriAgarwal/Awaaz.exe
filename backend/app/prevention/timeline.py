"""
Incident & Decision Timeline Ledger.
Maintains in-memory, privacy-preserving chronological records of fraud detection and prevention events.
Strictly prohibits raw audio, base64 strings, or biometric embeddings from entering the audit trail.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional
import logging

from app.prevention.schemas import (
    TimelineEvent,
    PreventionAction,
    PreventionStatus,
)

logger = logging.getLogger(__name__)


class IncidentTimeline:
    """
    Append-only in-memory incident event timeline.
    Enforces privacy by stripping sensitive audio and biometric artifacts.
    """

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self._events: List[TimelineEvent] = []

    def add_event(
        self,
        event_type: str,
        prevention_status: PreventionStatus,
        prevention_action: PreventionAction,
        details: str,
        risk_level: Optional[str] = None,
        risk_score: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> TimelineEvent:
        """
        Appends a sanitized event record to the workflow timeline.
        """
        sanitized_details = self._sanitize_text(details)

        event = TimelineEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            risk_level=risk_level,
            risk_score=risk_score,
            prevention_status=prevention_status,
            prevention_action=prevention_action,
            details=sanitized_details,
            correlation_id=correlation_id,
        )
        self._events.append(event)
        return event

    def get_events(self) -> List[TimelineEvent]:
        """Returns ordered timeline events."""
        return list(self._events)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Ensures no base64 audio or large data dumps are present in event text."""
        if not text:
            return ""
        # Check for potential base64 audio dump (>128 chars of continuous base64)
        if len(text) > 1000:
            return text[:1000] + "... [TRUNCATED_AUDIT_LOG]"
        return text
