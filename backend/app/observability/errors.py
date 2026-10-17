"""Unified Operational Error Taxonomy and Safe Failure Handling (Step 12)."""

from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Any, Dict, Optional
from fastapi import HTTPException, status

from pydantic import BaseModel

from app.observability.schemas import OperationalErrorResponse


class ErrorCode(str, Enum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_INTEGRITY_FAILURE = "MODEL_INTEGRITY_FAILURE"
    AUDIO_INVALID = "AUDIO_INVALID"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    STREAM_RATE_LIMITED = "STREAM_RATE_LIMITED"
    STREAM_MESSAGE_TOO_LARGE = "STREAM_MESSAGE_TOO_LARGE"
    SESSION_LIMIT_REACHED = "SESSION_LIMIT_REACHED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INTERNAL_PROCESSING_ERROR = "INTERNAL_PROCESSING_ERROR"


class OperationalException(HTTPException):
    """Structured exception mapped to the VoiceShield operational error taxonomy."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        request_id: Optional[str] = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        # Record error occurrence in telemetry
        ErrorTracker.record_error(error_code.value)

    def to_response(self) -> OperationalErrorResponse:
        return OperationalErrorResponse(
            error_code=self.error_code.value,
            message=self.message,
            request_id=self.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class ErrorTracker:
    """Thread-safe error recorder aggregating error frequencies by taxonomy code."""

    _lock = threading.Lock()
    _counts: Dict[str, int] = {e.value: 0 for e in ErrorCode}

    @classmethod
    def record_error(
        cls,
        error_code: Any,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        code_str = error_code.value if hasattr(error_code, "value") else str(error_code)
        with cls._lock:
            cls._counts[code_str] = cls._counts.get(code_str, 0) + 1

    @classmethod
    def get_error_counts(cls) -> Dict[str, int]:
        with cls._lock:
            return dict(cls._counts)

    @classmethod
    def get_total_errors(cls) -> int:
        with cls._lock:
            return sum(cls._counts.values())

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            for k in cls._counts:
                cls._counts[k] = 0


# Singleton alias for convenience
error_tracker = ErrorTracker

