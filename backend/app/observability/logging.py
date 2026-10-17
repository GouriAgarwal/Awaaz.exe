"""Structured JSON Logging with Cryptographic and Biometric Privacy Sanitization (Step 12)."""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, Optional
import numpy as np


class SensitiveDataSanitizer:
    """Rigorous privacy filter stripping raw audio, embeddings, and credentials from log payloads."""

    REDACTED_TEXT = "[REDACTED_SENSITIVE_DATA]"
    BASE64_AUDIO_PATTERN = re.compile(r"^[A-Za-z0-9+/=]{128,}$")

    @classmethod
    def sanitize(cls, data: Any) -> Any:
        """Recursively redact audio waveforms, speaker embeddings, base64 audio, and secrets."""
        if isinstance(data, dict):
            sanitized: Dict[str, Any] = {}
            for k, v in data.items():
                k_lower = k.lower()
                # Check sensitive key names
                if any(term in k_lower for term in ["raw_audio", "embedding", "tensors", "secret", "password", "token", "auth", "key", "credential"]):
                    sanitized[k] = cls.REDACTED_TEXT

                elif isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 128 and all(isinstance(x, (int, float, np.floating)) for x in v[:10]):
                    # Appears to be an embedding vector or audio waveform array
                    sanitized[k] = f"[REDACTED_VECTOR_LEN_{len(v)}]"
                else:
                    sanitized[k] = cls.sanitize(v)
            return sanitized

        elif isinstance(data, (list, tuple)):
            if len(data) >= 128 and all(isinstance(x, (int, float, np.floating)) for x in data[:10]):
                return f"[REDACTED_VECTOR_LEN_{len(data)}]"
            return [cls.sanitize(item) for item in data]

        elif isinstance(data, np.ndarray):
            return f"[REDACTED_NUMPY_ARRAY_SHAPE_{data.shape}]"

        elif isinstance(data, str):
            # Check for large base64 audio blocks
            if len(data) > 128 and cls.BASE64_AUDIO_PATTERN.match(data.strip()):
                return "[REDACTED_BASE64_AUDIO]"
            return data

        return data


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON with correlation IDs and privacy redaction."""

    # Built-in LogRecord attributes to ignore when extracting extra context
    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Correlation tracking and any custom extra kwargs passed to logger
        for key, val in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = val

        # Sanitize entire log object before serialization
        clean_entry = SensitiveDataSanitizer.sanitize(log_entry)
        return json.dumps(clean_entry)



def get_structured_logger(name: str) -> logging.Logger:
    """Return configured structured logger instance."""
    logger = logging.getLogger(name)
    return logger
