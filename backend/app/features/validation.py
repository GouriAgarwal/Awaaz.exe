"""Numerical sanitization and safety utilities for feature calculations."""

import math
from typing import Any, Optional


def safe_float(val: Any, default: Optional[float] = None, precision: int = 4) -> Optional[float]:
    """
    Sanitize float values, converting NaN or Infinity into None or a safe default.
    Rounds valid floats to specified decimal precision.
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, precision)
    except (ValueError, TypeError):
        return default
