"""Process and Host Resource Monitoring (Step 12)."""

import os
import time
from typing import Any, Dict
from app.core.config import settings


class ResourceMonitor:
    """Monitors process resource utilization without fabricating metrics."""

    _process_start_time = time.time()

    @classmethod
    def get_resource_summary(cls) -> Dict[str, Any]:
        """Aggregate process uptime, memory usage, and capacity statistics."""
        uptime = round(time.time() - cls._process_start_time, 2)

        # Query streaming manager for active session capacity
        from app.streaming.service import StreamingSessionManager
        manager = StreamingSessionManager.get_instance()
        active_count = len(manager._sessions)
        max_capacity = settings.MAX_ACTIVE_SESSIONS

        summary: Dict[str, Any] = {
            "uptime_seconds": uptime,
            "active_sessions": active_count,
            "max_active_sessions": max_capacity,
            "session_capacity_utilization_pct": round((active_count / max_capacity) * 100, 1),
            "pid": os.getpid(),
        }

        # Attempt safe memory inquiry via psutil if available
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            summary["memory_rss_mb"] = round(mem_info.rss / (1024 * 1024), 2)
            summary["cpu_percent"] = proc.cpu_percent(interval=None)
            summary["resource_telemetry_status"] = "AVAILABLE"
        except Exception:
            summary["memory_rss_mb"] = None
            summary["cpu_percent"] = None
            summary["resource_telemetry_status"] = "UNAVAILABLE"

        return summary
