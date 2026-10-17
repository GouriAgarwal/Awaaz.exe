"""Real-Time Streaming Operational Telemetry and Capacity Monitoring (Step 12)."""

from app.core.config import settings
from app.observability.metrics import MetricsRegistry
from app.observability.schemas import StreamingOperationalInfo
from app.streaming.service import StreamingSessionManager


class StreamingOperationalMonitor:
    """Monitors active WebSocket streaming sessions, queue depth, and backpressure."""

    @classmethod
    def get_streaming_telemetry(cls) -> StreamingOperationalInfo:
        manager = StreamingSessionManager.get_instance()
        metrics = MetricsRegistry()

        active_count = len(manager._sessions)
        max_capacity = settings.MAX_ACTIVE_SESSIONS
        utilization = round((active_count / max_capacity) * 100, 1)

        status_str = "NORMAL"
        if active_count >= max_capacity:
            status_str = "AT_CAPACITY"
        elif active_count >= (max_capacity * 0.8):
            status_str = "HIGH_LOAD"

        return StreamingOperationalInfo(
            active_sessions=active_count,
            max_active_sessions=max_capacity,
            utilization_percentage=utilization,
            total_chunks_ingested=metrics._streaming_chunks_processed,
            total_windows_analyzed=metrics._streaming_windows_analyzed,
            rate_limit_count=metrics._streaming_rate_limits,
            buffer_memory_cap_sec=15.0,
            ephemeral_privacy_policy="RAM_ONLY_NO_DISK_PERSISTENCE",
            status=status_str,
        )
