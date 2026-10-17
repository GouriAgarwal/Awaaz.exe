"""
Singleton Service Layer for Real-Time Streaming Session Management.
"""
import time
import logging
from typing import Dict, Optional, List

from app.streaming.schemas import StreamStartMessage, SessionState, StreamingSessionSummary
from app.streaming.session import StreamingSession

logger = logging.getLogger(__name__)


class StreamingSessionManager:
    """
    Central manager coordinating all concurrent real-time audio streams.
    Enforces concurrency limits, backpressure controls, and session garbage collection.
    """
    _instance: Optional["StreamingSessionManager"] = None
    MAX_ACTIVE_SESSIONS: int = 20
    SESSION_IDLE_TTL_SEC: float = 120.0

    def __init__(self):
        self._sessions: Dict[str, StreamingSession] = {}

    @classmethod
    def get_instance(cls) -> "StreamingSessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_session(self, config: "StreamStartMessage | str") -> StreamingSession:
        """
        Creates and registers a new isolated streaming session.
        Enforces maximum active session limits.
        """
        self.cleanup_stale_sessions()

        active_count = sum(1 for s in self._sessions.values() if s.state == SessionState.RUNNING)
        if active_count >= self.MAX_ACTIVE_SESSIONS:
            raise RuntimeError(
                f"Maximum concurrent streaming sessions limit ({self.MAX_ACTIVE_SESSIONS}) reached. Please retry later."
            )

        if isinstance(config, str):
            session_id = config
            sample_rate = 16000
            channels = 1
            enrolled_profile_id = None
            window_sec = 3.0
            hop_sec = 1.5
        else:
            session_id = config.session_id
            sample_rate = config.sample_rate
            channels = config.channels
            enrolled_profile_id = config.enrolled_profile_id
            window_sec = config.analysis_window_sec
            hop_sec = config.analysis_hop_sec

        session = StreamingSession(
            session_id=session_id,
            sample_rate=sample_rate,
            channels=channels,
            enrolled_profile_id=enrolled_profile_id,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )

        self._sessions[session.session_id] = session
        logger.info(f"Created new streaming session: {session.session_id} (active: {active_count + 1})")
        return session


    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Retrieves active session by ID."""
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> bool:
        """Closes and unregisters a session."""
        session = self._sessions.get(session_id)
        if session:
            await session.stop()
            del self._sessions[session_id]
            logger.info(f"Closed and cleaned up streaming session: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[StreamingSessionSummary]:
        """Returns non-sensitive metadata for all active sessions."""
        self.cleanup_stale_sessions()
        return [s.get_summary() for s in self._sessions.values()]

    def cleanup_stale_sessions(self) -> int:
        """Removes sessions that have been idle or completed beyond TTL."""
        now = time.time()
        stale_ids = []
        for sid, s in self._sessions.items():
            is_stale = (now - s.last_event_at) > self.SESSION_IDLE_TTL_SEC
            is_completed = s.state in [SessionState.COMPLETED, SessionState.ERROR] and (now - s.last_event_at) > 10.0
            if is_stale or is_completed:
                stale_ids.append(sid)

        for sid in stale_ids:
            logger.info(f"Garbage-collecting stale streaming session: {sid}")
            del self._sessions[sid]

        return len(stale_ids)
