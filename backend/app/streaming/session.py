"""
Streaming Session Abstraction and Lifecycle Manager.
"""
import uuid
import time
import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any

from app.streaming.schemas import (
    SessionState,
    StreamEventType,
    StreamEvent,
    StreamingSessionSummary,
    RollingRiskMetrics,
)
from app.streaming.buffer import StreamingAudioBuffer
from app.streaming.processor import StreamingAudioProcessor
from app.streaming.events import StreamEventQueue

logger = logging.getLogger(__name__)


class StreamingSession:
    """
    Isolated state and processing pipeline for a single real-time audio stream.
    """
    def __init__(
        self,
        session_id: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        enrolled_profile_id: Optional[str] = None,
        window_sec: float = 3.0,
        hop_sec: float = 1.5,
    ):
        self.session_id: str = session_id or f"stream_{uuid.uuid4().hex[:12]}"
        self.state: SessionState = SessionState.CREATED
        self.created_at: float = time.time()
        self.last_event_at: float = self.created_at
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.enrolled_profile_id: Optional[str] = enrolled_profile_id

        self.buffer = StreamingAudioBuffer(
            sample_rate=self.sample_rate,
            window_sec=window_sec,
            hop_sec=hop_sec,
            max_buffer_sec=15.0,
        )
        self.processor = StreamingAudioProcessor(
            session_id=self.session_id,
            enrolled_profile_id=self.enrolled_profile_id,
        )
        self.event_queue = StreamEventQueue(maxsize=100)
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Transitions session to RUNNING state and emits STREAM_STARTED event."""
        async with self._lock:
            if self.state in [SessionState.COMPLETED, SessionState.ERROR]:
                raise RuntimeError(f"Cannot start completed or errored session: {self.state}")
            self.state = SessionState.RUNNING
            self.last_event_at = time.time()

            start_event = StreamEvent(
                event_type=StreamEventType.STREAM_STARTED,
                session_id=self.session_id,
                timestamp=self.last_event_at,
                message=f"Streaming session {self.session_id} started (16kHz, target_profile={self.enrolled_profile_id or 'none'}).",
                audio_duration_received_sec=0.0,
            )
            await self.event_queue.put(start_event)

    async def push_chunk(self, chunk_samples: np.ndarray, sequence_number: Optional[int] = None) -> List[StreamEvent]:
        """
        Appends raw audio samples, evaluates any complete analysis windows, and enqueues resulting events.
        """
        async with self._lock:
            if self.state != SessionState.RUNNING:
                logger.warning(f"Pushing audio chunk to session {self.session_id} in state {self.state}")

            self.last_event_at = time.time()
            self.buffer.append_chunk(chunk_samples, sequence_number=sequence_number)

            # Check if any new analysis windows became available
            available_windows = self.buffer.get_available_windows()
            emitted_events: List[StreamEvent] = []

            for win_idx, start_sec, end_sec, audio_data in available_windows:
                events = self.processor.process_window(
                    window_index=win_idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    audio_data=audio_data,
                    stream_duration_sec=self.buffer.total_duration_received_sec,
                )
                for ev in events:
                    await self.event_queue.put(ev)
                    emitted_events.append(ev)

            return emitted_events

    async def stop(self) -> None:
        """Cleanly terminates the streaming session and emits STREAM_STOPPED event."""
        async with self._lock:
            if self.state in [SessionState.COMPLETED, SessionState.STOPPING]:
                return
            self.state = SessionState.STOPPING
            self.last_event_at = time.time()

            stop_event = StreamEvent(
                event_type=StreamEventType.STREAM_STOPPED,
                session_id=self.session_id,
                timestamp=self.last_event_at,
                message=f"Streaming session {self.session_id} completed successfully.",
                audio_duration_received_sec=round(self.buffer.total_duration_received_sec, 2),
            )
            await self.event_queue.put(stop_event)
            self.state = SessionState.COMPLETED

    async def emit_error(self, error_message: str) -> None:
        """Transitions session to ERROR state and emits STREAM_ERROR event."""
        async with self._lock:
            self.state = SessionState.ERROR
            self.last_event_at = time.time()

            error_event = StreamEvent(
                event_type=StreamEventType.STREAM_ERROR,
                session_id=self.session_id,
                timestamp=self.last_event_at,
                message=error_message,
                audio_duration_received_sec=round(self.buffer.total_duration_received_sec, 2),
            )
            await self.event_queue.put(error_event)

    def get_summary(self) -> StreamingSessionSummary:
        """Returns non-sensitive operational summary of current session state."""
        rolling_metrics = RollingRiskMetrics(
            latest_risk_score=round(self.processor.latest_risk_score, 1),
            latest_risk_level=self.processor.latest_risk_level,
            latest_action=self.processor.latest_action,
            previous_risk_score=None,
            rolling_max_risk=round(float(np.max(self.processor.risk_scores_history)), 1) if self.processor.risk_scores_history else 0.0,
            rolling_mean_risk=round(float(np.mean(self.processor.risk_scores_history)), 1) if self.processor.risk_scores_history else 0.0,
            analyzed_windows_count=len(self.processor.risk_scores_history),
            consecutive_high_risk_windows=self.processor.consecutive_high_risk_windows,
            consecutive_critical_windows=self.processor.consecutive_critical_windows,
        )

        return StreamingSessionSummary(
            session_id=self.session_id,
            state=self.state,
            created_at=self.created_at,
            last_event_at=self.last_event_at,
            sample_rate=self.sample_rate,
            channels=self.channels,
            enrolled_profile_id=self.enrolled_profile_id,
            total_chunks_received=self.buffer.total_chunks_received,
            total_samples_received=self.buffer.total_samples_received,
            audio_duration_received_sec=round(self.buffer.total_duration_received_sec, 2),
            analyzed_windows_count=len(self.processor.risk_scores_history),
            rolling_metrics=rolling_metrics,
            evaluation_status="NOT_VALIDATED",
            privacy_policy="raw_audio_ram_only_ephemeral",
        )
