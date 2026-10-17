"""
Event Dispatcher and Queue Management for Streaming Sessions.
"""
import asyncio
import logging
from typing import List, Optional
from app.streaming.schemas import StreamEvent

logger = logging.getLogger(__name__)


class StreamEventQueue:
    """
    Asynchronous event queue for a single streaming session.
    Allows WebSocket transports to subscribe and stream events in real time.
    """
    def __init__(self, maxsize: int = 100):
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=maxsize)
        self._history: List[StreamEvent] = []
        self._max_history: int = 50

    async def put(self, event: StreamEvent) -> None:
        """Pushes an event onto the queue and appends to local history."""
        try:
            # Drop oldest event if queue is full (backpressure mitigation)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put(event)
        except Exception as e:
            logger.error(f"Failed to put event in StreamEventQueue: {e}")

        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    async def push(self, event: StreamEvent) -> None:
        """Alias for put."""
        await self.put(event)

    def qsize(self) -> int:
        """Return current size of queue."""
        return self._queue.qsize()

    async def get(self) -> StreamEvent:
        """Awaits next event from queue."""
        return await self._queue.get()

    def get_history(self) -> List[StreamEvent]:
        """Returns snapshot of recent events."""
        return list(self._history)
