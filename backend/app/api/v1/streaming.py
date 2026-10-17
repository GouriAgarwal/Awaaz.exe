"""
WebSocket and REST API Endpoints for Real-Time Streaming Pipeline (Step 10).
"""
import json
import base64
import asyncio
import logging
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from pydantic import ValidationError

from app.streaming.schemas import (
    StreamStartMessage,
    StreamAudioChunkMessage,
    StreamStopMessage,
    StreamPingMessage,
    StreamEvent,
    StreamEventType,
    StreamingSessionSummary,
    AudioEncoding,
)
from app.streaming.service import StreamingSessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Real-Time Streaming"])


@router.websocket("/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Bi-directional WebSocket for real-time speech ingestion and risk analysis event streaming.
    Protocol:
      1. Client sends JSON: `{"action": "start", "sample_rate": 16000, "enrolled_profile_id": "optional_id"}`
      2. Client streams audio chunks (binary PCM frames or JSON base64 chunks).
      3. Server pushes JSON `StreamEvent` objects whenever analysis windows complete.
      4. Client sends `{"action": "stop"}` or closes socket to terminate.
    """
    await websocket.accept()
    manager = StreamingSessionManager.get_instance()
    session = None

    try:
        while True:
            # Receive either text (JSON control) or binary (raw PCM)
            message = await websocket.receive()
            
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_text(
                        StreamEvent(
                            event_type=StreamEventType.STREAM_ERROR,
                            session_id=session.session_id if session else "unknown",
                            message="Invalid JSON payload received.",
                        ).model_dump_json()
                    )
                    continue

                action = payload.get("action", "").lower()

                if action == "start":
                    if session is not None:
                        await websocket.send_text(
                            StreamEvent(
                                event_type=StreamEventType.STREAM_ERROR,
                                session_id=session.session_id,
                                message="Session already started on this connection.",
                            ).model_dump_json()
                        )
                        continue

                    try:
                        start_msg = StreamStartMessage(**payload)
                        session = manager.create_session(start_msg)
                        await session.start()
                        start_event = StreamEvent(
                            event_type=StreamEventType.STREAM_STARTED,
                            session_id=session.session_id,
                            timestamp=session.last_event_at,
                            message=f"Streaming session {session.session_id} started (16kHz, target_profile={session.enrolled_profile_id or 'none'}).",
                            audio_duration_received_sec=0.0,
                        )
                        await websocket.send_text(start_event.model_dump_json())
                    except Exception as e:
                        await websocket.send_text(
                            StreamEvent(
                                event_type=StreamEventType.STREAM_ERROR,
                                session_id="failed",
                                message=f"Failed to start streaming session: {str(e)}",
                            ).model_dump_json()
                        )
                        break

                elif action in ["chunk", "audio"]:
                    if session is None:
                        await websocket.send_text(
                            StreamEvent(
                                event_type=StreamEventType.STREAM_ERROR,
                                session_id="none",
                                message="Received audio chunk before session was started. Send 'start' action first.",
                            ).model_dump_json()
                        )
                        continue

                    seq_num = payload.get("sequence_number", 0)
                    b64_data = payload.get("data")
                    if b64_data:
                        try:
                            raw_bytes = base64.b64decode(b64_data)
                            samples = np.frombuffer(raw_bytes, dtype=np.float32)
                            events = await session.push_chunk(samples, sequence_number=seq_num)
                            for ev in events:
                                await websocket.send_text(ev.model_dump_json())
                        except Exception as e:
                            logger.error(f"Error decoding chunk: {e}")

                elif action == "ping":
                    pong_event = StreamEvent(
                        event_type=StreamEventType.HEARTBEAT,
                        session_id=session.session_id if session else "uninitialized",
                        message="pong",
                    )
                    await websocket.send_text(pong_event.model_dump_json())

                elif action == "stop":
                    if session:
                        await session.stop()
                        stop_event = StreamEvent(
                            event_type=StreamEventType.STREAM_STOPPED,
                            session_id=session.session_id,
                            timestamp=session.last_event_at,
                            message=f"Streaming session {session.session_id} completed successfully.",
                            audio_duration_received_sec=round(session.buffer.total_duration_received_sec, 2),
                        )
                        await websocket.send_text(stop_event.model_dump_json())
                    break

            elif "bytes" in message and message["bytes"]:
                if session is None:
                    continue
                # Raw binary float32 PCM frame
                raw_bytes = message["bytes"]
                try:
                    samples = np.frombuffer(raw_bytes, dtype=np.float32)
                    events = await session.push_chunk(samples)
                    for ev in events:
                        await websocket.send_text(ev.model_dump_json())
                except Exception as e:
                    logger.error(f"Error processing binary audio frame: {e}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (session: {session.session_id if session else 'none'})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session:
            await session.emit_error(str(e))
    finally:
        if session:
            await manager.close_session(session.session_id)


# REST Endpoints for Session Status & Ingestion
@router.post(
    "/sessions",
    response_model=StreamingSessionSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new streaming session."
)
async def create_streaming_session(config: StreamStartMessage):
    """Initializes a new streaming session."""
    manager = StreamingSessionManager.get_instance()
    try:
        session = manager.create_session(config)
        await session.start()
        return session.get_summary()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/sessions",
    response_model=List[StreamingSessionSummary],
    summary="List all active streaming session summaries."
)
async def list_streaming_sessions():
    """Lists non-sensitive operational summaries of all active streaming sessions."""
    manager = StreamingSessionManager.get_instance()
    return manager.list_sessions()


@router.get(
    "/sessions/{session_id}",
    response_model=StreamingSessionSummary,
    summary="Get status and rolling metrics for a specific streaming session."
)
async def get_streaming_session(session_id: str):
    """Retrieves current session state and rolling metrics."""
    manager = StreamingSessionManager.get_instance()
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Streaming session '{session_id}' not found.")
    return session.get_summary()


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Terminate and clean up an active streaming session."
)
async def delete_streaming_session(session_id: str):
    """Terminates and closes a streaming session."""
    manager = StreamingSessionManager.get_instance()
    success = await manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Streaming session '{session_id}' not found.")
    return {"message": f"Session '{session_id}' terminated successfully."}
