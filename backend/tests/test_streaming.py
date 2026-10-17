"""
Comprehensive Automated Test Suite for VoiceShield Step 10: Real-Time Streaming Pipeline.
"""
import time
import json
import base64
import asyncio
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.audio.schemas import AudioData
from app.streaming.buffer import StreamingAudioBuffer
from app.streaming.processor import StreamingAudioProcessor
from app.streaming.session import StreamingSession
from app.streaming.service import StreamingSessionManager
from app.streaming.schemas import (
    SessionState,
    StreamStartMessage,
    StreamEventType,
)


@pytest.fixture
def client():
    return TestClient(app)


def generate_pcm_chunk(duration_sec: float = 0.5, sample_rate: int = 16000, freq: float = 440.0) -> np.ndarray:
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return samples


class TestStreamingBufferAndWindowing:
    """Tests for the 16 kHz bounded rolling audio buffer."""

    def test_buffer_chunk_ordering_and_sample_count(self):
        buffer = StreamingAudioBuffer(sample_rate=16000, window_sec=3.0, hop_sec=1.5)
        c1 = generate_pcm_chunk(0.5, 16000)
        c2 = generate_pcm_chunk(1.0, 16000)

        buffer.append_chunk(c1, sequence_number=0)
        buffer.append_chunk(c2, sequence_number=1)

        assert buffer.total_samples_received == 8000 + 16000
        assert buffer.total_chunks_received == 2
        assert buffer.total_duration_received_sec == 1.5
        assert buffer.buffered_duration_sec == 1.5

    def test_3s_window_1_5s_hop_schedule(self):
        buffer = StreamingAudioBuffer(sample_rate=16000, window_sec=3.0, hop_sec=1.5)

        # 1. Append 2.0s -> Not enough for first 3.0s window
        buffer.append_chunk(generate_pcm_chunk(2.0, 16000))
        windows = buffer.get_available_windows()
        assert len(windows) == 0

        # 2. Append 1.5s -> Total 3.5s -> Window 0 (0.0s - 3.0s) extracted
        buffer.append_chunk(generate_pcm_chunk(1.5, 16000))
        windows = buffer.get_available_windows()
        assert len(windows) == 1
        w0_idx, w0_start, w0_end, w0_audio = windows[0]
        assert w0_idx == 0
        assert w0_start == 0.0
        assert w0_end == 3.0
        assert len(w0_audio.samples) == 48000
        assert w0_audio.duration_seconds == 3.0

        # 3. Append 1.5s -> Total 5.0s -> Window 1 (1.5s - 4.5s) extracted
        buffer.append_chunk(generate_pcm_chunk(1.5, 16000))
        windows = buffer.get_available_windows()
        assert len(windows) == 1
        w1_idx, w1_start, w1_end, w1_audio = windows[0]
        assert w1_idx == 1
        assert w1_start == 1.5
        assert w1_end == 4.5
        assert len(w1_audio.samples) == 48000

        # 4. Append 2.0s -> Total 7.0s -> Window 2 (3.0s - 6.0s) extracted
        buffer.append_chunk(generate_pcm_chunk(2.0, 16000))
        windows = buffer.get_available_windows()
        assert len(windows) == 1
        w2_idx, w2_start, w2_end, _ = windows[0]
        assert w2_idx == 2
        assert w2_start == 3.0
        assert w2_end == 6.0

    def test_buffer_bounded_memory_eviction(self):
        # Buffer max size 6.0 seconds
        buffer = StreamingAudioBuffer(sample_rate=16000, window_sec=3.0, hop_sec=1.5, max_buffer_sec=6.0)

        # Stream 10 seconds of audio in 1s chunks
        for i in range(10):
            buffer.append_chunk(generate_pcm_chunk(1.0, 16000), sequence_number=i)
            _ = buffer.get_available_windows()

        assert buffer.total_duration_received_sec == 10.0
        # Internal memory should be bounded
        assert buffer.buffered_duration_sec <= 6.5

    def test_buffer_nan_and_inf_handling(self):
        buffer = StreamingAudioBuffer(sample_rate=16000, window_sec=3.0, hop_sec=1.5)
        bad_chunk = np.array([np.nan, np.inf, -np.inf, 2.5, -3.0], dtype=np.float32)
        buffer.append_chunk(bad_chunk)

        assert np.isfinite(buffer._buffer).all()
        assert np.max(buffer._buffer) <= 1.0
        assert np.min(buffer._buffer) >= -1.0


class TestStreamingSessionLifecycle:
    """Tests for session creation, state transitions, and summaries."""

    def test_session_lifecycle(self):
        async def _run():
            session = StreamingSession(session_id="test_sess_01", enrolled_profile_id="spk_01")
            assert session.state == SessionState.CREATED

            await session.start()
            assert session.state == SessionState.RUNNING

            # Push 3.5s audio to trigger window 0
            chunk = generate_pcm_chunk(3.5, 16000)
            events = await session.push_chunk(chunk)
            assert len(events) >= 1
            assert events[0].event_type == StreamEventType.ANALYSIS_WINDOW_COMPLETED
            assert events[0].timing.window_index == 0

            summary = session.get_summary()
            assert summary.session_id == "test_sess_01"
            assert summary.analyzed_windows_count == 1
            assert summary.rolling_metrics.latest_risk_score >= 0.0

            await session.stop()
            assert session.state == SessionState.COMPLETED

        asyncio.run(_run())

    def test_session_summary_no_audio_or_embedding_leakage(self):
        async def _run():
            session = StreamingSession(session_id="privacy_test")
            await session.start()
            await session.push_chunk(generate_pcm_chunk(3.5, 16000))

            summary_dict = session.get_summary().model_dump()
            summary_str = json.dumps(summary_dict)

            # Guarantee no raw samples or vectors in summary
            assert "samples" not in summary_dict
            assert "embedding" not in summary_dict
            assert summary_dict["privacy_policy"] == "raw_audio_ram_only_ephemeral"

        asyncio.run(_run())


class TestStreamingProcessorAndRollingRisk:
    """Tests for multi-modal window inference and rolling temporal metrics."""

    def test_processor_with_no_profile_id_sets_not_available(self):
        proc = StreamingAudioProcessor(session_id="no_profile_test", enrolled_profile_id=None)
        audio = AudioData(samples=generate_pcm_chunk(3.0, 16000), sample_rate=16000, duration_seconds=3.0, channels=1)

        events = proc.process_window(
            window_index=0,
            start_sec=0.0,
            end_sec=3.0,
            audio_data=audio,
            stream_duration_sec=3.0,
        )

        assert len(events) >= 1
        main_event = events[0]
        assert main_event.speaker_signal.status == "NOT_AVAILABLE"
        assert main_event.speaker_signal.similarity_score is None
        assert main_event.synthetic_signal.score >= 0.0
        assert main_event.risk_score is not None

    def test_processor_rolling_risk_accumulation(self):
        proc = StreamingAudioProcessor(session_id="rolling_test")
        audio = AudioData(samples=generate_pcm_chunk(3.0, 16000), sample_rate=16000, duration_seconds=3.0, channels=1)

        # Process 3 consecutive windows
        for i in range(3):
            events = proc.process_window(
                window_index=i,
                start_sec=i * 1.5,
                end_sec=i * 1.5 + 3.0,
                audio_data=audio,
                stream_duration_sec=i * 1.5 + 3.0,
            )
            assert len(events) >= 1

        assert len(proc.risk_scores_history) == 3
        rolling = events[0].rolling_metrics
        assert rolling.analyzed_windows_count == 3
        assert rolling.rolling_max_risk >= rolling.rolling_mean_risk


class TestStreamingServiceAndBackpressure:
    """Tests for session manager, concurrency limits, and garbage collection."""

    def test_concurrency_limit_enforcement(self):
        manager = StreamingSessionManager()
        manager.MAX_ACTIVE_SESSIONS = 3
        manager._sessions.clear()

        # Create 3 sessions
        s1 = manager.create_session(StreamStartMessage(action="start", session_id="s1"))
        s1.state = SessionState.RUNNING
        s2 = manager.create_session(StreamStartMessage(action="start", session_id="s2"))
        s2.state = SessionState.RUNNING
        s3 = manager.create_session(StreamStartMessage(action="start", session_id="s3"))
        s3.state = SessionState.RUNNING

        # 4th session should fail with max active limit
        with pytest.raises(RuntimeError, match="Maximum concurrent streaming sessions limit"):
            manager.create_session(StreamStartMessage(action="start", session_id="s4"))


class TestStreamingApiIntegration:
    """Tests for REST endpoints and WebSocket protocol."""

    def test_rest_create_list_and_delete_session(self, client):
        # 1. Create session via REST
        res = client.post(
            "/api/stream/sessions",
            json={"action": "start", "sample_rate": 16000, "enrolled_profile_id": "api_spk_01"}
        )
        assert res.status_code == 201
        data = res.json()
        session_id = data["session_id"]
        assert data["state"] == "RUNNING"
        assert data["enrolled_profile_id"] == "api_spk_01"

        # 2. Get session summary
        get_res = client.get(f"/api/stream/sessions/{session_id}")
        assert get_res.status_code == 200
        assert get_res.json()["session_id"] == session_id

        # 3. List active sessions
        list_res = client.get("/api/stream/sessions")
        assert list_res.status_code == 200
        assert any(s["session_id"] == session_id for s in list_res.json())

        # 4. Delete session
        del_res = client.delete(f"/api/stream/sessions/{session_id}")
        assert del_res.status_code == 200

    def test_websocket_streaming_lifecycle(self, client):
        with client.websocket_connect("/api/stream/ws") as ws:
            # 1. Send start message
            ws.send_text(json.dumps({
                "action": "start",
                "sample_rate": 16000,
                "analysis_window_sec": 3.0,
                "analysis_hop_sec": 1.5,
            }))

            # 2. Receive stream started event
            start_event = json.loads(ws.receive_text())
            assert start_event["event_type"] == "STREAM_STARTED"

            # 3. Send ping -> receive heartbeat pong
            ws.send_text(json.dumps({"action": "ping"}))
            ping_event = json.loads(ws.receive_text())
            assert ping_event["event_type"] == "HEARTBEAT"

            # 4. Stream 3.5s of base64 audio in 0.5s chunks
            for i in range(7):
                pcm = generate_pcm_chunk(0.5, 16000)
                b64_str = base64.b64encode(pcm.tobytes()).decode("ascii")
                ws.send_text(json.dumps({
                    "action": "chunk",
                    "sequence_number": i,
                    "data": b64_str,
                }))

            # 5. Receive window completed event
            window_event = json.loads(ws.receive_text())
            assert window_event["event_type"] == "ANALYSIS_WINDOW_COMPLETED"
            assert window_event["timing"]["window_index"] == 0
            assert window_event["risk_score"] is not None
            assert window_event["synthetic_signal"]["score"] >= 0.0

            # 6. Send stop
            ws.send_text(json.dumps({"action": "stop"}))
            stop_event = json.loads(ws.receive_text())
            assert stop_event["event_type"] == "STREAM_STOPPED"
