"""
Streaming Data Models, Protocols, and Schemas for VoiceShield Step 10.
"""
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
import time


class SessionState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class StreamEventType(str, Enum):
    STREAM_STARTED = "STREAM_STARTED"
    STREAM_STOPPED = "STREAM_STOPPED"
    STREAM_ERROR = "STREAM_ERROR"
    ANALYSIS_WINDOW_COMPLETED = "ANALYSIS_WINDOW_COMPLETED"
    RISK_LEVEL_CHANGED = "RISK_LEVEL_CHANGED"
    HIGH_RISK_DETECTED = "HIGH_RISK_DETECTED"
    CRITICAL_RISK_DETECTED = "CRITICAL_RISK_DETECTED"
    SYNTHETIC_SCORE_UPDATED = "SYNTHETIC_SCORE_UPDATED"
    SPEAKER_MATCH_UPDATED = "SPEAKER_MATCH_UPDATED"
    AUDIO_QUALITY_DEGRADED = "AUDIO_QUALITY_DEGRADED"
    HEARTBEAT = "HEARTBEAT"


class AudioEncoding(str, Enum):
    PCM_F32LE = "pcm_f32le"
    PCM_S16LE = "pcm_s16le"
    BASE64_PCM_F32 = "base64_pcm_f32"
    BASE64_PCM_S16 = "base64_pcm_s16"
    WEBM_OPUS = "webm_opus"
    WAV = "wav"


# Client -> Server Inbound WebSocket Messages
class StreamStartMessage(BaseModel):
    action: str = Field("start", description="Must be 'start'")
    session_id: Optional[str] = Field(None, description="Optional custom session ID; generated if omitted.")
    sample_rate: int = Field(16000, description="Incoming audio sample rate in Hz (default: 16000).")
    channels: int = Field(1, description="Number of channels (default: 1 mono).")
    encoding: AudioEncoding = Field(AudioEncoding.PCM_F32LE, description="Audio encoding format.")
    enrolled_profile_id: Optional[str] = Field(None, description="Enrolled speaker profile ID for identity verification.")
    analysis_window_sec: float = Field(3.0, description="Analysis window length in seconds (default: 3.0s).")
    analysis_hop_sec: float = Field(1.5, description="Analysis hop interval in seconds (default: 1.5s).")


class StreamAudioChunkMessage(BaseModel):
    action: str = Field("chunk", description="Must be 'chunk' or 'audio'")
    sequence_number: int = Field(..., description="Monotonically increasing chunk index.")
    data: Optional[str] = Field(None, description="Base64 encoded PCM audio data if text frame.")
    sample_count: Optional[int] = Field(None, description="Number of samples contained in this chunk.")


class StreamStopMessage(BaseModel):
    action: str = Field("stop", description="Must be 'stop'")


class StreamPingMessage(BaseModel):
    action: str = Field("ping", description="Must be 'ping'")
    timestamp: float = Field(default_factory=time.time)


# Server -> Client Outbound Event Messages
class WindowTimingInfo(BaseModel):
    window_index: int
    window_start_sec: float
    window_end_sec: float
    duration_sec: float


class StreamingSyntheticSignal(BaseModel):
    score: float
    score_type: str = "uncalibrated_model_score"
    classification: str
    confidence_band: str
    detector_id: str
    latency_ms: float


class StreamingSpeakerSignal(BaseModel):
    status: str  # "MATCH", "NON_MATCH", "UNCERTAIN", "NOT_AVAILABLE"
    similarity_score: Optional[float] = None
    confidence_band: Optional[str] = None
    profile_id: Optional[str] = None
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None


class RollingRiskMetrics(BaseModel):
    latest_risk_score: float
    latest_risk_level: str
    latest_action: str
    previous_risk_score: Optional[float] = None
    rolling_max_risk: float
    rolling_mean_risk: float
    analyzed_windows_count: int
    consecutive_high_risk_windows: int
    consecutive_critical_windows: int


class StreamEvent(BaseModel):
    event_type: StreamEventType
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    message: str
    timing: Optional[WindowTimingInfo] = None
    synthetic_signal: Optional[StreamingSyntheticSignal] = None
    speaker_signal: Optional[StreamingSpeakerSignal] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommended_action: Optional[str] = None
    action_rationale: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    rolling_metrics: Optional[RollingRiskMetrics] = None
    processing_latency_ms: Optional[float] = None
    processing_lag_ms: Optional[float] = None
    audio_duration_received_sec: Optional[float] = None


class StreamingSessionSummary(BaseModel):
    session_id: str
    state: SessionState
    created_at: float
    last_event_at: float
    sample_rate: int
    channels: int
    enrolled_profile_id: Optional[str] = None
    total_chunks_received: int
    total_samples_received: int
    audio_duration_received_sec: float
    analyzed_windows_count: int
    rolling_metrics: RollingRiskMetrics
    evaluation_status: str = "NOT_VALIDATED"
    privacy_policy: str = "raw_audio_ram_only_ephemeral"
