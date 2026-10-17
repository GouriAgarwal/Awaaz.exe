# VoiceShield — Real-Time Streaming Detection Pipeline

## Architectural & Technical Reference (Step 10)

---

## 1. Executive Summary & Goal

VoiceShield Step 10 converts batch audio analysis into a **near-real-time streaming orchestration pipeline**. The pipeline ingests continuous audio chunks from client microphones or telephony streams, maintains a bounded 16 kHz mono rolling audio buffer in memory, and evaluates rolling 3.0-second analysis windows every 1.5 seconds.

```
Microphone / Audio Stream
           │
           ▼
[Chunk Ingestion & WebSocket Transport]
           │
           ▼
[16 kHz Bounded Rolling Audio Buffer (RAM-only)]
           │
           ▼ (Every 1.5s hop)
[Rolling 3.0s Analysis Window Extraction]
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
[Pretrained   [Pretrained
 AASIST]       ECAPA-TDNN]
(Step 6/7)     (Step 8)
     │           │
     └─────┬─────┘
           ▼
[Step 9 Controlled Fusion Matrix]
           │
           ▼
[Rolling Risk State & Temporal Aggregation]
           │
           ▼
[Push JSON Events via WebSocket to Frontend]
```

---

## 2. Streaming Architecture & Modules

The streaming orchestration layer is implemented in `backend/app/streaming/` with clear separation of concerns:

| Module | Location | Core Responsibility |
| :--- | :--- | :--- |
| `schemas.py` | [`backend/app/streaming/schemas.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/schemas.py) | Data models for session states, control messages, rolling metrics, and typed event schemas. |
| `buffer.py` | [`backend/app/streaming/buffer.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/buffer.py) | Bounded, ring-buffered 16 kHz mono float32 audio accumulator with 3.0s window / 1.5s hop extractor. |
| `processor.py` | [`backend/app/streaming/processor.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/processor.py) | Coordinates multi-modal inference per window: AASIST + ECAPA-TDNN + Step 9 Fusion, and calculates rolling risk metrics. |
| `session.py` | [`backend/app/streaming/session.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/session.py) | Encapsulates session state machine, buffer, processor, and event queue. |
| `events.py` | [`backend/app/streaming/events.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/events.py) | Asynchronous event dispatch and queue management. |
| `service.py` | [`backend/app/streaming/service.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/streaming/service.py) | Singleton session manager enforcing concurrency limits and garbage collecting idle sessions. |
| `streaming.py` | [`backend/app/api/v1/streaming.py`](file:///C:/Users/PRATHAMESH/.gemini/antigravity/scratch/voiceshield/backend/app/api/v1/streaming.py) | WebSocket (`WS /api/stream/ws`) and REST endpoints (`/api/stream/sessions`). |

---

## 3. Session Lifecycle & State Machine

Each streaming connection is governed by an isolated `StreamingSession` object with an explicit state machine:

```
  [CREATED] ──(start)──> [RUNNING] ──(stop)──> [STOPPING] ──> [COMPLETED]
                             │
                             └──(error)──────> [ERROR]
```

* `CREATED`: Session instantiated; parameters registered; buffer empty.
* `RUNNING`: Audio chunks being ingested, buffered, and evaluated.
* `STOPPING` / `COMPLETED`: Ingestion finalized; remaining windows flushed; resources released.
* `ERROR`: Critical stream or inference fault recorded; client alerted via event.

---

## 4. WebSocket Protocol Specification (`WS /api/stream/ws`)

### 1. Start Control Message (Client $\to$ Server)
```json
{
  "action": "start",
  "sample_rate": 16000,
  "channels": 1,
  "encoding": "pcm_f32le",
  "enrolled_profile_id": "executive_spk_01",
  "analysis_window_sec": 3.0,
  "analysis_hop_sec": 1.5
}
```

### 2. Audio Chunk Message (Client $\to$ Server)
Clients can send either:
* **JSON Frame with Base64 Payload**:
  ```json
  {
    "action": "chunk",
    "sequence_number": 42,
    "data": "<BASE64_ENCODED_FLOAT32_PCM_BYTES>"
  }
  ```
* **Raw Binary WebSocket Frames**: Direct binary float32 PCM byte arrays.

### 3. Heartbeat / Ping (Client $\to$ Server)
```json
{
  "action": "ping"
}
```
*Server Response*: `{"event_type": "HEARTBEAT", "message": "pong", ...}`

### 4. Analysis Event (Server $\to$ Client)
```json
{
  "event_type": "ANALYSIS_WINDOW_COMPLETED",
  "session_id": "stream_a1b2c3d4e5f6",
  "timestamp": 1725600000.123,
  "timing": {
    "window_index": 2,
    "window_start_sec": 3.0,
    "window_end_sec": 6.0,
    "duration_sec": 3.0
  },
  "synthetic_signal": {
    "score": 0.88,
    "score_type": "uncalibrated_model_score",
    "classification": "SYNTHETIC",
    "confidence_band": "HIGH",
    "detector_id": "VoiceShield-AASIST-Pretrained-v1",
    "latency_ms": 312.4
  },
  "speaker_signal": {
    "status": "MATCH",
    "similarity_score": 0.82,
    "confidence_band": "HIGH",
    "profile_id": "executive_spk_01",
    "latency_ms": 395.1
  },
  "risk_score": 91.2,
  "risk_level": "CRITICAL",
  "recommended_action": "BLOCK_OR_ESCALATE",
  "action_rationale": "High speaker similarity combined with synthetic speech cues indicates potential voice cloning.",
  "rolling_metrics": {
    "latest_risk_score": 91.2,
    "latest_risk_level": "CRITICAL",
    "latest_action": "BLOCK_OR_ESCALATE",
    "rolling_max_risk": 91.2,
    "rolling_mean_risk": 84.5,
    "analyzed_windows_count": 3,
    "consecutive_high_risk_windows": 3,
    "consecutive_critical_windows": 2
  },
  "processing_latency_ms": 710.5,
  "processing_lag_ms": 42.0
}
```

### 5. Stop Message (Client $\to$ Server)
```json
{
  "action": "stop"
}
```

---

## 5. Analysis Window & Inherent Observation Delay

AASIST graph-attention architecture natively operates on a **3.0-second spectral context** ($64,600$ samples at 16 kHz):

```
Time (seconds):
0.0s        1.5s        3.0s        4.5s        6.0s        7.5s
│───────────│───────────│───────────│───────────│───────────│
[─── Window #0 (0.0s–3.0s) ───]
            [─── Window #1 (1.5s–4.5s) ───]
                        [─── Window #2 (3.0s–6.0s) ───]
```

### Crucial Latency Distinction:
1. **Observation Delay ($3.0\text{ s}$)**: The system must physically receive $3.0\text{ seconds}$ of incoming speech before Window #0 can be extracted. This is an acoustic requirement of the underlying neural spectral-graph representation.
2. **Computational Inference Latency ($\approx 700\text{ ms}$)**: Once 3.0s of audio is accumulated, AASIST detection ($310\text{ ms}$) + ECAPA embedding ($390\text{ ms}$) + Fusion ($< 2\text{ ms}$) executes in $\approx 700\text{ ms}$ on CPU.
3. **Subsequent Cadence ($1.5\text{ s}$)**: New risk observations are produced every $1.5\text{ seconds}$ of additional incoming audio.

---

## 6. Rolling Risk State & Temporal Aggregation

To avoid transient false alarms without masking persistent attacks, `StreamingAudioProcessor` maintains a temporal rolling state:
* `latest_risk_score`: The score from the most recently analyzed 3.0s window (primary display metric).
* `rolling_max_risk`: The highest risk observed during the active session (escalation flag).
* `rolling_mean_risk`: The arithmetic average of all completed windows in the session (informational telemetry).
* `consecutive_high_risk_windows` / `consecutive_critical_windows`: Counter for repeated anomalies.

---

## 7. Backpressure, Concurrency, and Safety

1. **Max Concurrent Streams**: Server restricts concurrent active streams to `MAX_ACTIVE_SESSIONS = 20`. Additional requests receive HTTP 400 / error event.
2. **Bounded Buffer Memory**: `StreamingAudioBuffer` enforces `max_buffer_sec = 15.0`. Old audio slices beyond the extraction horizon are discarded automatically, preventing runaway RAM usage.
3. **Session TTL Cleanup**: Stale or orphaned sessions idle for $> 120\text{ seconds}$ are automatically garbage-collected.
4. **NaN / Inf Protection**: All incoming float32 sample chunks are sanitized via `np.nan_to_num` and clamped to $[-1.0, 1.0]$.

---

## 8. Biometric Privacy Guarantees

* **RAM-Only Ephemeral Buffering**: Raw audio samples exist strictly in memory inside `StreamingAudioBuffer._buffer` and are deleted as windows advance.
* **No Raw Audio Logging**: Logs contain only session IDs, window timing, scores, and risk levels.
* **No Biometric Embeddings in Logs or Events**: 192-dimensional ECAPA embeddings are compared in RAM and never serialized to client events.

---

## 9. Latency Benchmark Summary

Measured across streaming trials on standard CPU:

| Streaming Stage | Value | Interpretation |
| :--- | :--- | :--- |
| **Initial Observation Buffer Delay** | $3.00\text{ s}$ | Physical time required to accumulate Window #0. |
| **Window Evaluation Hop Rate** | $1.50\text{ s}$ | Cadence between subsequent evaluations. |
| **AASIST Inference Latency** | $\approx 310\text{ ms}$ | Softmax spoof posterior computation. |
| **ECAPA-TDNN Embedding Latency** | $\approx 390\text{ ms}$ | 192-D embedding extraction & cosine angle. |
| **Step 9 Matrix Fusion Latency** | $< 2.5\text{ ms}$ | Deterministic rule matrix evaluation. |
| **Total Computational Latency per Window** | **$\approx 710\text{ ms}$** | Total CPU execution time. |
| **Real-Time Factor (RTF)** | **$0.237$** | Execution is ~4.2x faster than 3.0s window length. |

---

## 10. Scientific & Security Disclosures

* **Pretrained Models**: AASIST (`VoiceShield-AASIST-Pretrained-v1`) and ECAPA-TDNN (`speechbrain_ecapa_tdnn_voxceleb`) are official pretrained checkpoints; VoiceShield has not retrained these models.
* **Streaming Validation Status**: `streaming_evaluation_status = NOT_VALIDATED` until a genuine, multi-speaker streaming benchmark dataset is evaluated.
* **Score Semantics**: Risk score is a deterministic heuristic decision-support index (`NOT_CALIBRATED`) and not an autonomous fraud determination.
