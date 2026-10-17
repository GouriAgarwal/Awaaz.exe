# VoiceShield Capacity Planning & Sizing Guidelines

## 1. Resource Consumption Profile
Based on empirical profiling of the AASIST graph attention neural network and SpeechBrain ECAPA-TDNN verifier:
- **Baseline Memory Footprint**: ~1.2 GB RSS (including PyTorch runtime and loaded weights).
- **Per-Active Streaming Session Memory**: ~12 MB RAM (3.0s windowing buffer, 15.0s maximum rolling history, in-memory circular float32 buffers).
- **CPU Inference Budget**:
  - AASIST inference on 3.0s window: ~35–50ms on modern x86_64 vCPU.
  - ECAPA embedding extraction on 3.0s window: ~20–35ms.
  - Controlled Risk Fusion: <1ms.
  - Total processing per window: ~60–85ms.

## 2. Sizing Recommendations

| Scale Tier | Active Streams | Recommended CPU | Recommended RAM | Backend Instances |
|---|---|---|---|---|
| **Development** | 1–5 | 2 vCPUs | 2 GB | 1 |
| **Small Production** | 10–20 | 4 vCPUs | 4 GB | 1 |
| **Medium Production** | 50–100 | 16 vCPUs | 16 GB | 4 instances (25/ea) behind Nginx |
| **Enterprise Fleet** | 500+ | Kubernetes Cluster | 64+ GB | Auto-scaled Pods (HPA on CPU > 70%) |

## 3. Ephemeral Buffer Limits
- Maximum audio duration stored in memory per session is hard-capped at **15.0 seconds**.
- Older chunks are shifted out in a strict FIFO circular manner.
- Upon session closure or timeout, all audio buffer memory is immediately dereferenced and cleaned up.
