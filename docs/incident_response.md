# VoiceShield Incident Response Playbook

## 1. Subsystem Failure Scenarios

### Scenario A: Readiness Probe Fails (`503 Service Unavailable`)
- **Symptoms**: Orchestrator stops routing traffic to the container. `GET /api/health/ready` returns `NOT_READY`.
- **Triaging**:
  1. Check response `components` map. Identify which component failed (`aasist_detector` or `ecapa_speaker_verifier`).
  2. Inspect container startup logs for checksum mismatch warnings.
  3. Verify weight paths: `models/weights/AASIST.pth` and `models/weights/embedding_model.ckpt`.
- **Resolution**:
  - If checksum failed: Re-download trusted model weights from primary upstream repository and verify against documented SHA-256 hashes (`51d2d9cf...` for AASIST, `0575cb64...` for ECAPA).
  - If non-strict mode is desired temporarily during maintenance, set `STRICT_MODEL_INTEGRITY_CHECK=false`.

### Scenario B: Streaming Session Saturation (`SESSION_LIMIT_REACHED`)
- **Symptoms**: New WebSocket connections rejected with HTTP 429 / closure code.
- **Triaging**:
  1. Check `GET /api/observability/streaming` for `active_sessions` vs `max_active_sessions`.
  2. Check for idle/abandoned connections.
- **Resolution**:
  - Scale up backend container replicas horizontally behind Nginx load balancer.
  - Tune `SESSION_IDLE_TTL_SEC` downward (e.g., from 300s to 120s) to aggressively reclaim leaked connections.

### Scenario C: High Audio Processing Lag
- **Symptoms**: `p95_lag_ms` climbing above 500ms in `/api/observability/streaming`.
- **Triaging**:
  1. Check host CPU throttling.
  2. Verify client chunk ingestion rate is within limits (default 20 chunks/sec).
- **Resolution**:
  - Adjust CPU quota in Docker Compose (`limits.cpus: '4.0'`).
  - Advise upstream clients to batch chunks into recommended 200ms increments.
