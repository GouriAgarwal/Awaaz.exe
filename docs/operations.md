# VoiceShield Operational Runbook

## 1. Daily Operations Checklist
1. **Health Verification**: Query `GET /api/health/ready` to verify model files and checksums are loaded and valid.
2. **Capacity Check**: Query `GET /api/observability/streaming` to inspect current active sessions versus max capacity (default 20 concurrent sessions).
3. **Error Auditing**: Check `GET /api/observability/errors` to inspect taxonomy frequencies and detect any surge in `AUDIO_INVALID`, `STREAM_RATE_LIMITED`, or `MODEL_UNAVAILABLE`.

## 2. Configuration Parameters
All operational thresholds can be configured via environment variables:
| Variable | Default | Purpose |
|---|---|---|
| `MAX_ACTIVE_SESSIONS` | 20 | Concurrency ceiling for real-time WebSocket streams |
| `MAX_STREAM_CHUNK_BYTES` | 65536 | Maximum permitted chunk size (64KB) |
| `MAX_STREAM_MESSAGE_BYTES` | 131072 | Maximum WebSocket payload (128KB) |
| `STREAM_RATE_LIMIT_CHUNKS_PER_SEC` | 20 | Max chunk ingestion rate per connection |
| `SESSION_IDLE_TTL_SEC` | 300 | Inactivity timeout before automated reaping |
| `STRICT_MODEL_INTEGRITY_CHECK` | true | Fail readiness if model SHA-256 does not match |
| `LOG_LEVEL` | INFO | Log severity filter (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `*` | Allowed client origins for web dashboards |
