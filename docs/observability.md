# VoiceShield Observability & Telemetry Reference

## 1. Overview
VoiceShield provides end-to-end operational observability across batch and real-time streaming operations without fabricating performance numbers or compromising biometrics privacy.

## 2. Health & Readiness Probes
- **Liveness Probe**: `GET /api/health/live`
  - Ultra-fast non-blocking endpoint verifying process health.
  - Never initiates neural model inference or blocking disk I/O.
- **Readiness Probe**: `GET /api/health/ready`
  - Multi-factor inspection of AASIST checkpoint, ECAPA checkpoint, streaming capacity, and metrics subsystem.
  - Returns `200 OK` for `READY` or `DEGRADED`, and `503 Service Unavailable` for `NOT_READY`.
- **Operational Status**: `GET /api/health/status`
  - Subsystem status, active detector name, model integrity flags, and scientific disclosures.

## 3. Telemetry Endpoints
- `GET /api/observability/metrics`: Aggregate HTTP request counts, status code distributions, percentiles, model inference counts, latencies, and streaming load.
- `GET /api/observability/models`: Model lifecycle events, verified SHA-256 hashes, inference latencies, and scientific validation disclosures.
- `GET /api/observability/streaming`: Real-time session counts, capacity utilization %, backpressure lag, and rate limit counters.
- `GET /api/observability/errors`: Unified operational error taxonomy frequencies and recent failures with correlation IDs.

## 4. Privacy-Preserving Structured Logging
- Formatted as single-line JSON with timestamps, logger names, levels, and correlation IDs (`X-Request-ID`).
- **Data Scrubbing Guarantee**: The `SensitiveDataSanitizer` automatically redacts:
  - Audio waveforms (`[REDACTED_VECTOR_LEN_N]`)
  - 512-dimensional speaker embeddings (`[REDACTED_SENSITIVE_DATA]`)
  - Base64 encoded audio buffers (`[REDACTED_BASE64_AUDIO]`)
  - API keys, passwords, and bearer tokens (`[REDACTED_SENSITIVE_DATA]`)
