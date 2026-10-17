# VoiceShield Production Deployment Guide

## 1. Overview
VoiceShield is engineered for production deployment across Docker, Kubernetes, or bare-metal Linux environments. It packages pretrained AASIST synthetic voice detection and SpeechBrain ECAPA-TDNN speaker verification behind FastAPI and an asynchronous WebSocket streaming pipeline.

## 2. Container Architecture
- **Base Image**: `python:3.11-slim`
- **Multi-Stage Build**: Separates compile-time wheels and dependencies from the minimal runtime image.
- **Unprivileged Execution**: Drops privileges to non-root user `voiceshield (uid: 10001, gid: 10001)`.
- **System Shared Libraries**: Minimal runtime dependencies (`libsndfile1`, `curl`).

## 3. Quick Start with Docker Compose

### Development Mode
```bash
cd deployment
docker-compose up --build
```
Runs the application on port `8000` with local code volumes mounted for hot-reloading.

### Production Mode
```bash
cd deployment
docker-compose -f docker-compose.production.yml up -d
```
Spawns:
1. `voiceshield-backend-prod`: Hardened container bound with CPU (`4.0`) and memory (`4GB`) resource limits.
2. `voiceshield-nginx-prod`: High-performance reverse proxy handling SSL termination, WebSocket connection upgrades, and 25MB client upload body limits.

## 4. Kubernetes Probes Configuration
```yaml
livenessProbe:
  httpGet:
    path: /api/health/live
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 3

readinessProbe:
  httpGet:
    path: /api/health/ready
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 10
  timeoutSeconds: 5
```

## 5. Security & Isolation
- Root filesystem is read-only except for ephemeral memory buffers.
- Model checkpoints are verified at startup against immutable cryptographic SHA-256 digests.
- Ephemeral audio streams reside strictly in volatile memory and are purged upon session teardown.
