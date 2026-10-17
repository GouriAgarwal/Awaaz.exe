# ==============================================================================
# VoiceShield Production Dockerfile
# Multi-stage build with security hardening (non-root execution, minimal attack surface)
# ==============================================================================

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies (libsndfile for audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt requirements-production.txt ./

# Build wheel dependencies into local wheels cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements-production.txt

# Stage 2: Runtime Minimal Image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime shared libraries only (libsndfile1, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged service user and group
RUN groupadd -g 10001 voiceshield && \
    useradd -u 10001 -g voiceshield -m -s /bin/bash voiceshield

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/voiceshield/.local
ENV PATH=/home/voiceshield/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application source code
COPY --chown=voiceshield:voiceshield app/ /app/app/

# Ensure proper permissions
RUN chown -R voiceshield:voiceshield /app

# Drop privileges to non-root user
USER voiceshield

# Environment configuration
ENV HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production \
    PYTHONUNBUFFERED=1 \
    METRICS_ENABLED=true \
    STRUCTURED_LOGGING_ENABLED=true

# Expose HTTP / WebSocket port
EXPOSE 8000

# Docker Healthcheck targeting the non-blocking liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/live || exit 1

# Production entrypoint with uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --access-log"]
