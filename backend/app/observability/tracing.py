"""Request Correlation and Tracing Middleware (Step 12)."""

import contextvars
import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.metrics import MetricsRegistry

# Context variable preserving request correlation ID throughout request async context
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def get_current_request_id() -> str:
    """Return active request correlation ID or generate fallback."""
    rid = request_id_ctx.get()
    return rid or f"req_{uuid.uuid4().hex[:12]}"


def get_request_id() -> Optional[str]:
    """Return active request correlation ID if available."""
    return request_id_ctx.get()



class TracingMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware enforcing X-Request-ID propagation and request duration tracking."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract existing X-Request-ID or generate new unique ID
        incoming_id = request.headers.get("X-Request-ID")
        request_id = incoming_id if incoming_id and len(incoming_id) <= 64 else f"req_{uuid.uuid4().hex[:12]}"
        
        token = request_id_ctx.set(request_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Record in metrics registry
            metrics = MetricsRegistry()
            metrics.record_http_request(status_code=response.status_code, latency_ms=latency_ms)

            # Attach correlation ID and processing duration to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
            return response
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics = MetricsRegistry()
            metrics.record_http_request(status_code=500, latency_ms=latency_ms)
            raise exc
        finally:
            request_id_ctx.reset(token)
