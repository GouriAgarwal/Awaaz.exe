"""Global exception handlers for consistent JSON error responses with operational taxonomy."""

import time
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.observability.errors import (
    ErrorCode,
    OperationalException,
    error_tracker,
)
from app.observability.tracing import get_request_id

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers with the FastAPI application."""

    @app.exception_handler(OperationalException)
    async def operational_exception_handler(request: Request, exc: OperationalException) -> JSONResponse:
        request_id = get_request_id() or exc.request_id
        error_tracker.record_error(
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
            details=exc.details,
        )
        logger.warning(
            "Operational exception: code=%s message=%s path=%s request_id=%s",
            exc.error_code.value,
            exc.message,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code.value,
                "message": exc.message,
                "request_id": request_id,
                "timestamp": time.time(),
                "details": exc.details if exc.details else None,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = get_request_id()
        # Map common HTTP codes to ErrorCodes
        code_map = {
            400: ErrorCode.AUDIO_INVALID,
            404: ErrorCode.SESSION_NOT_FOUND,
            429: ErrorCode.STREAM_RATE_LIMITED,
            503: ErrorCode.MODEL_UNAVAILABLE,
        }
        err_code = code_map.get(exc.status_code, ErrorCode.INTERNAL_PROCESSING_ERROR)
        error_tracker.record_error(
            error_code=err_code,
            message=str(exc.detail),
            request_id=request_id,
        )
        logger.warning(
            "HTTP error occurred: status_code=%s detail=%s path=%s request_id=%s",
            exc.status_code,
            exc.detail,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "path": request.url.path,
                },
                "error_code": err_code.value,
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = get_request_id()
        error_tracker.record_error(
            error_code=ErrorCode.AUDIO_INVALID,
            message="Request validation error",
            request_id=request_id,
            details={"errors": exc.errors()},
        )
        logger.warning(
            "Validation error occurred: path=%s errors=%s request_id=%s",
            request.url.path,
            exc.errors(),
            request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": "Validation Error",
                    "details": exc.errors(),
                    "path": request.url.path,
                },
                "error_code": ErrorCode.AUDIO_INVALID.value,
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = get_request_id()
        error_tracker.record_error(
            error_code=ErrorCode.INTERNAL_PROCESSING_ERROR,
            message=f"Unhandled internal error: {type(exc).__name__}",
            request_id=request_id,
        )
        # Suppress sensitive traceback details from user response
        logger.error(
            "Unhandled exception processing request to %s: %s [request_id=%s]",
            request.url.path,
            str(exc),
            request_id,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "Internal server error occurred.",
                    "path": request.url.path,
                },
                "error_code": ErrorCode.INTERNAL_PROCESSING_ERROR.value,
                "message": "Internal server error occurred.",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

