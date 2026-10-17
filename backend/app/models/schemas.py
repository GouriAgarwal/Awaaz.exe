"""Pydantic schemas for request and response validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health endpoint response."""

    status: str = Field(..., description="Overall service status", examples=["ok"])
    service: str = Field(..., description="Service name", examples=["VoiceShield API"])
    version: str = Field(..., description="Application version", examples=["0.1.0"])


class SystemStatusResponse(BaseModel):
    """Schema for system status endpoint response."""

    backend: str = Field(..., description="Backend operational state", examples=["online"])
    version: str = Field(..., description="Application version", examples=["0.1.0"])
    environment: str = Field(..., description="Current deployment environment", examples=["development"])
    detector_available: bool = Field(default=True, description="Whether synthetic voice detector is active")
    active_detector: Optional[str] = Field(default=None, description="Active synthetic voice detector name")
    torch_available: Optional[bool] = Field(default=None, description="Whether PyTorch ML runtime is active")


class ErrorDetail(BaseModel):
    """Schema for detailed error payload."""

    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error description")
    path: Optional[str] = Field(None, description="Request path that generated the error")
    details: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed validation error list")


class ErrorResponse(BaseModel):
    """Wrapper schema for error responses."""

    error: ErrorDetail
