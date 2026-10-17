"""Tests for application startup and factory creation."""

from app.main import create_application


def test_create_application_instance() -> None:
    """Application factory should instantiate FastAPI with correct metadata."""
    app_instance = create_application()
    assert app_instance.title == "VoiceShield API"
    assert app_instance.version == "0.1.0"
    assert app_instance.docs_url == "/docs"
