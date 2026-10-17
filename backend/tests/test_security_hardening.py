"""Automated Security Hardening, Checksum Integrity, and Privacy Tests (Phase 11H)."""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.decoder import AudioDecoder
from app.audio.normalizer import AudioNormalizer
from app.core.config import settings
from app.detection.pretrained.loader import PretrainedModelIntegrityError, PretrainedModelLoader
from app.evaluation.security_audit import SecurityAuditor
from app.main import app
from app.streaming.session import StreamingSession

client = TestClient(app)


class TestModelIntegrityAndTamperResistance:
    """Tests for cryptographic checkpoint validation and tamper detection."""

    def test_official_aasist_sha256_verification(self):
        is_valid, actual_hash, size = PretrainedModelLoader.verify_checkpoint_integrity()
        assert is_valid is True
        assert actual_hash == "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
        assert size == 1281532

    def test_tampered_checkpoint_rejection(self):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"TAMPERED_OR_CORRUPT_MODEL_WEIGHTS")
            f.flush()
            corrupted_path = f.name

        try:
            is_valid, actual_hash, _ = PretrainedModelLoader.verify_checkpoint_integrity(corrupted_path)
            assert is_valid is False
            assert actual_hash != "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
        finally:
            if os.path.exists(corrupted_path):
                os.remove(corrupted_path)

    def test_security_auditor_reports_valid_models(self):
        audits = SecurityAuditor.audit_model_integrity()
        assert len(audits) >= 2
        for audit in audits:
            assert audit.integrity_verified is True
            assert audit.status == "VALIDATED"


class TestEphemeralAudioAndPrivacyGuarantees:
    """Tests guaranteeing raw audio is ephemeral and never persisted to disk."""

    def test_streaming_session_lifecycle_does_not_create_disk_files(self):
        import asyncio

        async def _run():
            session = StreamingSession(session_id="privacy_test_session")
            await session.start()

            # Push 10 chunks of audio
            dummy_chunk = np.random.uniform(-0.1, 0.1, 1600).astype(np.float32)
            for i in range(10):
                session.buffer.push_chunk(dummy_chunk, seq=i)

            # Confirm buffer is in RAM
            assert session.buffer.current_sample_count == 16000

            # Terminate session
            await session.stop()
            assert session.state.value == "COMPLETED"

            # Verify no file with session_id exists on disk
            root_dir = Path(__file__).resolve().parents[2]
            matching_files = list(root_dir.glob(f"**/*{session.session_id}*"))
            assert len(matching_files) == 0

        asyncio.run(_run())

    def test_session_summary_never_exposes_embeddings(self):
        session = StreamingSession(session_id="embedding_privacy_test")
        summary = session.get_summary()
        summary_dict = summary.model_dump()
        assert "embedding" not in summary_dict
        assert "raw_audio" not in summary_dict


class TestApiInputBoundariesAndSanitization:
    """Tests API boundaries, duration constraints, and numerical safety."""

    def test_empty_audio_upload_rejected(self):
        files = {"file": ("empty.wav", b"", "audio/wav")}
        resp = client.post("/api/audio/inspect", files=files)
        assert resp.status_code in [400, 415, 422]

    def test_malformed_audio_bytes_rejected_safely(self):
        files = {"file": ("corrupt.wav", b"NOT_A_VALID_WAV_HEADER", "audio/wav")}
        resp = client.post("/api/audio/inspect", files=files)
        assert resp.status_code in [400, 415, 422]
        data = resp.json()
        # Ensure internal system paths are not leaked in error message
        assert "C:\\" not in str(data)
        assert "/home/" not in str(data)

