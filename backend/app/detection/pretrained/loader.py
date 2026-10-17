"""Pretrained model checkpoint loader, integrity verifier, and device manager."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import urllib.request
import torch

from app.core.logging import get_logger
from app.detection.pretrained.aasist_architecture import AASISTModel, get_default_aasist_config

logger = get_logger("detection.pretrained.loader")

OFFICIAL_AASIST_URL = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth"
EXPECTED_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
AASIST_EXPECTED_SHA256 = EXPECTED_SHA256
EXPECTED_SIZE_BYTES = 1281532
EXPECTED_PARAM_COUNT = 297866
AASIST_EXPECTED_PARAMS = EXPECTED_PARAM_COUNT

WEIGHTS_DIR = Path(os.path.dirname(__file__)).parent / "weights"
CHECKPOINT_PATH = WEIGHTS_DIR / "AASIST.pth"


class PretrainedModelIntegrityError(Exception):
    """Raised when checkpoint weights are missing, corrupted, or fail cryptographic verification."""
    pass


class PretrainedModelLoader:
    """Safely verifies and loads genuine pretrained voice clone detection checkpoints."""

    @classmethod
    def get_checkpoint_path(cls) -> Path:
        """Return the default path to the AASIST checkpoint file."""
        return CHECKPOINT_PATH

    @staticmethod
    def compute_file_sha256(filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def verify_checkpoint_integrity(cls, filepath: Optional[Union[Path, str]] = None) -> Tuple[bool, str, int]:
        """Verify checkpoint file exists, is non-empty, and matches recorded SHA-256 hash.
        
        Returns:
            Tuple of (is_valid: bool, sha256_hash: str, file_size_bytes: int)
        """
        path = Path(filepath) if filepath else CHECKPOINT_PATH
        if not path.exists():
            return False, "missing", 0

        size = os.path.getsize(path)
        if size == 0:
            return False, "empty_file", 0

        actual_hash = cls.compute_file_sha256(path)
        if actual_hash.lower() != EXPECTED_SHA256.lower():
            logger.warning(
                f"Checkpoint SHA-256 hash mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
            )
            return False, actual_hash, size

        return True, actual_hash, size

    @classmethod
    def ensure_checkpoint(cls) -> Path:
        """Ensure official AASIST checkpoint is downloaded and cryptographically verified."""
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        if not CHECKPOINT_PATH.exists() or os.path.getsize(CHECKPOINT_PATH) == 0:
            logger.info(f"Downloading official AASIST checkpoint from {OFFICIAL_AASIST_URL}...")
            try:
                req = urllib.request.Request(OFFICIAL_AASIST_URL, headers={"User-Agent": "VoiceShield-App/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(CHECKPOINT_PATH, "wb") as f:
                    f.write(resp.read())
                logger.info("AASIST checkpoint downloaded successfully.")
            except Exception as e:
                raise PretrainedModelIntegrityError(
                    f"Could not download official AASIST checkpoint from {OFFICIAL_AASIST_URL}: {e}"
                )

        is_valid, actual_hash, size = cls.verify_checkpoint_integrity(CHECKPOINT_PATH)
        if not is_valid:
            raise PretrainedModelIntegrityError(
                f"Checkpoint validation failed: SHA256={actual_hash}, expected={EXPECTED_SHA256}"
            )

        return CHECKPOINT_PATH

    @classmethod
    def load_pretrained_aasist(cls, device: Optional[str] = None) -> AASISTModel:
        """Instantiate official AASIST architecture and load genuine pretrained weights.
        
        Args:
            device: Optional compute device ('cpu' or 'cuda').
            
        Returns:
            Loaded and verified AASISTModel instance in eval mode.
        """
        checkpoint_file = cls.ensure_checkpoint()

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading genuine pretrained AASIST model on device: {resolved_device}")

        model = AASISTModel(get_default_aasist_config())
        state_dict = torch.load(str(checkpoint_file), map_location=resolved_device, weights_only=False)

        # Strict state_dict load
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise PretrainedModelIntegrityError(f"AASIST state_dict mismatch: {e}")

        param_count = sum(p.numel() for p in model.parameters())
        if param_count != EXPECTED_PARAM_COUNT:
            raise PretrainedModelIntegrityError(
                f"Unexpected parameter count: got {param_count}, expected {EXPECTED_PARAM_COUNT}"
            )

        model.to(resolved_device)
        model.eval()
        logger.info(f"Pretrained AASIST model verified and loaded ({param_count:,} parameters).")
        return model

    @classmethod
    def load_aasist_model(cls, device: Optional[str] = None, download_if_missing: bool = True) -> AASISTModel:
        """Alias for load_pretrained_aasist."""
        return cls.load_pretrained_aasist(device=device)
