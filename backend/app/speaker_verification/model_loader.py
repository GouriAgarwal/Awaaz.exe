"""
Pretrained ECAPA-TDNN Model Loader for Speaker Verification.
Handles cryptographic verification, automatic acquisition, and strict PyTorch loading.
"""
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional
import torch

from app.speaker_verification.architecture import ECAPA_TDNN

logger = logging.getLogger(__name__)

EXPECTED_SHA256 = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "ecapa_tdnn.ckpt"
HF_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
HF_FILENAME = "embedding_model.ckpt"


class PretrainedSpeakerLoader:
    """
    Cryptographically verifies and instantiates the SpeechBrain ECAPA-TDNN model.
    """

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @classmethod
    def ensure_weights(cls, weights_path: Optional[Path] = None) -> Path:
        """
        Ensures the checkpoint weights file is available locally and valid.
        """
        path = weights_path or DEFAULT_WEIGHTS_PATH
        path = Path(path)

        if path.exists() and path.stat().st_size > 0:
            actual_sha256 = cls.compute_sha256(path)
            if actual_sha256 == EXPECTED_SHA256:
                logger.info(f"Verified speaker weights at {path} (SHA-256 match)")
                return path
            else:
                logger.warning(
                    f"Corrupted or mismatched weights at {path}. Expected {EXPECTED_SHA256}, got {actual_sha256}. Re-acquiring..."
                )

        # Fallback: attempt download from HuggingFace Hub
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"Downloading {HF_REPO_ID}/{HF_FILENAME} via HuggingFace Hub...")
            downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
            import shutil
            shutil.copyfile(downloaded, path)
        except Exception as e:
            logger.error(f"Failed to acquire weights from HuggingFace Hub: {e}")
            raise FileNotFoundError(
                f"ECAPA-TDNN checkpoint not found at {path} and could not be downloaded from {HF_REPO_ID}. Error: {e}"
            )

        actual_sha256 = cls.compute_sha256(path)
        if actual_sha256 != EXPECTED_SHA256:
            raise ValueError(
                f"Cryptographic SHA-256 verification failed for {path}! Expected {EXPECTED_SHA256}, got {actual_sha256}."
            )

        return path

    @classmethod
    def load_model(
        cls,
        weights_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> ECAPA_TDNN:
        """
        Loads the ECAPA_TDNN PyTorch model strictly in eval mode on target device.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = cls.ensure_weights(weights_path)
        model = ECAPA_TDNN(in_channels=80, channels=1024, lin_neurons=192)

        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            model.to(device)
            model.eval()
            logger.info(f"Loaded SpeechBrain ECAPA-TDNN on {device} ({sum(p.numel() for p in model.parameters())} params)")
            return model
        except Exception as e:
            logger.error(f"Failed to load state dict from {path}: {e}")
            raise
