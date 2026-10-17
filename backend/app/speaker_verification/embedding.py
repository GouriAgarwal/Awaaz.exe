"""
Speaker Embedding Generation using Pretrained SpeechBrain ECAPA-TDNN.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio

from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.model_loader import PretrainedSpeakerLoader, DEFAULT_WEIGHTS_PATH


class SpeechBrainECAPAEncoder(BaseSpeakerEncoder):
    """
    Extracts 192-dimensional L2-normalized speaker embeddings using SpeechBrain ECAPA-TDNN.
    """
    model_id: str = "speechbrain_ecapa_tdnn_voxceleb"
    model_name: str = "SpeechBrain ECAPA-TDNN (VoxCeleb)"
    embedding_dimension: int = 192
    expected_sample_rate: int = 16000
    is_l2_normalized: bool = True

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PretrainedSpeakerLoader.load_model(weights_path=weights_path, device=self.device)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.expected_sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        ).to(self.device)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Computes 80-dim log Mel-filterbanks with sentence-level mean normalization.
        Args:
            waveform: Tensor of shape (1, Time) on self.device
        Returns:
            Tensor of shape (1, 80, Frames)
        """
        mels = self.mel_transform(waveform)
        log_mels = torch.log(mels + 1e-6)
        # Sentence mean normalization across time
        feat = log_mels - log_mels.mean(dim=-1, keepdim=True)
        return feat

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts 192-dimensional L2-normalized speaker embedding.
        Args:
            audio: 1D float32 numpy array normalized to [-1.0, 1.0]
            sample_rate: Sample rate in Hz (must be 16000)
        Returns:
            1D float32 numpy array of shape (192,), L2-normalized
        """
        if audio is None or len(audio) == 0:
            raise ValueError("Audio waveform cannot be empty.")

        if sample_rate != self.expected_sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.expected_sample_rate}, got {sample_rate}.")

        # Ensure 1D float32
        audio_flat = np.asarray(audio, dtype=np.float32).flatten()
        if not np.all(np.isfinite(audio_flat)):
            raise ValueError("Audio waveform contains NaN or infinite values.")

        # Minimum length handling (at least 400 samples = 25ms for 1 STFT frame)
        if len(audio_flat) < 400:
            # Pad with zeroes up to 400 samples
            audio_flat = np.pad(audio_flat, (0, 400 - len(audio_flat)), mode="constant")

        waveform_tensor = torch.from_numpy(audio_flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.extract_features(waveform_tensor)
            raw_emb = self.model(feat) # shape (1, 192)

            # L2 Normalization
            norm = torch.norm(raw_emb, p=2, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-12)
            normalized_emb = raw_emb / norm

            emb_np = normalized_emb.squeeze(0).cpu().numpy().astype(np.float32)

        # Final sanity check
        if not np.all(np.isfinite(emb_np)):
            raise ValueError("Computed speaker embedding contains NaN or Inf.")

        return emb_np
