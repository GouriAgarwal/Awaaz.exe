"""Model loading, device selection, and weight management for voice clone detectors."""

import os
from pathlib import Path
from typing import Optional, Any
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("detection.model_loader")


class ModelLoader:
    """Safely loads ML model weights and selects optimal compute device (CPU/CUDA)."""

    def __init__(self, weights_dir: Optional[Path] = None):
        self.weights_dir = weights_dir or Path(os.path.dirname(__file__)) / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._device: Optional[str] = None
        self._torch_available: Optional[bool] = None

    @property
    def is_torch_available(self) -> bool:
        """Check whether PyTorch runtime is installed and importable."""
        if self._torch_available is None:
            try:
                import torch
                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available

    def get_device(self) -> str:
        """Resolve compute device (CUDA if configured & available, else CPU)."""
        if self._device is not None:
            return self._device

        if not self.is_torch_available:
            self._device = "cpu"
            return self._device

        import torch
        if settings.DETECTION_DEVICE.lower() == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
            logger.info(f"CUDA accelerator detected: {torch.cuda.get_device_name(0)}")
        else:
            self._device = "cpu"
            logger.info("Using CPU device for synthetic voice detection inference")

        return self._device

    def load_pytorch_model(self, model_class: Any, weight_filename: Optional[str] = None) -> Any:
        """Instantiate model class and load weights if available.
        
        Args:
            model_class: PyTorch nn.Module class to instantiate.
            weight_filename: Optional filename in weights directory.
            
        Returns:
            Instantiated and configured model instance, moved to device and eval mode.
        """
        if not self.is_torch_available:
            raise RuntimeError("PyTorch runtime is not installed. Cannot load PyTorch model.")

        import torch

        device = self.get_device()
        model = model_class()

        if weight_filename:
            weight_path = self.weights_dir / weight_filename
            if weight_path.exists():
                logger.info(f"Loading checkpoint weights from {weight_path}")
                state_dict = torch.load(str(weight_path), map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
            else:
                logger.warning(
                    f"Weight file {weight_path} not found. Initializing model with baseline parameters."
                )

        model.to(device)
        model.eval()
        return model
