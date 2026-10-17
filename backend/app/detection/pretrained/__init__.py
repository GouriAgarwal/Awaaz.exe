"""Pretrained speech anti-spoofing detector integration package."""

from app.detection.pretrained.aasist_architecture import AASISTModel
from app.detection.pretrained.adapter import AASISTPretrainedAdapter
from app.detection.pretrained.loader import (
    CHECKPOINT_PATH,
    EXPECTED_PARAM_COUNT,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    PretrainedModelIntegrityError,
    PretrainedModelLoader,
)
from app.detection.pretrained.provenance import get_pretrained_aasist_provenance

__all__ = [
    "AASISTModel",
    "AASISTPretrainedAdapter",
    "PretrainedModelLoader",
    "PretrainedModelIntegrityError",
    "get_pretrained_aasist_provenance",
    "CHECKPOINT_PATH",
    "EXPECTED_SHA256",
    "EXPECTED_SIZE_BYTES",
    "EXPECTED_PARAM_COUNT",
]
