"""Adapter connecting official pretrained AASIST model to the VoiceShield detection pipeline."""

import time
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from app.audio.schemas import AudioData
from app.core.config import settings
from app.core.logging import get_logger
from app.detection.base import BaseSyntheticVoiceDetector
from app.detection.calibration import ScoreCalibrator
from app.detection.metadata import DetectorMetadata
from app.detection.pretrained.loader import (
    CHECKPOINT_PATH,
    PretrainedModelIntegrityError,
    PretrainedModelLoader,
)
from app.detection.preprocessing import AudioWindowPreprocessor
from app.detection.schemas import (
    ClassificationLabel,
    DetectionResult,
    ScoreType,
    WindowScore,
)

logger = get_logger("detection.pretrained.adapter")


class AASISTPretrainedAdapter(BaseSyntheticVoiceDetector):
    """Production adapter for official pretrained AASIST anti-spoofing model."""

    TARGET_SAMPLES = 64600  # ~4.0375 seconds at 16 kHz (official AASIST input window)

    def __init__(
        self,
        device: Optional[str] = None,
        threshold_natural: float = settings.DETECTION_THRESHOLD_NATURAL,
        threshold_synthetic: float = settings.DETECTION_THRESHOLD_SYNTHETIC,
        aggregation_method: str = settings.DETECTION_AGGREGATION_METHOD,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregation_method = aggregation_method
        self._is_loaded = False
        self._model = None

        self.preprocessor = AudioWindowPreprocessor(
            window_size_sec=settings.DETECTION_WINDOW_SECONDS,
            window_hop_sec=settings.DETECTION_HOP_SECONDS,
            sample_rate=settings.TARGET_SAMPLE_RATE,
        )

        self.calibrator = ScoreCalibrator(
            threshold_natural=threshold_natural,
            threshold_synthetic=threshold_synthetic,
            score_type=ScoreType.UNCALIBRATED_MODEL_SCORE,
        )

    def load(self) -> None:
        """Load and verify genuine pretrained AASIST model checkpoint."""
        if self._is_loaded and self._model is not None:
            return

        try:
            self._model = PretrainedModelLoader.load_pretrained_aasist(device=self.device)
            self._is_loaded = True
            logger.info("AASISTPretrainedAdapter loaded and verified successfully.")
        except Exception as e:
            self._is_loaded = False
            self._model = None
            logger.error(f"Failed to load pretrained AASIST model: {e}")
            raise

    def is_loaded(self) -> bool:
        return self._is_loaded and (self._model is not None)

    def metadata(self) -> DetectorMetadata:
        """Return architectural and provenance metadata."""
        is_valid, sha256_hash, _ = PretrainedModelLoader.verify_checkpoint_integrity()
        hash_str = sha256_hash[:16] + "..." if is_valid else "UNVERIFIED"

        return DetectorMetadata(
            model_name="VoiceShield-AASIST-Pretrained-v1",
            model_type="pretrained_graph_attention",
            architecture="AASIST (Automated Anti-Spoofing Integration with Integrated Spectro-Temporal Graph Attention)",
            checkpoint_or_source=f"AASIST.pth (SHA-256: {hash_str})",
            license="BSD-3-Clause / MIT (NAVER Corp / Clova AI)",
            expected_sample_rate=settings.TARGET_SAMPLE_RATE,
            window_size_sec=settings.DETECTION_WINDOW_SECONDS,
            window_hop_sec=settings.DETECTION_HOP_SECONDS,
            score_type=ScoreType.UNCALIBRATED_MODEL_SCORE,
            score_interpretation=(
                "Continuous spoof posterior probability from softmax output: softmax(logits)[1]. "
                "Scores > 0.65 indicate synthetic/cloned speech evidence; scores < 0.35 indicate natural human speech."
            ),
            scientific_disclaimer=(
                "Genuine official pretrained AASIST checkpoint. Model scores represent uncalibrated neural softmax outputs. "
                "The presence of a pretrained checkpoint does not mean VoiceShield has independently validated the model on local benchmarks."
            ),
            device=self.device,
            is_fallback=False,
        )

    def _prepare_window_tensor(self, chunk: np.ndarray) -> torch.Tensor:
        """Format 16 kHz audio chunk into exact 64,600 samples expected by AASIST."""
        chunk_len = len(chunk)
        if chunk_len < self.TARGET_SAMPLES:
            # Repeat audio to reach target length (standard AASIST test protocol)
            repeat_factor = int(np.ceil(self.TARGET_SAMPLES / chunk_len))
            repeated = np.tile(chunk, repeat_factor)[:self.TARGET_SAMPLES]
            tensor_data = repeated
        else:
            tensor_data = chunk[:self.TARGET_SAMPLES]

        tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, audio_data: AudioData) -> DetectionResult:
        """Execute windowed inference on standardized 16 kHz mono AudioData."""
        if not self._is_loaded or self._model is None:
            self.load()

        start_time = time.perf_counter()
        windows = self.preprocessor.process(audio_data)

        window_scores: List[WindowScore] = []
        scores_list: List[float] = []

        with torch.no_grad():
            for window_idx, start_sec, end_sec, chunk in windows:
                tensor = self._prepare_window_tensor(chunk)
                _, logits = self._model(tensor)

                # AASIST output logits shape: (1, 2) -> [bonafide_logit, spoof_logit]
                probs = F.softmax(logits, dim=1)
                spoof_prob = float(probs[0, 1].item())
                raw_logit_diff = float((logits[0, 1] - logits[0, 0]).item())

                label, _ = self.calibrator.classify_score(spoof_prob)

                window_scores.append(
                    WindowScore(
                        window_index=window_idx,
                        start_sec=round(start_sec, 3),
                        end_sec=round(end_sec, 3),
                        raw_score=round(raw_logit_diff, 4),
                        synthetic_score=round(spoof_prob, 4),
                        label=label,
                    )
                )
                scores_list.append(spoof_prob)

        # Score aggregation across windows
        if not scores_list:
            agg_score = 0.5
        elif self.aggregation_method == "mean":
            agg_score = float(np.mean(scores_list))
        else:  # default median
            agg_score = float(np.median(scores_list))

        final_label, confidence_band = self.calibrator.classify_score(agg_score)
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            detector_metadata=self.metadata(),
            classification=final_label,
            score=round(agg_score, 4),
            score_type=ScoreType.UNCALIBRATED_MODEL_SCORE,
            confidence_band=confidence_band,
            thresholds_applied=self.calibrator.get_thresholds_dict(),
            window_scores=window_scores,
            aggregation_method=self.aggregation_method,
            inference_latency_ms=round(inference_latency_ms, 2),
            audio_duration_sec=round(audio_data.duration_seconds, 3),
            total_windows=len(window_scores),
            warnings=[],
        )
