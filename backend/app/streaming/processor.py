"""
Streaming Audio Processor.
Executes rolling window analysis with AASIST synthetic detection, ECAPA-TDNN speaker verification, and Step 9 fusion.
"""
import time
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from app.audio.schemas import AudioData
from app.detection.registry import DetectorRegistry
from app.speaker_verification.registry import SpeakerVerificationService
from app.risk.fusion import ControlledImpersonationRiskEngine
from app.streaming.schemas import (
    StreamEvent,
    StreamEventType,
    WindowTimingInfo,
    StreamingSyntheticSignal,
    StreamingSpeakerSignal,
    RollingRiskMetrics,
)

logger = logging.getLogger(__name__)


class StreamingAudioProcessor:
    """
    Evaluates individual 3.0s analysis windows and maintains rolling risk statistics.
    """
    def __init__(
        self,
        session_id: str,
        enrolled_profile_id: Optional[str] = None,
        detector_registry: Optional[DetectorRegistry] = None,
        speaker_service: Optional[SpeakerVerificationService] = None,
    ):
        self.session_id = session_id
        self.enrolled_profile_id = enrolled_profile_id
        self.detector_registry = detector_registry or DetectorRegistry()
        self.speaker_service = speaker_service or SpeakerVerificationService.get_instance()

        # Rolling risk state
        self.risk_scores_history: List[float] = []
        self.previous_risk_level: Optional[str] = None
        self.latest_risk_score: float = 0.0
        self.latest_risk_level: str = "LOW"
        self.latest_action: str = "ALLOW"
        self.consecutive_high_risk_windows: int = 0
        self.consecutive_critical_windows: int = 0

    def process_window(
        self,
        window_index: int,
        start_sec: float,
        end_sec: float,
        audio_data: AudioData,
        stream_duration_sec: float,
    ) -> List[StreamEvent]:
        """
        Runs multi-modal inference on a 3.0-second AudioData chunk and returns generated events.
        """
        window_start_time = time.perf_counter()
        events: List[StreamEvent] = []

        # 1. Step 6 Pretrained AASIST Synthetic Voice Detection
        synth_start_t = time.perf_counter()
        detector = self.detector_registry.get_detector()
        detection_result = detector.predict(audio_data)
        synth_latency_ms = (time.perf_counter() - synth_start_t) * 1000.0

        meta = detector.metadata() if callable(detector.metadata) else detector.metadata
        synth_detector_id = getattr(meta, "model_name", "VoiceShield-AASIST-Pretrained-v1")
        synth_class_str = (
            detection_result.classification.value
            if hasattr(detection_result.classification, "value")
            else str(detection_result.classification)
        )

        synth_signal = StreamingSyntheticSignal(
            score=detection_result.score,
            score_type="uncalibrated_model_score",
            classification=synth_class_str,
            confidence_band=detection_result.confidence_band,
            detector_id=synth_detector_id,
            latency_ms=round(synth_latency_ms, 2),
        )

        # 2. Step 8 Pretrained ECAPA-TDNN Speaker Identity Verification
        spk_similarity: float = 0.0
        spk_decision_str: str = "UNCERTAIN"
        spk_conf_str: str = "LOW"
        spk_model_id: str = "speechbrain_ecapa_tdnn_voxceleb"
        spk_latency_ms: float = 0.0
        spk_status: str = "NOT_AVAILABLE"

        if self.enrolled_profile_id:
            spk_start_t = time.perf_counter()
            try:
                ver_res = self.speaker_service.verify(
                    profile_id=self.enrolled_profile_id,
                    audio_data=audio_data,
                )
                spk_similarity = ver_res.similarity_score
                spk_decision_str = (
                    ver_res.decision.value
                    if hasattr(ver_res.decision, "value")
                    else str(ver_res.decision)
                )
                spk_conf_str = (
                    ver_res.confidence_band.value
                    if hasattr(ver_res.confidence_band, "value")
                    else str(ver_res.confidence_band)
                )
                spk_model_id = ver_res.model_id
                spk_status = spk_decision_str
                spk_latency_ms = (time.perf_counter() - spk_start_t) * 1000.0
            except KeyError:
                logger.warning(f"Enrolled profile '{self.enrolled_profile_id}' not found during streaming.")
                spk_status = "PROFILE_NOT_FOUND"
            except Exception as e:
                logger.error(f"Speaker verification error during streaming: {e}")
                spk_status = "ERROR"

        spk_signal = StreamingSpeakerSignal(
            status=spk_status,
            similarity_score=spk_similarity if self.enrolled_profile_id else None,
            confidence_band=spk_conf_str if self.enrolled_profile_id else None,
            profile_id=self.enrolled_profile_id,
            model_id=spk_model_id if self.enrolled_profile_id else None,
            latency_ms=round(spk_latency_ms, 2) if self.enrolled_profile_id else None,
        )

        # 3. Audio Quality Check
        rms = float(np.sqrt(np.mean(audio_data.samples ** 2)))
        quality_status = "good" if rms > 0.01 else "degraded"

        # 4. Step 9 Controlled Impersonation Risk Fusion
        total_compute_latency_ms = (time.perf_counter() - window_start_time) * 1000.0

        risk_result = ControlledImpersonationRiskEngine.analyze_signals(
            synth_score=synth_signal.score,
            synth_classification=synth_signal.classification,
            synth_confidence=synth_signal.confidence_band,
            synth_detector_id=synth_signal.detector_id,
            spk_similarity=spk_similarity,
            spk_decision=spk_decision_str,
            spk_confidence=spk_conf_str,
            spk_profile_id=self.enrolled_profile_id or "unspecified",
            spk_model_id=spk_model_id,
            duration_seconds=audio_data.duration_seconds,
            audio_quality_status=quality_status,
            contextual_signals=None,
            latency_ms=total_compute_latency_ms,
        )

        # 5. Update Rolling State
        current_risk_score = risk_result.risk_score
        current_risk_level = risk_result.risk_level.value if hasattr(risk_result.risk_level, "value") else str(risk_result.risk_level)
        current_action = risk_result.recommended_action.value if hasattr(risk_result.recommended_action, "value") else str(risk_result.recommended_action)

        self.risk_scores_history.append(current_risk_score)
        rolling_max = float(np.max(self.risk_scores_history))
        rolling_mean = float(np.mean(self.risk_scores_history))

        if current_risk_level == "CRITICAL":
            self.consecutive_critical_windows += 1
            self.consecutive_high_risk_windows += 1
        elif current_risk_level == "HIGH":
            self.consecutive_high_risk_windows += 1
            self.consecutive_critical_windows = 0
        else:
            self.consecutive_high_risk_windows = 0
            self.consecutive_critical_windows = 0

        rolling_metrics = RollingRiskMetrics(
            latest_risk_score=round(current_risk_score, 1),
            latest_risk_level=current_risk_level,
            latest_action=current_action,
            previous_risk_score=round(self.latest_risk_score, 1) if self.risk_scores_history else None,
            rolling_max_risk=round(rolling_max, 1),
            rolling_mean_risk=round(rolling_mean, 1),
            analyzed_windows_count=len(self.risk_scores_history),
            consecutive_high_risk_windows=self.consecutive_high_risk_windows,
            consecutive_critical_windows=self.consecutive_critical_windows,
        )

        timing_info = WindowTimingInfo(
            window_index=window_index,
            window_start_sec=round(start_sec, 2),
            window_end_sec=round(end_sec, 2),
            duration_sec=round(audio_data.duration_seconds, 2),
        )

        # Compute Processing Lag (difference between stream time and analyzed window end time)
        processing_lag_ms = max(0.0, (stream_duration_sec - end_sec) * 1000.0)

        # Main completion event
        evidence_dicts = [
            {
                "code": e.code,
                "severity": e.severity.value if hasattr(e.severity, "value") else str(e.severity),
                "message": e.message,
            }
            for e in risk_result.evidence
        ]

        action_reason_str = getattr(risk_result, "action_reason", getattr(risk_result, "action_rationale", ""))

        main_event = StreamEvent(
            event_type=StreamEventType.ANALYSIS_WINDOW_COMPLETED,
            session_id=self.session_id,
            timestamp=time.time(),
            message=f"Analysis window #{window_index} ({start_sec:.1f}s - {end_sec:.1f}s) evaluated.",
            timing=timing_info,
            synthetic_signal=synth_signal,
            speaker_signal=spk_signal,
            risk_score=round(current_risk_score, 1),
            risk_level=current_risk_level,
            recommended_action=current_action,
            action_rationale=action_reason_str,
            evidence=evidence_dicts,
            rolling_metrics=rolling_metrics,
            processing_latency_ms=round(total_compute_latency_ms, 2),
            processing_lag_ms=round(processing_lag_ms, 2),
            audio_duration_received_sec=round(stream_duration_sec, 2),
        )
        events.append(main_event)

        # Secondary threshold trigger events
        if self.previous_risk_level is not None and self.previous_risk_level != current_risk_level:
            events.append(
                StreamEvent(
                    event_type=StreamEventType.RISK_LEVEL_CHANGED,
                    session_id=self.session_id,
                    timestamp=time.time(),
                    message=f"Risk level transitioned from {self.previous_risk_level} to {current_risk_level}.",
                    risk_score=round(current_risk_score, 1),
                    risk_level=current_risk_level,
                    recommended_action=current_action,
                    rolling_metrics=rolling_metrics,
                )
            )

        if current_risk_level == "CRITICAL":
            events.append(
                StreamEvent(
                    event_type=StreamEventType.CRITICAL_RISK_DETECTED,
                    session_id=self.session_id,
                    timestamp=time.time(),
                    message=f"CRITICAL impersonation risk detected (Score: {current_risk_score:.1f}/100)!",
                    risk_score=round(current_risk_score, 1),
                    risk_level=current_risk_level,
                    recommended_action=current_action,
                    rolling_metrics=rolling_metrics,
                )
            )
        elif current_risk_level == "HIGH":
            events.append(
                StreamEvent(
                    event_type=StreamEventType.HIGH_RISK_DETECTED,
                    session_id=self.session_id,
                    timestamp=time.time(),
                    message=f"High synthetic speech risk detected (Score: {current_risk_score:.1f}/100).",
                    risk_score=round(current_risk_score, 1),
                    risk_level=current_risk_level,
                    recommended_action=current_action,
                    rolling_metrics=rolling_metrics,
                )
            )

        if quality_status == "degraded":
            events.append(
                StreamEvent(
                    event_type=StreamEventType.AUDIO_QUALITY_DEGRADED,
                    session_id=self.session_id,
                    timestamp=time.time(),
                    message="Degraded audio quality detected (low SNR / near silence).",
                    risk_level=current_risk_level,
                    rolling_metrics=rolling_metrics,
                )
            )

        self.latest_risk_score = current_risk_score
        self.latest_risk_level = current_risk_level
        self.latest_action = current_action
        self.previous_risk_level = current_risk_level

        return events
