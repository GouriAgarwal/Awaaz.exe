"""
Prosody & Behavioral Analysis Engine.
Extracts fundamental frequency (F0), RMS dynamics, pause ratios, speech rhythm, and microvariations.
All processing is ephemeral and purely in-memory; raw audio is immediately released.
"""
import math
import logging
from typing import List, Tuple, Optional
import numpy as np

from app.prosody.schemas import (
    ProsodyClassification,
    ProsodySeverity,
    ProsodyEvidenceItem,
    ProsodyFeatureSummary,
    ProsodyAnalysisResult,
)

logger = logging.getLogger(__name__)


class ProsodyAnalyzer:
    """
    Deterministic, explainable acoustic prosody and behavioral dynamics analyzer.
    """
    SAMPLE_RATE = 16000
    WIN_SIZE = 400   # 25ms @ 16kHz
    HOP_SIZE = 160   # 10ms @ 16kHz
    PITCH_MIN = 50.0   # Hz
    PITCH_MAX = 500.0  # Hz
    VOICING_THRESHOLD = 0.40  # Normalized autocorrelation threshold
    MIN_DURATION_SEC = 0.50
    MIN_VOICED_FRAMES = 5

    @classmethod
    def analyze(cls, samples: np.ndarray, duration_seconds: Optional[float] = None) -> ProsodyAnalysisResult:
        """
        Executes complete prosody extraction and behavioral analysis on 16 kHz float32 audio.
        """
        evidence: List[ProsodyEvidenceItem] = []

        # 1. Sanitize input array
        if samples is None or len(samples) == 0:
            return cls._insufficient_result(
                reason="Audio buffer is empty.",
                duration=0.0,
                evidence_code="PROSODY_EMPTY_AUDIO",
            )

        # Check for NaN / Inf
        if not np.isfinite(samples).all():
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_NAN_SANITIZED",
                severity=ProsodySeverity.LOW,
                message="Non-finite audio samples were sanitized prior to prosodic feature extraction."
            ))

        total_samples = len(samples)
        calc_duration = total_samples / float(cls.SAMPLE_RATE)
        actual_duration = duration_seconds if duration_seconds is not None else calc_duration

        # 2. Check minimum duration
        if actual_duration < cls.MIN_DURATION_SEC:
            return cls._insufficient_result(
                reason=f"Audio duration ({actual_duration:.2f}s) is below minimum required 0.5s for prosody tracking.",
                duration=actual_duration,
                evidence_code="PROSODY_DURATION_TOO_SHORT",
            )

        # 3. Overall RMS and silence check
        overall_rms = float(np.sqrt(np.mean(samples**2)))
        if overall_rms < 0.001:
            return cls._insufficient_result(
                reason=f"Audio energy is near-silent (RMS {overall_rms:.5f} < 0.001). Cannot extract speech prosody.",
                duration=actual_duration,
                evidence_code="PROSODY_SILENCE_DETECTED",
            )

        # 4. Clipping check
        clip_count = int(np.sum(np.abs(samples) >= 0.999))
        clip_ratio = clip_count / float(total_samples)
        if clip_ratio > 0.05:
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_AUDIO_CLIPPING",
                severity=ProsodySeverity.MEDIUM,
                message=f"Severe digital clipping detected ({clip_ratio * 100:.1f}% samples). Harmonic tracking may be degraded."
            ))

        # 5. Frame-by-frame analysis
        total_frames = max(1, (total_samples - cls.WIN_SIZE) // cls.HOP_SIZE + 1)
        frame_rms_list: List[float] = []
        f0_list: List[Optional[float]] = []
        voiced_flags: List[bool] = []

        lag_min = int(cls.SAMPLE_RATE / cls.PITCH_MAX)  # 32
        lag_max = int(cls.SAMPLE_RATE / cls.PITCH_MIN)  # 320

        for i in range(total_frames):
            frame = samples[i * cls.HOP_SIZE : i * cls.HOP_SIZE + cls.WIN_SIZE]
            if len(frame) < cls.WIN_SIZE:
                frame = np.pad(frame, (0, cls.WIN_SIZE - len(frame)))

            rms = float(np.sqrt(np.mean(frame**2)))
            frame_rms_list.append(rms)

            if rms < 0.003:  # Unvoiced / silence frame
                f0_list.append(None)
                voiced_flags.append(False)
                continue

            # Normalized Autocorrelation
            frame_centered = frame - np.mean(frame)
            autocorr = np.correlate(frame_centered, frame_centered, mode="full")
            center = len(autocorr) // 2
            r = autocorr[center:]
            r0 = r[0] + 1e-12

            bound = min(lag_max, len(r) - 1)
            if lag_min >= bound:
                f0_list.append(None)
                voiced_flags.append(False)
                continue

            search_region = r[lag_min : bound + 1]
            peak_offset = int(np.argmax(search_region))
            peak_idx = peak_offset + lag_min
            norm_peak = float(r[peak_idx] / r0)

            if norm_peak >= cls.VOICING_THRESHOLD:
                # Parabolic peak interpolation
                if 0 < peak_idx < len(r) - 1:
                    alpha = r[peak_idx - 1]
                    beta = r[peak_idx]
                    gamma = r[peak_idx + 1]
                    denom = alpha - 2.0 * beta + gamma
                    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
                    true_lag = peak_idx + delta
                else:
                    true_lag = float(peak_idx)

                if true_lag > 0:
                    f0 = cls.SAMPLE_RATE / true_lag
                    if cls.PITCH_MIN <= f0 <= cls.PITCH_MAX:
                        f0_list.append(f0)
                        voiced_flags.append(True)
                        continue

            f0_list.append(None)
            voiced_flags.append(False)

        # 6. Extract Voiced F0 Statistics
        voiced_f0 = [f for f in f0_list if f is not None]
        voiced_count = len(voiced_f0)

        if voiced_count < cls.MIN_VOICED_FRAMES:
            return cls._insufficient_result(
                reason=f"Insufficient voiced speech frames detected ({voiced_count} < {cls.MIN_VOICED_FRAMES}).",
                duration=actual_duration,
                evidence_code="PROSODY_INSUFFICIENT_VOICING",
                total_frames=total_frames,
                voiced_frames=voiced_count,
            )

        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        f0_range = f0_max - f0_min
        f0_cv = float(f0_std / f0_mean) if f0_mean > 0 else 0.0

        # Pitch Microvariation / Jitter proxy: mean relative frame-to-frame F0 change
        pitch_diffs = []
        for j in range(1, len(f0_list)):
            if f0_list[j] is not None and f0_list[j - 1] is not None:
                diff = abs(f0_list[j] - f0_list[j - 1]) / max(f0_list[j - 1], 1e-6)
                pitch_diffs.append(diff)
        jitter_proxy = float(np.mean(pitch_diffs)) if pitch_diffs else None

        # 7. Energy Statistics
        energy_mean = float(np.mean(frame_rms_list))
        energy_std = float(np.std(frame_rms_list))
        active_energies = [e for e in frame_rms_list if e > 0.003]
        if active_energies:
            min_active = max(min(active_energies), 1e-6)
            max_active = max(active_energies)
            energy_dyn_db = float(20.0 * np.log10(max_active / min_active))
        else:
            energy_dyn_db = 0.0

        # Frame-to-frame energy transition delta
        energy_deltas = [abs(frame_rms_list[k] - frame_rms_list[k - 1]) for k in range(1, len(frame_rms_list))]
        energy_delta_mean = float(np.mean(energy_deltas)) if energy_deltas else 0.0

        # 8. Activity & Pause Detection
        active_frames = sum(1 for e in frame_rms_list if e > 0.003)
        speech_activity_ratio = float(active_frames / total_frames) if total_frames > 0 else 0.0
        voiced_unvoiced_ratio = float(voiced_count / max(1, (active_frames - voiced_count)))

        # Detect silent pause intervals (>150ms of consecutive RMS < 0.003)
        min_pause_frames = int(0.150 / (cls.HOP_SIZE / cls.SAMPLE_RATE))  # 15 frames = 150ms
        pause_durations_ms: List[float] = []
        current_silent_run = 0
        total_pause_frames = 0

        for rms in frame_rms_list:
            if rms < 0.003:
                current_silent_run += 1
            else:
                if current_silent_run >= min_pause_frames:
                    dur_ms = current_silent_run * (cls.HOP_SIZE / cls.SAMPLE_RATE) * 1000.0
                    pause_durations_ms.append(dur_ms)
                    total_pause_frames += current_silent_run
                current_silent_run = 0

        if current_silent_run >= min_pause_frames:
            dur_ms = current_silent_run * (cls.HOP_SIZE / cls.SAMPLE_RATE) * 1000.0
            pause_durations_ms.append(dur_ms)
            total_pause_frames += current_silent_run

        pause_count = len(pause_durations_ms)
        pause_ratio = float(total_pause_frames / total_frames) if total_frames > 0 else 0.0
        pause_mean_dur = float(np.mean(pause_durations_ms)) if pause_durations_ms else 0.0
        pause_max_dur = float(np.max(pause_durations_ms)) if pause_durations_ms else 0.0

        # Speech Rhythm Proxy: count of voiced bursts per second
        voiced_bursts = 0
        in_burst = False
        for is_v in voiced_flags:
            if is_v and not in_burst:
                voiced_bursts += 1
                in_burst = True
            elif not is_v:
                in_burst = False
        rhythm_proxy = float(voiced_bursts / max(0.1, actual_duration))

        # 9. Scientifically Honest Prosody Classification
        classification, quality_score, confidence = cls._classify_prosody(
            f0_cv=f0_cv,
            f0_range=f0_range,
            energy_std=energy_std,
            energy_dyn_db=energy_dyn_db,
            jitter_proxy=jitter_proxy,
            voiced_ratio=voiced_count / total_frames,
            evidence=evidence,
        )

        features = ProsodyFeatureSummary(
            pitch_mean_hz=round(f0_mean, 2),
            pitch_std_hz=round(f0_std, 2),
            pitch_min_hz=round(f0_min, 2),
            pitch_max_hz=round(f0_max, 2),
            pitch_range_hz=round(f0_range, 2),
            pitch_variation_coef=round(f0_cv, 4),
            energy_rms_mean=round(energy_mean, 5),
            energy_rms_std=round(energy_std, 5),
            energy_dynamic_range_db=round(energy_dyn_db, 2),
            voiced_unvoiced_ratio=round(voiced_unvoiced_ratio, 3),
            speech_activity_ratio=round(speech_activity_ratio, 3),
            pause_ratio=round(pause_ratio, 3),
            pause_count=pause_count,
            pause_mean_duration_ms=round(pause_mean_dur, 1),
            pause_max_duration_ms=round(pause_max_dur, 1),
            speech_rhythm_proxy=round(rhythm_proxy, 2),
            microvariation_jitter_proxy=round(jitter_proxy, 4) if jitter_proxy is not None else None,
            energy_delta_mean=round(energy_delta_mean, 5),
        )

        return ProsodyAnalysisResult(
            classification=classification,
            features=features,
            quality_score=quality_score,
            confidence=confidence,
            evidence=evidence,
            duration_seconds=round(actual_duration, 2),
            voiced_frames_count=voiced_count,
            total_frames_count=total_frames,
        )

    @classmethod
    def _classify_prosody(
        cls,
        f0_cv: float,
        f0_range: float,
        energy_std: float,
        energy_dyn_db: float,
        jitter_proxy: Optional[float],
        voiced_ratio: float,
        evidence: List[ProsodyEvidenceItem],
    ) -> Tuple[ProsodyClassification, float, str]:
        """
        Determines prosody classification with explainable evidence.
        """
        # Flag 1: Low variation (monotone / synthetic-like flat prosody)
        is_flat_pitch = f0_cv < 0.05 and f0_range < 30.0
        is_flat_energy = energy_dyn_db < 10.0 or energy_std < 0.005

        if is_flat_pitch:
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_FLAT_PITCH",
                severity=ProsodySeverity.MEDIUM,
                message=f"Atypically flat pitch contour detected (CV={f0_cv:.3f}, range={f0_range:.1f}Hz). Characteristic of monotone speech or basic neural TTS."
            ))

        if is_flat_energy:
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_LOW_DYNAMIC_RANGE",
                severity=ProsodySeverity.LOW,
                message=f"Narrow acoustic dynamic range ({energy_dyn_db:.1f} dB). Limited natural vocal intensity modulation."
            ))

        # Flag 2: Unusual prosody (extreme jumps, discontinuities, abnormal jitter)
        is_erratic_jitter = jitter_proxy is not None and jitter_proxy > 0.35
        is_erratic_cv = f0_cv > 0.45 and f0_range > 300.0

        if is_erratic_jitter:
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_ERRATIC_JITTER",
                severity=ProsodySeverity.MEDIUM,
                message=f"High pitch discontinuity / jitter proxy ({jitter_proxy:.3f}). Audible pitch tracking instability or unnatural synthesis glitch."
            ))

        if is_erratic_cv:
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_ERRATIC_PITCH_VARIATION",
                severity=ProsodySeverity.LOW,
                message=f"Wide pitch swings (CV={f0_cv:.3f}, range={f0_range:.1f}Hz). May indicate expressive emotional speech or tracking artifacts."
            ))

        # Decision synthesis
        if is_flat_pitch and is_flat_energy:
            classification = ProsodyClassification.LOW_VARIATION
            quality_score = 0.85
            confidence = "HIGH"
        elif is_flat_pitch or is_flat_energy:
            classification = ProsodyClassification.LOW_VARIATION
            quality_score = 0.75
            confidence = "MEDIUM"
        elif is_erratic_jitter or is_erratic_cv:
            classification = ProsodyClassification.UNUSUAL_PROSODY
            quality_score = 0.80
            confidence = "MEDIUM"
        else:
            classification = ProsodyClassification.NATURAL_VARIATION
            quality_score = 0.90
            confidence = "HIGH"
            evidence.append(ProsodyEvidenceItem(
                code="PROSODY_NATURAL_MODULATION",
                severity=ProsodySeverity.INFO,
                message="Pitch variation, vocal dynamic range, and rhythm exhibit natural human conversational dynamics."
            ))

        return classification, quality_score, confidence

    @classmethod
    def _insufficient_result(
        cls,
        reason: str,
        duration: float,
        evidence_code: str,
        total_frames: int = 0,
        voiced_frames: int = 0,
    ) -> ProsodyAnalysisResult:
        """Constructs safe fallback result for insufficient audio."""
        return ProsodyAnalysisResult(
            classification=ProsodyClassification.INSUFFICIENT_AUDIO,
            features=ProsodyFeatureSummary(
                pitch_mean_hz=None,
                pitch_std_hz=None,
                pitch_min_hz=None,
                pitch_max_hz=None,
                pitch_range_hz=None,
                pitch_variation_coef=None,
                energy_rms_mean=0.0,
                energy_rms_std=0.0,
                energy_dynamic_range_db=0.0,
                voiced_unvoiced_ratio=0.0,
                speech_activity_ratio=0.0,
                pause_ratio=0.0,
                pause_count=0,
                pause_mean_duration_ms=0.0,
                pause_max_duration_ms=0.0,
                speech_rhythm_proxy=0.0,
                microvariation_jitter_proxy=None,
                energy_delta_mean=0.0,
            ),
            quality_score=0.0,
            confidence="UNRELIABLE",
            evidence=[
                ProsodyEvidenceItem(
                    code=evidence_code,
                    severity=ProsodySeverity.HIGH,
                    message=reason,
                )
            ],
            duration_seconds=round(duration, 2),
            voiced_frames_count=voiced_frames,
            total_frames_count=total_frames,
        )
