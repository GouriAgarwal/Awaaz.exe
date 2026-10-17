"""
Evaluation Runner for Speaker Verification Subsystem with Speaker Leakage Detection.
"""
import csv
import json
import logging
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Set

from app.services.audio_service import AudioService
from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.similarity import CosineSimilarityMetric
from app.speaker_verification.provenance import get_speaker_model_provenance
from app.speaker_verification.evaluation.schemas import (
    TrialType,
    VerificationTrial,
    SpeakerVerificationEvaluationReport,
)
from app.speaker_verification.evaluation.metrics import compute_comprehensive_metrics
from app.speaker_verification.evaluation.reports import save_speaker_evaluation_report

logger = logging.getLogger(__name__)


class SpeakerVerificationEvaluator:
    """
    Evaluates speaker verification accuracy on trial manifests and detects speaker leakage.
    """
    def __init__(self, encoder: BaseSpeakerEncoder):
        self.encoder = encoder
        self.similarity_metric = CosineSimilarityMetric()

    @staticmethod
    def parse_manifest(manifest_path: Path) -> List[VerificationTrial]:
        """
        Parses trial manifest in CSV or JSONL format.
        Expected fields:
        trial_id,enrollment_audio_path,verification_audio_path,enrollment_speaker_id,verification_speaker_id,trial_type
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        trials = []
        if manifest_path.suffix.lower() == ".csv":
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_type_str = row.get("trial_type", "").strip().lower()
                    t_type = TrialType.TARGET if t_type_str == "target" else TrialType.NON_TARGET
                    trials.append(
                        VerificationTrial(
                            trial_id=row.get("trial_id", f"trial_{len(trials)+1}"),
                            enrollment_audio_path=row.get("enrollment_audio_path", "").strip(),
                            verification_audio_path=row.get("verification_audio_path", "").strip(),
                            enrollment_speaker_id=row.get("enrollment_speaker_id", "").strip(),
                            verification_speaker_id=row.get("verification_speaker_id", "").strip(),
                            trial_type=t_type,
                            generator=row.get("generator"),
                            language=row.get("language", "en"),
                            accent=row.get("accent", "standard"),
                            gender=row.get("gender", "unknown"),
                            codec=row.get("codec", "pcm"),
                            noise_condition=row.get("noise_condition", "clean"),
                        )
                    )
        elif manifest_path.suffix.lower() in [".json", ".jsonl"]:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    t_type_str = item.get("trial_type", "").strip().lower()
                    t_type = TrialType.TARGET if t_type_str == "target" else TrialType.NON_TARGET
                    trials.append(
                        VerificationTrial(
                            trial_id=item.get("trial_id", f"trial_{len(trials)+1}"),
                            enrollment_audio_path=item.get("enrollment_audio_path", "").strip(),
                            verification_audio_path=item.get("verification_audio_path", "").strip(),
                            enrollment_speaker_id=item.get("enrollment_speaker_id", "").strip(),
                            verification_speaker_id=item.get("verification_speaker_id", "").strip(),
                            trial_type=t_type,
                            generator=item.get("generator"),
                            language=item.get("language", "en"),
                            accent=item.get("accent", "standard"),
                            gender=item.get("gender", "unknown"),
                            codec=item.get("codec", "pcm"),
                            noise_condition=item.get("noise_condition", "clean"),
                        )
                    )
        else:
            raise ValueError(f"Unsupported manifest extension: {manifest_path.suffix}. Use .csv or .jsonl")

        if not trials:
            raise ValueError(f"Manifest at {manifest_path} contains 0 valid trials.")

        return trials

    @staticmethod
    def detect_speaker_leakage(
        trials: List[VerificationTrial],
        training_speakers: Optional[Set[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Audits trial pairs for speaker identity leakage or invalid label contradictions.
        """
        warnings = []
        train_spks = training_speakers or set()

        for t in trials:
            # Check training overlap
            if t.enrollment_speaker_id in train_spks or t.verification_speaker_id in train_spks:
                warnings.append(
                    f"Trial {t.trial_id}: Speaker '{t.enrollment_speaker_id}' / '{t.verification_speaker_id}' "
                    f"leaked from model training partition."
                )

            # Check label contradictions
            if t.trial_type == TrialType.TARGET and t.enrollment_speaker_id != t.verification_speaker_id:
                warnings.append(
                    f"Trial {t.trial_id}: Label is 'target' but enrollment speaker ({t.enrollment_speaker_id}) "
                    f"!= verification speaker ({t.verification_speaker_id})."
                )
            elif t.trial_type == TrialType.NON_TARGET and t.enrollment_speaker_id == t.verification_speaker_id:
                warnings.append(
                    f"Trial {t.trial_id}: Label is 'non_target' but enrollment speaker == verification speaker ({t.enrollment_speaker_id})."
                )

        if warnings:
            full_msg = "CRITICAL WARNING: Speaker identity leakage detected across splits. " + " ".join(warnings[:5])
            logger.warning(full_msg)
            return True, full_msg

        return False, None

    def run_evaluation(
        self,
        manifest_path: Path,
        dataset_name: str = "custom_speaker_benchmark",
        threshold: float = 0.65,
        training_speakers: Optional[Set[str]] = None
    ) -> SpeakerVerificationEvaluationReport:
        """
        Executes speaker verification evaluation across trials.
        """
        trials = self.parse_manifest(manifest_path)
        is_leakage, leakage_msg = self.detect_speaker_leakage(trials, training_speakers)

        provenance = get_speaker_model_provenance()
        scores = []

        for t in trials:
            enr_path = Path(t.enrollment_audio_path)
            ver_path = Path(t.verification_audio_path)

            if not enr_path.exists():
                raise FileNotFoundError(f"Enrollment audio file not found: {enr_path}")
            if not ver_path.exists():
                raise FileNotFoundError(f"Verification audio file not found: {ver_path}")

            with open(enr_path, "rb") as f:
                enr_audio = AudioService.get_processed_audio_data(f.read())
            with open(ver_path, "rb") as f:
                ver_audio = AudioService.get_processed_audio_data(f.read())

            enr_emb = self.encoder.embed(enr_audio.samples, sample_rate=16000)
            ver_emb = self.encoder.embed(ver_audio.samples, sample_rate=16000)

            sim = self.similarity_metric.compute_similarity(enr_emb, ver_emb)
            scores.append(sim)

        metrics = compute_comprehensive_metrics(trials, scores, threshold=threshold)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report = SpeakerVerificationEvaluationReport(
            evaluation_id=f"spk_eval_{now_iso.replace(':', '-').replace('.', '-')}",
            timestamp=now_iso,
            model_id=self.encoder.model_id,
            checkpoint_sha256=provenance.checkpoint_sha256,
            dataset_name=dataset_name,
            status="COMPLETED",
            evaluation_metrics=metrics,
            speaker_leakage_detected=is_leakage,
            speaker_leakage_warning=leakage_msg,
            limitations=[
                "Cross-channel acoustic variability and background noise can shift similarity distributions.",
                "Short verification utterances (< 1.5s) exhibit wider variance in embedding space.",
                "Speaker similarity measures acoustic identity; voice cloning defenses require independent anti-spoofing fusion."
            ]
        )

        save_speaker_evaluation_report(report)
        return report
