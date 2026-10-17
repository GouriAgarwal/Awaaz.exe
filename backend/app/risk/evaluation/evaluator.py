"""
Evaluator for Impersonation Risk Fusion Benchmarks.
"""
import csv
import json
import logging
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Set

from app.risk.schemas import RiskLevel
from app.risk.service import ImpersonationRiskService
from app.risk.evaluation.schemas import (
    FusionTrial,
    SpeakerLabel,
    SyntheticLabel,
    AttackLabel,
    FusionEvaluationReport,
)
from app.risk.evaluation.metrics import compute_fusion_metrics
from app.risk.evaluation.reports import save_fusion_evaluation_report

logger = logging.getLogger(__name__)


class FusionEvaluator:
    """
    Evaluates multi-modal risk predictions against trial manifests.
    """
    def __init__(self, service: Optional[ImpersonationRiskService] = None):
        self.service = service or ImpersonationRiskService.get_instance()

    @staticmethod
    def parse_manifest(manifest_path: Path) -> List[FusionTrial]:
        """
        Parses manifest file into a list of FusionTrials.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        trials = []
        if manifest_path.suffix.lower() == ".csv":
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spk_lbl = SpeakerLabel.AUTHORIZED if row.get("speaker_label", "").lower() == "authorized" else SpeakerLabel.UNAUTHORIZED
                    syn_lbl_str = row.get("synthetic_label", "").lower()
                    syn_lbl = SyntheticLabel.SYNTHETIC if syn_lbl_str == "synthetic" else SyntheticLabel.NATURAL
                    atk_lbl = AttackLabel.IMPERSONATION if row.get("attack_label", "").lower() == "impersonation" else AttackLabel.LEGITIMATE

                    trials.append(
                        FusionTrial(
                            trial_id=row.get("trial_id", f"fusion_trial_{len(trials)+1}"),
                            audio_path=row.get("audio_path", "").strip(),
                            profile_id=row.get("profile_id", "").strip(),
                            speaker_id=row.get("speaker_id", "").strip(),
                            speaker_label=spk_lbl,
                            synthetic_label=syn_lbl,
                            attack_label=atk_lbl,
                            generator=row.get("generator"),
                            language=row.get("language", "en"),
                            accent=row.get("accent", "standard"),
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
                    spk_lbl = SpeakerLabel.AUTHORIZED if item.get("speaker_label", "").lower() == "authorized" else SpeakerLabel.UNAUTHORIZED
                    syn_lbl_str = item.get("synthetic_label", "").lower()
                    syn_lbl = SyntheticLabel.SYNTHETIC if syn_lbl_str == "synthetic" else SyntheticLabel.NATURAL
                    atk_lbl = AttackLabel.IMPERSONATION if item.get("attack_label", "").lower() == "impersonation" else AttackLabel.LEGITIMATE

                    trials.append(
                        FusionTrial(
                            trial_id=item.get("trial_id", f"fusion_trial_{len(trials)+1}"),
                            audio_path=item.get("audio_path", "").strip(),
                            profile_id=item.get("profile_id", "").strip(),
                            speaker_id=item.get("speaker_id", "").strip(),
                            speaker_label=spk_lbl,
                            synthetic_label=syn_lbl,
                            attack_label=atk_lbl,
                            generator=item.get("generator"),
                            language=item.get("language", "en"),
                            accent=item.get("accent", "standard"),
                            codec=item.get("codec", "pcm"),
                            noise_condition=item.get("noise_condition", "clean"),
                        )
                    )
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}. Use .csv or .jsonl")

        if not trials:
            raise ValueError(f"Manifest at {manifest_path} contains 0 valid trials.")

        return trials

    @staticmethod
    def detect_speaker_leakage(
        trials: List[FusionTrial],
        training_speakers: Optional[Set[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Audits fusion trials for speaker partition contamination.
        """
        train_spks = training_speakers or set()
        warnings = []

        for t in trials:
            if t.speaker_id in train_spks:
                warnings.append(f"Trial {t.trial_id}: Speaker '{t.speaker_id}' leaked from training partition.")

        if warnings:
            msg = "CRITICAL WARNING: Speaker identity leakage detected in fusion evaluation. " + " ".join(warnings[:5])
            logger.warning(msg)
            return True, msg

        return False, None

    def run_evaluation(
        self,
        manifest_path: Path,
        dataset_name: str = "custom_fusion_benchmark",
        training_speakers: Optional[Set[str]] = None,
    ) -> FusionEvaluationReport:
        """
        Runs evaluation on trial manifest.
        """
        trials = self.parse_manifest(manifest_path)
        is_leak, leak_msg = self.detect_speaker_leakage(trials, training_speakers)

        predicted_levels: List[RiskLevel] = []

        for t in trials:
            audio_path = Path(t.audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            with open(audio_path, "rb") as f:
                result = self.service.analyze_audio(f.read(), profile_id=t.profile_id)
                predicted_levels.append(result.risk_level)

        metrics = compute_fusion_metrics(trials, predicted_levels)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report = FusionEvaluationReport(
            evaluation_id=f"fusion_eval_{now_iso.replace(':', '-').replace('.', '-')}",
            timestamp=now_iso,
            status="COMPLETED",
            dataset_name=dataset_name,
            metrics=metrics,
            speaker_leakage_detected=is_leak,
            speaker_leakage_warning=leak_msg,
            limitations=[
                "Cross-channel distortion and noise can shift risk assessments.",
                "Uncalibrated underlying classifiers require domain-specific tuning.",
                "Impersonation risk score is an operational heuristic metric."
            ]
        )

        save_fusion_evaluation_report(report)
        return report
