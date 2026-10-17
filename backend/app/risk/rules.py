"""
Deterministic Rule Matrix for Impersonation Risk Fusion.
Evaluates normalized signal bands and returns risk score, decision label, and structured evidence items.
"""
from typing import Tuple, List
from app.risk.schemas import RiskLevel, EvidenceItem, EvidenceSeverity
from app.risk.normalization import SyntheticBand, SpeakerSimilarityBand


class FusionRuleMatrix:
    """
    Transparent rule matrix mapping synthetic detection and speaker verification signals to risk assessments.
    """

    @classmethod
    def evaluate(
        cls,
        synth_band: SyntheticBand,
        spk_band: SpeakerSimilarityBand,
        synth_score: float,
        spk_sim: float,
        audio_quality_status: str,
    ) -> Tuple[int, RiskLevel, str, List[EvidenceItem]]:
        """
        Executes rule evaluation.
        Returns:
            (risk_score, risk_level, decision_code, evidence_list)
        """
        evidence: List[EvidenceItem] = []

        # Handle degraded audio first
        if audio_quality_status == "invalid":
            evidence.append(
                EvidenceItem(
                    code="AUDIO_QUALITY_DEGRADED",
                    severity=EvidenceSeverity.MEDIUM,
                    message="Signal quality is severely degraded (excessive silence or clipping); evidence is inconclusive."
                )
            )
            return 30, RiskLevel.MEDIUM, "DEGRADED_AUDIO_INCONCLUSIVE", evidence

        # Scenario A: Authorized + Natural
        if spk_band == SpeakerSimilarityBand.HIGH and synth_band == SyntheticBand.LOW:
            evidence.append(
                EvidenceItem(
                    code="AUTHORIZED_NATURAL_VOICE",
                    severity=EvidenceSeverity.INFO,
                    message="High speaker similarity combined with low synthetic score indicates genuine authorized caller."
                )
            )
            score = int(round(10 + max(0.0, synth_score) * 15)) # 10–15 range
            return min(24, score), RiskLevel.LOW, "LIKELY_LEGITIMATE_SPEAKER", evidence

        # Scenario B: Authorized-looking + Synthetic (POTENTIAL VOICE CLONE)
        if spk_band == SpeakerSimilarityBand.HIGH and synth_band == SyntheticBand.HIGH:
            evidence.append(
                EvidenceItem(
                    code="POTENTIAL_VOICE_CLONE",
                    severity=EvidenceSeverity.CRITICAL,
                    message=(
                        "CRITICAL: High speaker similarity combined with strong synthetic-voice evidence "
                        "is consistent with a potential voice-cloning impersonation attack."
                    )
                )
            )
            evidence.append(
                EvidenceItem(
                    code="HIGH_SYNTHETIC_EVIDENCE",
                    severity=EvidenceSeverity.HIGH,
                    message=f"Synthetic voice detector reports elevated synthetic score ({synth_score:.3f})."
                )
            )
            evidence.append(
                EvidenceItem(
                    code="HIGH_SPEAKER_SIMILARITY",
                    severity=EvidenceSeverity.INFO,
                    message=f"Voice acoustically resembles enrolled profile (cosine similarity {spk_sim:.3f})."
                )
            )
            score = int(round(80 + (synth_score - 0.65) * 50)) # 80–98 range
            return min(98, max(75, score)), RiskLevel.CRITICAL, "POTENTIAL_VOICE_CLONE", evidence

        # Scenario C: Unknown + Natural
        if spk_band in [SpeakerSimilarityBand.LOW, SpeakerSimilarityBand.VERY_LOW] and synth_band == SyntheticBand.LOW:
            evidence.append(
                EvidenceItem(
                    code="SPEAKER_IDENTITY_MISMATCH",
                    severity=EvidenceSeverity.MEDIUM,
                    message=f"Voice does not match enrolled profile (similarity {spk_sim:.3f}), but exhibits natural speech dynamics."
                )
            )
            evidence.append(
                EvidenceItem(
                    code="NATURAL_VOICE_DETECTED",
                    severity=EvidenceSeverity.INFO,
                    message="No significant synthetic voice artifacts detected."
                )
            )
            return 35, RiskLevel.MEDIUM, "UNKNOWN_NATURAL_SPEAKER", evidence

        # Scenario D: Unknown + Synthetic
        if spk_band in [SpeakerSimilarityBand.LOW, SpeakerSimilarityBand.VERY_LOW] and synth_band == SyntheticBand.HIGH:
            evidence.append(
                EvidenceItem(
                    code="UNKNOWN_SYNTHETIC_SPEAKER",
                    severity=EvidenceSeverity.HIGH,
                    message="Strong synthetic voice evidence combined with speaker mismatch indicates unauthorized synthetic caller."
                )
            )
            evidence.append(
                EvidenceItem(
                    code="HIGH_SYNTHETIC_EVIDENCE",
                    severity=EvidenceSeverity.HIGH,
                    message=f"Elevated synthetic probability detected ({synth_score:.3f})."
                )
            )
            score = int(round(65 + (synth_score - 0.65) * 20)) # 65–72 range
            return min(74, max(50, score)), RiskLevel.HIGH, "UNKNOWN_SYNTHETIC_SPEAKER", evidence

        # Intermediate 1: HIGH Speaker + MEDIUM Synthetic
        if spk_band == SpeakerSimilarityBand.HIGH and synth_band == SyntheticBand.MEDIUM:
            evidence.append(
                EvidenceItem(
                    code="MODERATE_SYNTHETIC_ANOMALY",
                    severity=EvidenceSeverity.MEDIUM,
                    message="High speaker similarity with borderline synthetic features warrants heightened caution."
                )
            )
            return 60, RiskLevel.HIGH, "SUSPICIOUS_ACOUSTIC_ANOMALY", evidence

        # Intermediate 2: MEDIUM Speaker + HIGH Synthetic
        if spk_band == SpeakerSimilarityBand.MEDIUM and synth_band == SyntheticBand.HIGH:
            evidence.append(
                EvidenceItem(
                    code="POTENTIAL_SYNTHETIC_ATTACK",
                    severity=EvidenceSeverity.HIGH,
                    message="Elevated synthetic evidence with partial speaker similarity indicates probable synthetic spoof."
                )
            )
            return 72, RiskLevel.HIGH, "POTENTIAL_SYNTHETIC_ATTACK", evidence

        # Intermediate 3: MEDIUM Speaker + MEDIUM Synthetic
        if spk_band == SpeakerSimilarityBand.MEDIUM and synth_band == SyntheticBand.MEDIUM:
            evidence.append(
                EvidenceItem(
                    code="AMBIGUOUS_EVIDENCE",
                    severity=EvidenceSeverity.LOW,
                    message="Both identity similarity and synthetic detection yield borderline intermediate readings."
                )
            )
            return 45, RiskLevel.MEDIUM, "AMBIGUOUS_EVIDENCE", evidence

        # Intermediate 4: MEDIUM Speaker + LOW Synthetic
        if spk_band == SpeakerSimilarityBand.MEDIUM and synth_band == SyntheticBand.LOW:
            evidence.append(
                EvidenceItem(
                    code="PROBABLE_AUTHORIZED_SPEAKER",
                    severity=EvidenceSeverity.INFO,
                    message="Natural voice characteristics with moderate speaker alignment."
                )
            )
            return 22, RiskLevel.LOW, "PROBABLE_AUTHORIZED_SPEAKER", evidence

        # Intermediate 5: LOW Speaker + MEDIUM Synthetic
        if spk_band in [SpeakerSimilarityBand.LOW, SpeakerSimilarityBand.VERY_LOW] and synth_band == SyntheticBand.MEDIUM:
            evidence.append(
                EvidenceItem(
                    code="UNKNOWN_SUSPICIOUS_SPEAKER",
                    severity=EvidenceSeverity.MEDIUM,
                    message="Speaker identity mismatch with borderline synthetic artifacts."
                )
            )
            return 52, RiskLevel.HIGH, "UNKNOWN_SUSPICIOUS_SPEAKER", evidence

        # Fallback / Uncertain
        evidence.append(
            EvidenceItem(
                code="UNCERTAIN_ACOUSTIC_EVIDENCE",
                severity=EvidenceSeverity.LOW,
                message="Acoustic features or classifier outputs are inconclusive."
            )
        )
        return 40, RiskLevel.MEDIUM, "UNCERTAIN_ACOUSTIC_EVIDENCE", evidence
