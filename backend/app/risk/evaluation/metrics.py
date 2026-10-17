"""
Evaluation Metrics Computation for Impersonation Risk Fusion.
Calculates 4-Quadrant analysis, confusion matrix, precision, recall, and F1.
"""
from typing import List, Dict, Any, Tuple
from app.risk.schemas import RiskLevel
from app.risk.evaluation.schemas import (
    FusionTrial,
    AttackLabel,
    SpeakerLabel,
    SyntheticLabel,
    FourQuadrantMatrix,
    FusionEvaluationMetrics,
)


def compute_fusion_metrics(
    trials: List[FusionTrial],
    predicted_risk_levels: List[RiskLevel],
    impersonation_threshold_level: RiskLevel = RiskLevel.HIGH,
) -> FusionEvaluationMetrics:
    """
    Computes 4-quadrant benchmark performance and binary impersonation detection accuracy.
    """
    if len(trials) != len(predicted_risk_levels):
        raise ValueError("Trials and predictions count mismatch.")

    q_auth_nat = 0
    q_auth_syn = 0
    q_unauth_nat = 0
    q_unauth_syn = 0

    tp = 0 # True Positives: Attack labeled IMPERSONATION & Risk >= HIGH
    fp = 0 # False Positives: Attack labeled LEGITIMATE & Risk >= HIGH
    tn = 0 # True Negatives: Attack labeled LEGITIMATE & Risk < HIGH
    fn = 0 # False Negatives: Attack labeled IMPERSONATION & Risk < HIGH

    high_risk_levels = [RiskLevel.HIGH, RiskLevel.CRITICAL]

    for trial, pred_level in zip(trials, predicted_risk_levels):
        # 4-quadrant assignment
        if trial.speaker_label == SpeakerLabel.AUTHORIZED and trial.synthetic_label == SyntheticLabel.NATURAL:
            q_auth_nat += 1
        elif trial.speaker_label == SpeakerLabel.AUTHORIZED and trial.synthetic_label == SyntheticLabel.SYNTHETIC:
            q_auth_syn += 1
        elif trial.speaker_label == SpeakerLabel.UNAUTHORIZED and trial.synthetic_label == SyntheticLabel.NATURAL:
            q_unauth_nat += 1
        else:
            q_unauth_syn += 1

        is_predicted_attack = pred_level in high_risk_levels
        is_actual_attack = trial.attack_label == AttackLabel.IMPERSONATION

        if is_actual_attack and is_predicted_attack:
            tp += 1
        elif not is_actual_attack and is_predicted_attack:
            fp += 1
        elif not is_actual_attack and not is_predicted_attack:
            tn += 1
        else:
            fn += 1

    total = len(trials)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return FusionEvaluationMetrics(
        total_trials=total,
        accuracy=round(float(acc), 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        specificity=round(float(spec), 4),
        f1_score=round(float(f1), 4),
        fpr=round(float(fpr), 4),
        fnr=round(float(fnr), 4),
        four_quadrant_matrix=FourQuadrantMatrix(
            authorized_natural=q_auth_nat,
            authorized_synthetic=q_auth_syn,
            unauthorized_natural=q_unauth_nat,
            unauthorized_synthetic=q_unauth_syn,
        ),
        subgroups={},
    )
