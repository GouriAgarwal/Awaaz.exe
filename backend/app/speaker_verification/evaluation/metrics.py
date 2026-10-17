"""
Statistical and Biometric Verification Metrics Calculation.
Calculates FAR, FRR, TAR, TRR, EER, ROC-AUC, Threshold Sweeps, and Subgroup Stratifications.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from app.speaker_verification.evaluation.schemas import (
    TrialType,
    VerificationTrial,
    ThresholdOperatingPoint,
    SubgroupVerificationMetrics,
    VerificationMetrics,
)


def compute_rates_at_threshold(
    scores: List[float],
    labels: List[TrialType],
    threshold: float
) -> Dict[str, Any]:
    """
    Computes confusion matrix and error rates (FAR, FRR, TAR, TRR) at a given threshold.
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have identical lengths.")

    ta = 0 # True Accepts (Target & score >= threshold)
    fr = 0 # False Rejects (Target & score < threshold)
    tr = 0 # True Rejects (Non-Target & score < threshold)
    fa = 0 # False Accepts (Non-Target & score >= threshold)

    target_count = sum(1 for l in labels if l == TrialType.TARGET)
    non_target_count = sum(1 for l in labels if l == TrialType.NON_TARGET)

    for score, label in zip(scores, labels):
        if label == TrialType.TARGET:
            if score >= threshold:
                ta += 1
            else:
                fr += 1
        elif label == TrialType.NON_TARGET:
            if score >= threshold:
                fa += 1
            else:
                tr += 1

    far = float(fa / non_target_count) if non_target_count > 0 else 0.0
    frr = float(fr / target_count) if target_count > 0 else 0.0
    tar = float(ta / target_count) if target_count > 0 else 0.0
    trr = float(tr / non_target_count) if non_target_count > 0 else 0.0

    return {
        "true_accepts": ta,
        "false_rejects": fr,
        "true_rejects": tr,
        "false_accepts": fa,
        "target_trials": target_count,
        "non_target_trials": non_target_count,
        "far": round(far, 4),
        "frr": round(frr, 4),
        "tar": round(tar, 4),
        "trr": round(trr, 4),
    }


def compute_roc_auc(scores: List[float], labels: List[TrialType]) -> Optional[float]:
    """
    Computes ROC-AUC via trapezoidal integration.
    """
    if not scores or not labels:
        return None

    target_count = sum(1 for l in labels if l == TrialType.TARGET)
    non_target_count = sum(1 for l in labels if l == TrialType.NON_TARGET)

    if target_count == 0 or non_target_count == 0:
        return None

    # Sort descending by score
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    
    tp = 0
    fp = 0
    tpr_list = [0.0]
    fpr_list = [0.0]

    for _, label in sorted_pairs:
        if label == TrialType.TARGET:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / target_count)
        fpr_list.append(fp / non_target_count)

    auc = 0.0
    for i in range(1, len(fpr_list)):
        width = fpr_list[i] - fpr_list[i - 1]
        height = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc += width * height

    return round(float(auc), 4)


def compute_eer(scores: List[float], labels: List[TrialType]) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes Equal Error Rate (EER) and the corresponding threshold.
    """
    if not scores or not labels:
        return None, None

    target_count = sum(1 for l in labels if l == TrialType.TARGET)
    non_target_count = sum(1 for l in labels if l == TrialType.NON_TARGET)

    if target_count == 0 or non_target_count == 0:
        return None, None

    thresholds = np.linspace(-1.0, 1.0, num=401)
    best_diff = float("inf")
    best_eer = None
    best_thresh = None

    for t in thresholds:
        rates = compute_rates_at_threshold(scores, labels, float(t))
        diff = abs(rates["far"] - rates["frr"])
        if diff < best_diff:
            best_diff = diff
            best_eer = (rates["far"] + rates["frr"]) / 2.0
            best_thresh = float(t)

    return (round(best_eer, 4) if best_eer is not None else None, round(best_thresh, 4) if best_thresh is not None else None)


def compute_threshold_sweep(
    scores: List[float],
    labels: List[TrialType],
    num_points: int = 51
) -> List[ThresholdOperatingPoint]:
    """
    Computes operating points across the cosine similarity threshold domain.
    """
    points = []
    thresholds = np.linspace(-0.2, 1.0, num=num_points)

    for t in thresholds:
        rates = compute_rates_at_threshold(scores, labels, float(t))
        points.append(
            ThresholdOperatingPoint(
                threshold=round(float(t), 4),
                far=rates["far"],
                frr=rates["frr"],
                tar=rates["tar"],
                trr=rates["trr"],
                true_accepts=rates["true_accepts"],
                false_accepts=rates["false_accepts"],
                true_rejects=rates["true_rejects"],
                false_rejects=rates["false_rejects"],
            )
        )
    return points


def compute_comprehensive_metrics(
    trials: List[VerificationTrial],
    scores: List[float],
    threshold: float = 0.65
) -> VerificationMetrics:
    """
    Computes overall biometric verification metrics, EER, sweep, and subgroup breakdown.
    """
    labels = [t.trial_type for t in trials]
    rates = compute_rates_at_threshold(scores, labels, threshold)
    eer, eer_thresh = compute_eer(scores, labels)
    roc_auc = compute_roc_auc(scores, labels)
    sweep = compute_threshold_sweep(scores, labels)

    # Subgroups
    subgroups: Dict[str, List[SubgroupVerificationMetrics]] = {}
    for attr in ["gender", "language", "accent", "codec", "noise_condition"]:
        attr_groups: Dict[str, List[int]] = {}
        for idx, t in enumerate(trials):
            val = getattr(t, attr, None) or "unknown"
            attr_groups.setdefault(val, []).append(idx)

        subgroups[attr] = []
        for val, indices in attr_groups.items():
            sub_scores = [scores[i] for i in indices]
            sub_labels = [labels[i] for i in indices]
            n_samples = len(indices)

            if n_samples < 5:
                subgroups[attr].append(
                    SubgroupVerificationMetrics(
                        subgroup_key=attr,
                        subgroup_value=val,
                        sample_count=n_samples,
                        status="insufficient_data",
                    )
                )
            else:
                sub_rates = compute_rates_at_threshold(sub_scores, sub_labels, threshold)
                sub_eer, _ = compute_eer(sub_scores, sub_labels)
                subgroups[attr].append(
                    SubgroupVerificationMetrics(
                        subgroup_key=attr,
                        subgroup_value=val,
                        sample_count=n_samples,
                        status="valid",
                        target_trials=sub_rates["target_trials"],
                        non_target_trials=sub_rates["non_target_trials"],
                        far=sub_rates["far"],
                        frr=sub_rates["frr"],
                        eer=sub_eer,
                    )
                )

    return VerificationMetrics(
        total_trials=len(trials),
        target_trials=rates["target_trials"],
        non_target_trials=rates["non_target_trials"],
        true_accepts=rates["true_accepts"],
        false_accepts=rates["false_accepts"],
        true_rejects=rates["true_rejects"],
        false_rejects=rates["false_rejects"],
        far=rates["far"],
        frr=rates["frr"],
        tar=rates["tar"],
        trr=rates["trr"],
        eer=eer,
        eer_threshold=eer_thresh,
        roc_auc=roc_auc,
        threshold_sweep=sweep,
        subgroups=subgroups,
    )
