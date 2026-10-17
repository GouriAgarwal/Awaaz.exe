"""Scientific mathematical evaluation metrics for AI voice detection systems."""

from typing import List, Optional, Tuple
import numpy as np

from app.detection.evaluation.schemas import (
    ConfusionMatrix,
    EvaluationMetricsSummary,
    ThresholdPoint,
)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    """Compute binary confusion matrix (0=REAL, 1=SYNTHETIC)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def compute_rates_from_counts(
    tp: int, tn: int, fp: int, fn: int
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate accuracy, precision, recall, specificity, fpr, fnr, and f1 from counts.
    
    Returns:
        Tuple of (accuracy, precision, recall, specificity, fpr, fnr, f1_score)
    """
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return (
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(specificity, 4),
        round(fpr, 4),
        round(fnr, 4),
        round(f1, 4),
    )


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) where False Alarm Rate (FAR) equals False Reject Rate (FRR).
    
    Args:
        bonafide_scores: Scores for genuine human speech (0=REAL). Expected to be low.
        spoof_scores: Scores for synthetic/cloned speech (1=SYNTHETIC). Expected to be high.
        
    Returns:
        Tuple of (eer_percentage, threshold_at_eer)
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.5

    all_scores = np.sort(np.concatenate([bonafide_scores, spoof_scores]))
    thresholds = np.unique(all_scores)

    if len(thresholds) == 1:
        return 50.0, float(thresholds[0])

    # FAR: bonafide score >= threshold (false synthetic alarm)
    # FRR: spoof score < threshold (missed synthetic clone)
    far = np.array([np.mean(bonafide_scores >= t) for t in thresholds])
    frr = np.array([np.mean(spoof_scores < t) for t in thresholds])

    # Find intersection index where |FAR - FRR| is minimized
    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))

    eer = float((far[idx] + frr[idx]) / 2.0 * 100.0)
    eer_thresh = float(thresholds[idx])
    return round(eer, 2), round(eer_thresh, 4)


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
    """Compute Area Under the ROC Curve (ROC-AUC) using trapezoidal numerical integration."""
    if len(y_true) == 0 or len(y_scores) == 0:
        return None

    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    if pos_count == 0 or neg_count == 0:
        return None

    # Sort descending by score
    sort_idx = np.argsort(-y_scores)
    sorted_labels = y_true[sort_idx]
    sorted_scores = y_scores[sort_idx]

    # Evaluate FPR and TPR at all unique score cutoffs
    unique_thresholds = np.unique(sorted_scores)
    tpr_list = [0.0]
    fpr_list = [0.0]

    for thresh in np.sort(unique_thresholds)[::-1]:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr_list.append(tp / pos_count)
        fpr_list.append(fp / neg_count)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Sort points by FPR ascending
    points = sorted(zip(fpr_list, tpr_list), key=lambda x: (x[0], x[1]))
    fprs = [p[0] for p in points]
    tprs = [p[1] for p in points]

    # Trapezoidal rule: sum (x_{i} - x_{i-1}) * (y_{i} + y_{i-1}) / 2
    auc = 0.0
    for i in range(1, len(points)):
        dx = fprs[i] - fprs[i - 1]
        avg_y = (tprs[i] + tprs[i - 1]) / 2.0
        auc += dx * avg_y

    return round(float(np.clip(auc, 0.0, 1.0)), 4)


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[ThresholdPoint]:
    """Generate threshold performance analysis across the score range."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 21)]

    sweep_points: List[ThresholdPoint] = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        acc, prec, rec, spec, fpr, fnr, f1 = compute_rates_from_counts(tp, tn, fp, fn)

        sweep_points.append(
            ThresholdPoint(
                threshold=t,
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                precision=prec,
                recall=rec,
                specificity=spec,
                fpr=fpr,
                fnr=fnr,
                f1=f1,
            )
        )

    return sweep_points


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    default_threshold: float = 0.50,
) -> EvaluationMetricsSummary:
    """Calculate comprehensive benchmark metrics and threshold sweeps.
    
    Args:
        y_true: 1D array of ground truth labels (0=REAL, 1=SYNTHETIC).
        y_scores: 1D array of continuous synthetic likelihood scores [0.0, 1.0].
        default_threshold: Operating cutoff boundary for binary classification.
        
    Returns:
        EvaluationMetricsSummary with full metrics breakdown.
    """
    if len(y_true) == 0:
        raise ValueError("Cannot compute evaluation metrics on empty labels array.")

    y_pred = (y_scores >= default_threshold).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred)
    acc, prec, rec, spec, fpr, fnr, f1 = compute_rates_from_counts(cm.tp, cm.tn, cm.fp, cm.fn)

    # EER calculation
    bonafide = y_scores[y_true == 0]
    spoof = y_scores[y_true == 1]
    eer_val, eer_thresh = compute_eer(bonafide, spoof) if (len(bonafide) > 0 and len(spoof) > 0) else (None, None)

    # ROC-AUC calculation
    auc = compute_roc_auc(y_true, y_scores)

    # Threshold sweep
    sweep = compute_threshold_sweep(y_true, y_scores)

    return EvaluationMetricsSummary(
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        f1_score=f1,
        fpr=fpr,
        fnr=fnr,
        roc_auc=auc,
        eer=eer_val,
        eer_threshold=eer_thresh,
        confusion_matrix=cm,
        threshold_sweep=sweep,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    dataset_name: Optional[str] = None,
) -> "EvaluationMetrics":
    """Backwards-compatible helper for legacy test suites."""
    from app.detection.evaluation.schemas import EvaluationMetrics

    if len(y_true) == 0 or len(y_scores) == 0:
        return EvaluationMetrics(evaluation_status="NOT_RUN")

    bonafide = y_scores[y_true == 0]
    spoof = y_scores[y_true == 1]
    eer_val, eer_thresh = compute_eer(bonafide, spoof) if (len(bonafide) > 0 and len(spoof) > 0) else (None, None)

    y_pred = (y_scores >= 0.5).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred)
    acc, prec, rec, spec, fpr, fnr, f1 = compute_rates_from_counts(cm.tp, cm.tn, cm.fp, cm.fn)

    return EvaluationMetrics(
        evaluation_status="COMPLETED",
        dataset_name=dataset_name,
        equal_error_rate=eer_val,
        f1_score=f1,
        accuracy=acc,
        threshold_at_eer=eer_thresh,
        total_eval_samples=len(y_true),
        disclaimer="Empirical evaluation on supplied test array.",
    )
