"""Subgroup Performance Stratification and Safeguard Analysis (Phase 11J)."""

from typing import Any, Dict, List, Optional
import numpy as np

from app.evaluation.schemas import SubgroupMetric, SubgroupStatus


class SubgroupAnalyzer:
    """Analyzes benchmark performance across metadata subgroups with statistical sample size safeguards."""

    MIN_SUBGROUP_SAMPLES: int = 20

    @classmethod
    def evaluate_subgroups(
        cls,
        samples_metadata: List[Dict[str, Any]],
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.50,
        min_samples: int = MIN_SUBGROUP_SAMPLES,
    ) -> List[SubgroupMetric]:
        """Group samples by metadata dimensions and compute metrics with safeguards.
        
        Dimensions evaluated:
        - generator
        - language
        - accent
        - gender
        - codec
        - noise_condition
        - duration_bucket
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        if len(samples_metadata) != len(y_true):
            return []

        dimensions = ["generator", "language", "accent", "gender", "codec", "noise_condition", "duration_bucket"]
        metrics: List[SubgroupMetric] = []

        for dim in dimensions:
            grouped_indices: Dict[str, List[int]] = {}
            for i, meta in enumerate(samples_metadata):
                val = meta.get(dim)
                if val:
                    grouped_indices.setdefault(str(val), []).append(i)

            for val_name, idx_list in grouped_indices.items():
                count = len(idx_list)
                if count < min_samples:
                    metrics.append(
                        SubgroupMetric(
                            subgroup_dimension=dim,
                            subgroup_value=val_name,
                            sample_count=count,
                            status=SubgroupStatus.INSUFFICIENT_DATA,
                            note=f"Sample count ({count}) is below minimum threshold ({min_samples}). Statistical evaluation suppressed.",
                        )
                    )
                else:
                    sub_true = y_true[idx_list]
                    sub_score = y_score[idx_list]
                    sub_pred = (sub_score >= threshold).astype(int)

                    acc = float(np.mean(sub_true == sub_pred))
                    tp = int(np.sum((sub_true == 1) & (sub_pred == 1)))
                    fp = int(np.sum((sub_true == 0) & (sub_pred == 1)))
                    fn = int(np.sum((sub_true == 1) & (sub_pred == 0)))
                    
                    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    metrics.append(
                        SubgroupMetric(
                            subgroup_dimension=dim,
                            subgroup_value=val_name,
                            sample_count=count,
                            status=SubgroupStatus.VALIDATED,
                            accuracy=round(acc, 4),
                            f1=round(f1, 4),
                            note=f"Empirically validated on {count} samples.",
                        )
                    )

        return metrics
