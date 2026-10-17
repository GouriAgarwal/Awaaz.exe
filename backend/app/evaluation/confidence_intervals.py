"""Deterministic statistical bootstrap confidence interval estimation (Phase 11I)."""

from typing import Callable, List, Optional, Tuple
import numpy as np
from app.evaluation.schemas import ConfidenceInterval


class BootstrapEvaluator:
    """Computes reproducible non-parametric bootstrap confidence intervals for classification and biometric metrics."""

    @staticmethod
    def compute_metric_ci(
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str,
        confidence_level: float = 0.95,
        n_iterations: int = 1000,
        random_seed: int = 42,
    ) -> ConfidenceInterval:
        """Compute bootstrap confidence interval for a custom binary metric function.
        
        Args:
            y_true: Ground truth binary labels (0=Negative/Real, 1=Positive/Synthetic).
            y_score: Predicted continuous scores or binary decisions.
            metric_func: Function signature (y_true_boot, y_score_boot) -> float.
            metric_name: Name of metric being estimated.
            confidence_level: Target confidence interval coverage (default 0.95).
            n_iterations: Number of bootstrap resamples (default 1000).
            random_seed: Seed for reproducibility.
            
        Returns:
            ConfidenceInterval dataclass.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        n_samples = len(y_true)

        if n_samples == 0:
            return ConfidenceInterval(
                metric=metric_name,
                point_estimate=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                confidence_level=confidence_level,
                bootstrap_iterations=0,
            )

        # Point estimate on full dataset
        point_estimate = float(metric_func(y_true, y_score))

        # Bootstrap resampling
        rng = np.random.RandomState(random_seed)
        boot_values: List[float] = []

        for _ in range(n_iterations):
            idx = rng.randint(0, n_samples, size=n_samples)
            boot_true = y_true[idx]
            boot_score = y_score[idx]

            # Ensure both classes exist in resample to prevent metric divergence
            if len(np.unique(boot_true)) < 2:
                continue

            try:
                val = metric_func(boot_true, boot_score)
                if np.isfinite(val):
                    boot_values.append(val)
            except Exception:
                continue

        if len(boot_values) < 10:
            return ConfidenceInterval(
                metric=metric_name,
                point_estimate=point_estimate,
                ci_lower=point_estimate,
                ci_upper=point_estimate,
                confidence_level=confidence_level,
                bootstrap_iterations=len(boot_values),
            )

        alpha = (1.0 - confidence_level) / 2.0
        lower = float(np.percentile(boot_values, alpha * 100))
        upper = float(np.percentile(boot_values, (1.0 - alpha) * 100))

        return ConfidenceInterval(
            metric=metric_name,
            point_estimate=point_estimate,
            ci_lower=lower,
            ci_upper=upper,
            confidence_level=confidence_level,
            bootstrap_iterations=len(boot_values),
        )

    @classmethod
    def compute_standard_classification_cis(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.50,
        n_iterations: int = 1000,
        random_seed: int = 42,
    ) -> List[ConfidenceInterval]:
        """Compute standard set of bootstrap 95% CIs for classification metrics."""
        y_pred = (y_score >= threshold).astype(int)

        def acc_fn(yt, ys):
            yp = (ys >= threshold).astype(int)
            return float(np.mean(yt == yp))

        def prec_fn(yt, ys):
            yp = (ys >= threshold).astype(int)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        def recall_fn(yt, ys):
            yp = (ys >= threshold).astype(int)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            fn = int(np.sum((yt == 1) & (yp == 0)))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        def spec_fn(yt, ys):
            yp = (ys >= threshold).astype(int)
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        def f1_fn(yt, ys):
            p = prec_fn(yt, ys)
            r = recall_fn(yt, ys)
            return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        metrics = [
            ("accuracy", acc_fn),
            ("precision", prec_fn),
            ("recall", recall_fn),
            ("specificity", spec_fn),
            ("f1_score", f1_fn),
        ]

        results = []
        for name, fn in metrics:
            ci = cls.compute_metric_ci(
                y_true=y_true,
                y_score=y_score,
                metric_func=fn,
                metric_name=name,
                n_iterations=n_iterations,
                random_seed=random_seed,
            )
            results.append(ci)

        return results
