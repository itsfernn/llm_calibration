"""Calibration and discrimination metrics.

Single source of truth for metric computation in this repo. Every notebook and
evaluation module computes metrics through :func:`calculate_metrics` and
:func:`bootstrap_metrics` defined here.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from netcal.metrics import ECE
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def verbalized_calibration_error(y_true, y_conf):
    """Discrete calibration error for step-valued verbalized confidence.

    Groups samples by their (rounded) confidence value and computes the
    sample-weighted mean absolute difference between accuracy and confidence
    within each group.
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_conf": np.asarray(y_conf).round(2)}
    )
    binned = (
        df.groupby("y_conf", observed=False)
        .agg(accuracy=("y_true", "mean"), count=("y_true", "count"))
        .reset_index()
    )

    binned["error"] = np.abs(binned["accuracy"] - binned["y_conf"])
    total_samples = len(df)
    binned["weight"] = binned["count"] / total_samples

    return np.sum(binned["weight"] * binned["error"])


# backward-compatible alias
_compute_verbalized_calibration_error = verbalized_calibration_error


def calculate_metrics(y_true, y_conf, ece="binned", n_bins=10):
    """Compute calibration and discrimination metrics for one sample set.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_conf : array-like
        Predicted confidences in [0, 1].
    ece : {"binned", "discrete", "both"}
        - "binned" (default): equal-width binned ECE (netcal).
        - "discrete": discrete verbalized-calibration error, stored under "ece".
        - "both": binned ECE under "ece" plus discrete under "d_ece".
    n_bins : int
        Number of bins for the binned ECE.

    Returns
    -------
    dict with keys: ece[, d_ece], brier, ap_success, ap_errors, auc_roc, acc, nll
    """
    y_true = np.asarray(y_true)
    y_conf = np.asarray(y_conf)

    metrics = {}
    if ece == "binned":
        metrics["ece"] = ECE(bins=n_bins).measure(y_conf, y_true)
    elif ece == "discrete":
        metrics["ece"] = verbalized_calibration_error(y_true, y_conf)
    elif ece == "both":
        metrics["ece"] = ECE(bins=n_bins).measure(y_conf, y_true)
        metrics["d_ece"] = verbalized_calibration_error(y_true, y_conf)
    else:
        raise ValueError(
            f"ece must be one of 'binned', 'discrete', 'both'; got {ece!r}"
        )

    metrics["brier"] = brier_score_loss(y_true, y_conf)
    metrics["ap_success"] = average_precision_score(y_true, y_conf)
    metrics["ap_errors"] = average_precision_score(1 - y_true, 1 - y_conf)
    metrics["auc_roc"] = roc_auc_score(y_true, y_conf)
    metrics["acc"] = y_true.mean()
    metrics["nll"] = log_loss(y_true, y_conf)

    return metrics


def _single_bootstrap(seed, y_true, y_conf, ece, n_bins, num_samples):
    """Run one bootstrap iteration (helper for :func:`bootstrap_metrics`)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_samples, size=num_samples, replace=True)
    return calculate_metrics(y_true[idx], y_conf[idx], ece=ece, n_bins=n_bins)


def bootstrap_metrics(
    y_true,
    y_conf,
    ece="binned",
    n_bootstrap=1000,
    n_bins=10,
    seed=42,
    n_jobs=-1,
):
    """Point estimates plus bootstrap confidence intervals for every metric.

    Runs :func:`calculate_metrics` on ``n_bootstrap`` resamples of the data
    (parallel, reproducible via ``seed``).

    Returns a dict with ``<metric>``, ``<metric>_ci_lower``, ``<metric>_ci_upper``
    and ``<metric>_se`` for each metric of :func:`calculate_metrics`.
    """
    y_true = np.asarray(y_true)
    y_conf = np.asarray(y_conf)
    num_samples = len(y_true)

    # 1. Point estimates
    point_metrics = calculate_metrics(y_true, y_conf, ece=ece, n_bins=n_bins)

    # 2. Parallel random seed generation for exact reproducibility across cores
    sq = np.random.SeedSequence(seed)
    seeds = sq.spawn(n_bootstrap)

    # 3. Parallel execution across CPU cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(seeds[i], y_true, y_conf, ece, n_bins, num_samples)
        for i in range(n_bootstrap)
    )

    # 4. Vectorized dictionary & array aggregation
    keys = point_metrics.keys()
    boot_arrays = {
        k: np.fromiter((res[k] for res in results), dtype=float, count=n_bootstrap)
        for k in keys
    }

    # 5. Compute percentiles (single call) and standard error
    final_metrics = {}
    for key, point_val in point_metrics.items():
        vals = boot_arrays[key]
        ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])
        std_err = np.std(vals, ddof=1)

        final_metrics[key] = point_val
        final_metrics[f"{key}_ci_lower"] = ci_lower
        final_metrics[f"{key}_ci_upper"] = ci_upper
        final_metrics[f"{key}_se"] = std_err

    return final_metrics
