import numpy as np
import pandas as pd
from netcal.metrics import ECE
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _compute_verbalized_calibration_error(y_true, y_conf):
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


def calculate_metrics(y_true, y_conf, d_ece=False, n_bins=10):
    metrics = {}

    if d_ece:
        metrics["d_ece"] = _compute_verbalized_calibration_error(y_true, y_conf)

    metrics["ece"] = ECE(bins=n_bins).measure(y_conf, y_true)
    metrics["brier"] = brier_score_loss(y_true, y_conf)
    metrics["ap_success"] = average_precision_score(y_true, y_conf)
    metrics["ap_errors"] = average_precision_score(1 - y_true, 1 - y_conf)
    metrics["auc_roc"] = roc_auc_score(y_true, y_conf)
    metrics["acc"] = y_true.mean()
    metrics["nll"] = log_loss(y_true, y_conf)

    return metrics


import numpy as np
from joblib import Parallel, delayed


def _single_bootstrap(seed, y_true, y_conf, d_ece, n_bins, num_samples):
    """Helper function to execute a single bootstrap iteration."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_samples, size=num_samples, replace=True)
    return calculate_metrics(y_true[idx], y_conf[idx], d_ece=d_ece, n_bins=n_bins)


def bootstrap_metrics(
    y_true, y_conf, d_ece=False, n_bootstrap=1000, n_bins=10, seed=42, n_jobs=-1
):
    """
    Compute point estimates and parallelized bootstrap confidence intervals.

    Parameters:
    - n_jobs: Number of CPU cores to use (-1 uses all available cores).
    """
    y_true = np.asarray(y_true)
    y_conf = np.asarray(y_conf)
    num_samples = len(y_true)

    # 1. Point estimates
    point_metrics = calculate_metrics(y_true, y_conf, d_ece=d_ece, n_bins=n_bins)

    # 2. Parallel random seed generation for exact reproducibility across cores
    sq = np.random.SeedSequence(seed)
    seeds = sq.spawn(n_bootstrap)

    # 3. Parallel execution across CPU cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(seeds[i], y_true, y_conf, d_ece, n_bins, num_samples)
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
