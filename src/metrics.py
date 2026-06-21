import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from .utils import compute_bin_stats


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    stats = compute_bin_stats(y_true, y_prob, bins)

    stats = stats.dropna(subset=["mean_y"])

    weights = stats["n"] / stats["n"].sum()
    ece = np.sum(weights * np.abs(stats["mean_conf"] - stats["mean_y"]))

    return ece

def bootstrap_ece(y_true, y_prob, n_bins=10, n_bootstrap=1000, seed=None):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    ece_boot = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        ece_b = expected_calibration_error(y_true[idx], y_prob[idx], n_bins)
        ece_boot.append(ece_b)

    ece_boot = np.array(ece_boot)

    ece = expected_calibration_error(y_true, y_prob, n_bins)
    se = np.std(ece_boot, ddof=1)

    # 95% percentile CI
    ci_lower = np.percentile(ece_boot, 2.5)
    ci_upper = np.percentile(ece_boot, 97.5)

    return {
        "ece": ece,
        "se": se,
        "ci_95": (ci_lower, ci_upper),
        "bootstrap_samples": ece_boot
    }

def calculate_metrics(y_true, y_prob):
    """
    Calculate various metrics for binary classification.

    y_true: Ground truth labels (0 or 1)
    y_prob: Predicted probabilities for the positive class
    """
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)

    return {
        "brier_score": brier,
        "roc_auc": auc,
        "average_precision": ap,
        "auc"
        "ece": ece
    }

