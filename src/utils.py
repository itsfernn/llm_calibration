import numpy as np
import pandas as pd

def compute_bin_stats(y_true, y_prob, bins):
    """
    Compute per-bin statistics for calibration tasks.
    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class (floats in [0, 1]).
    bins : array-like
        Bin edges for confidence intervals (e.g., [0.0, 0.1, ..., 1.0]).

    Returns a DataFrame with:
        bin, mid, mean_y, mean_conf, n
    """
    conf = np.asarray(y_prob)
    y = np.asarray(y_true)
    bins = np.asarray(bins, dtype=float)

    labels = [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(len(bins)-1)]
    midpoints = (bins[:-1] + bins[1:]) / 2.0
    startpoints = bins[:-1]
    endpoints = bins[1:]
    widths = endpoints - startpoints

    df = pd.DataFrame({"conf": conf, "y": y})
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)

    stats = (
        df.groupby("bin")
        .agg(
            mean_y=("y", "mean"),
            std_y=("y", "std"),
            mean_conf=("conf", "mean"),
            n=("y", "count"),
        )
        .reindex(labels)
        .reset_index()
    )

    stats["mid"] = midpoints
    stats["start"] = startpoints
    stats["end"] = endpoints
    stats["width"] = widths
    return stats

def compute_equal_frequency_bin_stats(y_true, y_prob, n_bins=10):
    """
    Create equal-frequency bins based on predicted probabilities and calculate mean accuracy per bin.
    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class (floats in [0, 1]).
    n_bins : int
        Number of equal-frequency bins to create.
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Bin, Accuracy, Confidence (mean predicted probability in the bin).
    """
    bins = pd.qcut(y_prob, n_bins, labels=False, duplicates='drop')
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'bin': bins})

    binned = df.groupby('bin').agg(
        mean_y=('y_true', 'mean'),
        std_y=('y_true', 'std'),
        mean_conf=('y_prob', 'mean'),
        n=('y_true', 'count'),
        start=('y_prob', 'min'),
        end=('y_prob', 'max'),
        width=('y_prob', lambda x: x.max() - x.min())
    ).reset_index()

    return binned
