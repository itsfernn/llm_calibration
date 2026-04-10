import numpy as np
import matplotlib.pyplot as plt

from .utils import compute_bin_stats, compute_equal_frequency_bin_stats

def confidence_bar_plot(y_true, y_probs, bins=None, ax=None):
    """
    y_true: list of 1D arrays (or a single 1D array)
    y_probs: list of 1D arrays (or a single 1D array)
    bins: optional array of bin edges
    ax: optional matplotlib axis
    """

    # Flatten if list of arrays
    if isinstance(y_true, list):
        y_true = np.concatenate(y_true)
    if isinstance(y_probs, list):
        y_probs = np.concatenate(y_probs)

    if bins is None:
        bins = np.linspace(0,1, 11)

    bin_stats = compute_bin_stats(y_true, y_probs, bins)

    x = bin_stats["mid"].values
    y = bin_stats["mean_y"].values
    n = bin_stats["n"].fillna(0).values  # handle NaN safely
    y_errs = bin_stats["std_y"].values / (n + 1e-8)  # avoid divide by zero
    widths = bin_stats["width"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    cmap = plt.get_cmap("Blues")
    norm = (n - np.min(n)) / (np.max(n) - np.min(n) + 1e-8) 

    for xi, yi, ni, wi, yerr in zip(x, y, norm, widths, y_errs):
        ax.bar(xi, yi, width=wi, color=cmap(ni), yerr=yerr, edgecolor="black")


    ax.plot([0, 1], [0, 1], '--', alpha=0.7, color='gray')

    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)


    return ax



def confidence_plot(y_true, y_probs, ax=None, label=None, bins=10):
    """
    y_true: 1D array of true binary labels (0 or 1)
    y_conf: 1D array of predicted confidence scores (between 0 and 1
    """

    bin_stats = compute_equal_frequency_bin_stats(y_true, y_probs, n_bins=bins)


    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle='--', alpha=0.6)

    label = label or 'Model calibration'
    ax.plot(bin_stats["mean_conf"], bin_stats["mean_y"], '.-',  label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def interploated_confidence_plot(y, x, n_bins=10, error_band=None, ax=None):
    """
    y: list of 1D arrays (accuracy values per run)
    x: list of 1D arrays (confidence values per run)
    error_band: None, 'std', 'stderr', or 'minmax' to indicate whether to plot error bands
    """

    assert len(y) == len(x), "y and x must have the same number of runs"

    if ax is None:
        fig, ax = plt.subplots()
        plt.figure(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], '--', color='gray')

    bins = []

    for yi, xi in zip(y, x):
        bin_stats = compute_equal_frequency_bin_stats(yi, xi, n_bins=n_bins)
        bins.append((bin_stats["mean_conf"].values, bin_stats["mean_y"].values))


    # Determine global min/max from all runs
    all_conf = np.concatenate([conf for conf, _ in bins])
    x_min, x_max = all_conf.min(), all_conf.max()

    # Interpolate across actual min/max range
    fixed_x = np.linspace(x_min, x_max, 100)
    interpolated_y = []

    for conf_mean, y_mean in bins:
        interp_y = np.interp(fixed_x, conf_mean, y_mean)
        interpolated_y.append(interp_y)

    fixed_y = np.mean(interpolated_y, axis=0)

    ax.plot(fixed_x, fixed_y)


    if error_band is None:
        pass
    elif error_band == 'std':
        std = np.std(interpolated_y, axis=0)
        ax.fill_between(fixed_x, fixed_y - std, fixed_y + std, alpha=0.3)
    elif error_band == 'stderr':
        stderr = np.std(interpolated_y, axis=0) / np.sqrt(len(interpolated_y))
        ax.fill_between(fixed_x, fixed_y - stderr, fixed_y + stderr, alpha=0.3)
    elif error_band == 'minmax':
        min_y = np.min(interpolated_y, axis=0)
        max_y = np.max(interpolated_y, axis=0)
        ax.fill_between(fixed_x, min_y, max_y, alpha=0.3)
    else:
        raise ValueError("Invalid error_band option")

    return ax









def confidence_plot_multi(confs_list, acc_list, ax=None, label=None, n_points=200):
    """
    confs_list: list of 1D arrays (confidence values per run)
    acc_list:   list of 1D arrays (accuracy values per run)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle='--', alpha=0.6)

    label = label or "Model calibration"

    # --- Common interpolation grid ---
    x_common = np.linspace(0, 1, n_points)

    # --- Interpolate all runs ---
    y_interp = []
    for confs, acc in zip(confs_list, acc_list):
        y_interp.append(np.interp(x_common, confs, acc))

    y_interp = np.array(y_interp)

    # --- Mean and standard error ---
    mean = np.mean(y_interp, axis=0)
    std = np.std(y_interp, axis=0)
    stderr = std / np.sqrt(len(y_interp))

    # --- Plot mean curve ---
    ax.plot(x_common, mean, '-', linewidth=2, label=label)

    # --- Plot shaded standard error band ---
    ax.fill_between(
        x_common,
        mean - stderr,
        mean + stderr,
        alpha=0.3
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax
