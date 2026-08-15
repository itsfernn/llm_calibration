import matplotlib.pyplot as plt
import numpy as np

from .metrics import calculate_ece


def plot_confidence_error(
    scores: np.ndarray,
    confidences: np.ndarray,
    ax=None,
    M: int = 10,
    xlabel: str = "Confidence",
    ylabel: str = "Accuracy",
    title: str = "Calibration Plot",
):
    """
    Plots a reliability diagram comparing confidence to accuracy.

    Parameters:
        scores (np.ndarray): Binary correctness of predictions (0 or 1).
        confidences (np.ndarray): Model confidence scores (0 to 1).
        ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, a new figure and axis are created.
        M (int): Number of bins.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Plot title.

    Returns:
        Matplotlib figure and axis. If ax was provided, only the axis is returned.
    """

    # Define bin edges and centers
    bin_edges = np.linspace(0, 1, M + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute bin indices
    # Use np.digitize and np.clip to ensure indices are within bounds
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, M - 1)

    # Compute bin sizes and accuracy per bin
    bin_sizes = np.bincount(bin_indices, minlength=M)
    accuracy_per_bin = np.zeros(M)
    confidence_per_bin = np.zeros(M) # Calculate average confidence per bin for text annotation

    for i in range(M):
        if bin_sizes[i] > 0:
            bin_mask = (bin_indices == i)
            accuracy_per_bin[i] = scores[bin_mask].mean()
            confidence_per_bin[i] = confidences[bin_mask].mean() # Use average confidence for plotting


    # Normalize bin sizes for color scaling
    max_bin_size = max(bin_sizes) if max(bin_sizes) > 0 else 1
    # Using a darker color for bins with more samples
    color_intensity = bin_sizes / max_bin_size
    color_map = plt.get_cmap("Blues") # Using Blues colormap, PuBu was in original but Blues is common

    # Create figure
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot bars representing the accuracy in each confidence bin
    # Using bin_centers for x position, accuracy_per_bin for height
    # Using a fixed width, might adjust based on number of bins
    bar_width = 1 / M * 0.9 # Adjust width based on number of bins with a small gap
    ax.bar(
        bin_centers,
        accuracy_per_bin,
        width=bar_width,
        color=color_map(color_intensity),
        edgecolor="black",
        linewidth=0.8,
        label='Accuracy' # Label for legend if needed
    )

    # Plot identity line (perfect calibration)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    # Add Expected Calibration Error (ECE)
    # Calculate ECE using the imported function
    ece_value = calculate_ece(scores, confidences, M)
    ax.text(
        0.95, # x-coordinate (right side)
        0.05, # y-coordinate (bottom side)
        f"ECE: {ece_value:.4f}",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
        horizontalalignment='right', # Anchor text to the right
        verticalalignment='bottom' # Anchor text to the bottom
    )


    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box') # Make the plot square
    ax.grid(True, linestyle="--", alpha=0.6, which='both', axis='both') # Add grid

    # Add a legend
    # ax.legend() # Uncomment if you want a legend for the bars and identity line

    if fig is None:
        return ax
    else:
        return fig, ax
