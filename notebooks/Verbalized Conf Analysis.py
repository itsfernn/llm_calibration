import marimo

__generated_with = "0.23.14"
app = marimo.App()

with app.setup:
    import marimo as mo
    import pandas as pd
    from sklearn.metrics import (
        brier_score_loss,
        average_precision_score,
        roc_auc_score,
        log_loss,
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from src import scaling, metrics
    import json
    from pathlib import Path

    import seaborn as sns


@app.cell
def _(prompt_mapping):
    CV_FOLDS = 5
    prompt_order_raw = ["direct", "top-k",  "multistep", "cot",]
    prompt_order = [prompt_mapping[p] for p in prompt_order_raw]
    return (prompt_order,)


@app.cell
def _(prompt_mapping):
    runs_dir = Path.cwd() / "runs-legacy"
    with open(runs_dir / "data.json") as f:
        _data = json.load(f)

    _runs = []
    for _run in _data:
        _runs.append(_run)

    runs_df = pd.DataFrame(_runs)
    runs_df["prompt"] = runs_df["prompt"].map(prompt_mapping)
    return runs_df, runs_dir


@app.cell
def _(runs_df):
    mo.ui.table(
        runs_df,
        selection="multi",
    )
    return


@app.cell
def _(runs_dir):
    def get_run_df(run):
        csv_file = runs_dir / run["csv_file"]
        df = pd.read_csv(csv_file)
        df["dataset"] = run["dataset"]
        return df

    return (get_run_df,)


@app.cell(hide_code=True)
def _(runs_df):
    _models = list(runs_df["model"].unique())
    _datasets = list(runs_df["dataset"].unique())

    models_dropdown = mo.ui.dropdown(value=_models[0], options=_models)
    datasets_dropdown = mo.ui.dropdown(value=_datasets[0], options=_datasets)

    mo.hstack(
        [
            models_dropdown,
            datasets_dropdown,
        ],
        justify="start",
    )
    return datasets_dropdown, models_dropdown


@app.cell
def _(
    plot_reliability_bar_diagrams_sns,
    plot_reliability_diagrams,
    selection_df,
):
    mo.hstack(
        [
            plot_reliability_bar_diagrams_sns(selection_df),
            plot_reliability_diagrams(selection_df),
        ]
    )
    return


@app.cell
def _(datasets_dropdown, models_dropdown, prepare_bins, prompt_order):
    def plot_reliability_bar_diagrams_sns(df):
        """Plot calibration reliability diagram."""
        bins_df = prepare_bins(df)

        g = sns.FacetGrid(
            bins_df,
            col="prompt",
            hue="prompt",
            hue_order=prompt_order,
            col_wrap=2,
            col_order=prompt_order,
            despine=False,
        )

        # Map the custom plotting function across facets
        g.map_dataframe(
            confidence_bar_plot_alt,
            x="mid",
            y="mean_y",
        )

        g.set_titles(col_template="{col_name}", size=14)
        g.set_axis_labels("Confidence", "Accuracy")
        g.fig.suptitle(f"{models_dropdown.value} on {datasets_dropdown.value}", y=1.03, size=16)

        return g

    return (plot_reliability_bar_diagrams_sns,)


@app.cell
def _(P):
    P
    return


@app.cell
def _(datasets_dropdown, models_dropdown, runs_df):
    _selected_model = models_dropdown.value

    selection_df = runs_df.copy()

    selection_df = selection_df[selection_df["model"] == _selected_model]

    _selected_dataset = datasets_dropdown.value

    selection_df = selection_df[selection_df["dataset"] == _selected_dataset]
    return (selection_df,)


@app.cell
def _(datasets_dropdown, get_run_df, models_dropdown):
    def prepare_bins(selection_df):
        combined_dfs = []

        for i, run in selection_df.iterrows():
            df = get_run_df(run)

            row = compute_bin_stats(
                y_true=df["gpt_eval"],
                y_prob=df["confidence"],
                n_bins=10
            )

            # Attach run metadata so Seaborn can group by them
            row["unit"] = i
            row['prompt'] = run['prompt']
            row['model'] = run.get('model', models_dropdown.value)
            row['dataset'] = run.get('dataset', datasets_dropdown.value)

            combined_dfs.append(row)

        return pd.concat(combined_dfs, ignore_index=True)

    return (prepare_bins,)


@app.cell(hide_code=True)
def _(datasets_dropdown, get_run_df, models_dropdown):
    def prepare_eq_bins(selection_df):
        combined_dfs = []

        for i, run in selection_df.iterrows():
            df = get_run_df(run)

            row = compute_equal_frequency_bin_stats(
                y_true=df["gpt_eval"],
                y_prob=df["confidence"],
                n_bins=10
            )

            # Attach run metadata so Seaborn can group by them
            row["unit"] = i
            row['prompt'] = run['prompt']
            row['model'] = run.get('model', models_dropdown.value)
            row['dataset'] = run.get('dataset', datasets_dropdown.value)

            combined_dfs.append(row)

        return pd.concat(combined_dfs, ignore_index=True)

    return (prepare_eq_bins,)


@app.cell
def _(datasets_dropdown, models_dropdown, prepare_eq_bins, prompt_order):
    def plot_reliability_diagrams(selection_df, hue="prompt", style=None):
        bins_df = prepare_eq_bins(selection_df)

        fig, ax = plt.subplots(figsize=(7, 6))

        # Reference line for perfect calibration (y = x)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)

        # Seaborn handles hue (colors) and style (linetypes for models/datasets)
        sns.lineplot(
            data=bins_df,
            x="mean_conf",
            y="mean_y",
            hue=hue,# e.g., "prompt"
            hue_order=prompt_order,
            units="unit",
            style=style, 
            marker='o',
            markersize=8,
            markeredgecolor="none",
            linewidth=3,
            ax=ax
        )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{models_dropdown.value} on {datasets_dropdown.value}")
        ax.grid(True, linestyle=":", alpha=0.6)

        return fig

    return (plot_reliability_diagrams,)


@app.cell
def _(selection_df):
    selection_df
    return


@app.cell(hide_code=True)
def _(runs_df):
    _models = list(runs_df["model"].unique())
    _models = ["all"] + _models

    density_model_dropdown = mo.ui.dropdown(value=_models[0], options=_models)
    density_model_dropdown
    return (density_model_dropdown,)


@app.cell(hide_code=True)
def _(density_model_dropdown, get_run_df, prompt_order, runs_df):
    _selected_model = density_model_dropdown.value
    if _selected_model == "all":
        filtered_runs = runs_df
        _title = "Density of Verblized Confidences"
    else:
        try:
            filtered_runs = runs_df[runs_df["model"] == _selected_model]
            _title = f"Density of Verblized Confidences for {_selected_model}"
        except ValueError:
            raise f"Unknown model {_selected_model}"

    parts = []

    # get all confidences from selected runs and assign prompt and dataset to them
    for _, run in filtered_runs.iterrows():
        df = get_run_df(run).copy()
        parts.append(
            df[["confidence", "gpt_eval"]]
            .assign(prompt=run["prompt"])
            .assign(dataset=run["dataset"])
        )

    plot_df = pd.concat(parts, ignore_index=True)

    # sns.displot manages the grid, facets, and single shared legend automatically
    g = sns.displot(
        data=plot_df,
        x="confidence",
        hue="dataset",
        col="prompt",
        col_order=prompt_order,
        col_wrap=2,  # Wraps the layout into a 2x2 grid
        kind="kde",  # Density plot
        fill=True,
        common_norm=False,
        clip=(0, 1),
        alpha=0.3,
        height=3.5,  # Controls size of each individual facet
        aspect=1,  # Enforces 1:1 square subplots
        palette="Set2"
    )

    # Set axis limits & labels across all subplots
    g.set(xlim=(0, 1), xlabel="Confidence")
    g.set_titles(col_template="{col_name}")

    # Move title up slightly and make room for it
    g.fig.suptitle(_title, y=1.02)

    # Adjust padding between rows and columns
    g.figure.subplots_adjust(hspace=0.3, wspace=0.2)

    g
    return g, plot_df


@app.cell
def _(density_model_dropdown, g, plot_df, prompt_order):
    # --- Plot: ECDF ---
    _selected_model = density_model_dropdown.value
    if _selected_model == "all":
        _title = "ECDF of Verblized Confidences"
    else:
        try:
            _title = f"ECDF of Verblized Confidences for {_selected_model}"
        except ValueError:
            raise f"Unknown model {_selected_model}"

    _g = sns.displot(
        data=plot_df,
        x="confidence",
        hue="dataset",
        col="prompt",
        col_order=prompt_order,
        col_wrap=4,
        kind="ecdf",  # Empirical Cumulative Distribution Function
        linewidth=2,
        height=3.5,
        aspect=1,
        palette="Set2",
    )

    _g.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Cumulative Probability"
    )
    _g.set_titles(col_template="{col_name}")

    # Add reference diagonal line (Uniform Distribution)
    for ax in g.axes.flat:
        ax.plot(
            [0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5, label="Uniform"
        )
        ax.grid(True, linestyle=":", alpha=0.6)

    _g.fig.suptitle(_title, y=1.02)
    _g.figure.subplots_adjust(hspace=0.3, wspace=0.2)
    for _ax in _g.axes.flat:
        _ax.plot(
            [0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5, label="Uniform"
        )
        _ax.grid(True, linestyle=":", alpha=0.6)

    _g
    return


@app.cell
def _(g, metrics_df, plot_df, prompt_order):
    plot_df["Outcome"] = plot_df["gpt_eval"].map({1: "Correct", 0: "Incorrect"})

    # --- Plot: Split ECDF (Correct vs. Incorrect) ---
    _g = sns.displot(
        data=plot_df,
        x="confidence",
        hue="Outcome",
        col="prompt",
        col_order=prompt_order,
        col_wrap=2,
        stat="probability",  # Normalizes each class independently (vital if error rate is low!)
        common_norm=False,
        bins=15,  # Discrete bins (e.g., 0.05 width) preserve round-number peaks
        element="step",
        kind="hist",  # Density plot
        fill=True,
        alpha=0.3,
        palette={"Correct": "#2ca02c", "Incorrect": "#d62728"},
        height=3.0,
        aspect=1.2,
    )

    g.set(xlim=(0, 1), xlabel="Confidence", ylabel="Probability within Class")
    g.set_titles(col_template="{col_name}", size=18)

    # Customize formatting and add AUROC annotations per facet
    # _g.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Cumulative Prob.")

    for _pr, _ax in _g.axes_dict.items():
        _auroc = metrics_df[metrics_df["prompt"] == _pr]["ap_errors"].mean()

        _ax.text(
            0.05,
            0.85,
            f"AUROC: {_auroc:.2f}",
            transform=_ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        _ax.grid(True, linestyle=":", alpha=0.5)

    _g.fig.suptitle("Confidence Separation: Correct vs. Incorrect Predictions", y=1.02)
    _g.figure.subplots_adjust(hspace=0.25, wspace=0.15)

    _g
    return


@app.cell
def _(density_model_dropdown, plot_df, prompt_order, true_false_palette):
    _title = "Confidence Separation: Correct vs. Incorrect Predictions"

    _selected_model = density_model_dropdown.value

    if _selected_model != "all":
        _title = f"{_title} for {_selected_model}"


    plot_df["Outcome"] = plot_df["gpt_eval"].map({1: "Correct", 0: "Incorrect"})
    plot_df["_dummy"] = ""  # Single categorical anchor for split violins

    # --- Plot: Faceted Split Violin Plot ---
    _g = sns.catplot(
        data=plot_df,
        x="_dummy",
        y="confidence",
        hue="Outcome",
        col="prompt",
        col_order=prompt_order,
        col_wrap=4,
        kind="violin",
        split=True,     # Splits left/right for Correct vs. Incorrect
        inner="quart",
        cut=1,
        palette=true_false_palette,
        height=4.0,
        #aspect=1.2,
        alpha=0.8,
    )

    _g.set(ylim=(-0.1, 1.1), xlabel="", ylabel="Confidence")

    _g.set_titles(col_template="{col_name}", size=18)


    for _pr, _ax in _g.axes_dict.items():
        _ax.grid(True, linestyle=":", alpha=0.5)

    _g.fig.suptitle(_title, y=1.1, size=20)
    _g.figure.subplots_adjust(hspace=0.25, wspace=0.15)

    _g
    return


@app.cell
def _(metrics_df):
    metrics_df
    return


@app.cell
def _(get_run_df, runs_df):
    _metrics = []

    for _, _run in runs_df.iterrows():
        _run_df = get_run_df(_run)
        m = calculate_metrics(_run_df, corr_col="gpt_eval")
        _metrics.append(m)

    metrics_df = pd.concat([runs_df, pd.DataFrame(_metrics)], axis=1)
    return (metrics_df,)


@app.cell(hide_code=True)
def _(metrics_df, prompt_order):
    # Create a figure with 1 row and 2 columns using prefixed variables
    _fig, _ax1 = plt.subplots()

    _fig.suptitle("Error Detection (AP) vs. Model Accuracy")

    # --- Left Plot: Verbalized Confidences ---
    _ax1.plot([1, 0], "--", color="gray", label="Random Baseline")
    sns.scatterplot(
        data=metrics_df, x="auc_roc", y="ap_errors", hue="prompt", hue_order=prompt_order, ax=_ax1, s=80
    )

    _ax1.set_title("Verbalized Confidences")
    _ax1.set_xlim(0, 1)
    _ax1.set_ylim(0, 1)
    _ax1.set_xlabel("Accuracy")
    _ax1.set_ylabel("Error Detection (AP)")
    return


@app.cell(hide_code=True)
def _(metric_mapping, metrics_df, prompt_order):
    # Create a figure with 1 row and 2 columns using prefixed variables
    _fig, _ax1 = plt.subplots()

    x_var = "acc"
    y_var = "nll"

    _fig.suptitle(f"{metric_mapping[x_var]} vs. {metric_mapping[y_var]}")

    # --- Left Plot: Verbalized Confidences ---
    sns.scatterplot(data=metrics_df, x=x_var, y=y_var, hue="prompt", hue_order=prompt_order, ax=_ax1, s=80)

    _ax1.set_title("Verbalized Confidences")

    _ax1.set_xlabel("Accuracy")
    _ax1.set_ylabel("Error Detection (AP)")

    _fig
    return


@app.cell
def _(metric_mapping):
    _metrics = list(metric_mapping.keys())
    scatter_metric_dropdown = mo.ui.dropdown(_metrics, value=_metrics[0])
    scatter_metric_dropdown
    return (scatter_metric_dropdown,)


@app.cell(hide_code=True)
def _(metric_mapping, metrics_df, prompt_order, scatter_metric_dropdown):
    _selected_metric = scatter_metric_dropdown.value
    _ax = sns.stripplot(
        metrics_df,
        x="prompt",
        hue="prompt",
        order=prompt_order,
        y=_selected_metric,
        jitter=0.1,        # Adds horizontal noise to reveal overlapping dots
        size=8,
        alpha=0.6,
    )

    _ax.set_title(f"{metric_mapping[_selected_metric]}")
    _ax.set_ylabel(f"{metric_mapping[_selected_metric]}")
    _ax.set_xlabel("")
    _ax
    return


@app.function
def compute_bin_stats(y_true, y_prob, bins=None, n_bins=10):
    """Compute per-bin statistics for calibration tasks."""
    if bins is None:
        bins = np.linspace(0, 1, n_bins+1)
    else:
        bins = np.asarray(bins)

    df = pd.DataFrame({"conf": y_prob, "y": y_true})
    df["bin"] = pd.cut(df["conf"], bins=bins, include_lowest=True)

    # `observed=False` ensures empty bins are kept, removing the need for `.reindex()`
    stats = (
        df.groupby("bin", observed=False)
        .agg(
            mean_y=("y", "mean"),
            std_y=("y", "std"),
            mean_conf=("conf", "mean"),
            n=("y", "count"),
        )
        .reset_index(drop=True)
    )

    # Calculate intervals directly using array math
    stats["start"] = bins[:-1]
    stats["end"] = bins[1:]
    stats["width"] = np.diff(bins)
    stats["mid"] = stats["start"] + stats["width"] / 2
    stats["std_err"] = stats["std_y"] / (np.sqrt(stats["n"]) + 1e-8)  # Avoid division by zero

    return stats


@app.function
def confidence_bar_plot_alt_old(
    data=None, x=None, y=None, n="n", widths="width", std_err="std_err", **kwargs
):
    ax = kwargs.get("ax") or plt.gca()

    # Extract the actual Series from the facet's DataFrame subset
    x_val = data[x]
    y_val = data[y]
    n_val = data[n]
    w_val = data[widths]
    err_val = data[std_err]

    # Normalize n for colormap
    norm = mcolors.Normalize(vmin=n_val.min(), vmax=n_val.max())
    colors = plt.cm.Blues(norm(n_val))

    # Vectorized bar plot
    ax.bar(
        x_val,
        y_val,
        width=w_val,
        color=colors,
        yerr=err_val,
        edgecolor="black",
    )
    ax.grid(True, linestyle="--", alpha=0.6)

    # Reference diagonal line
    ax.plot([0, 1], [0, 1], "--", alpha=0.7, color="gray")
    ax.set(xlim=(0, 1.01), ylim=(0, 1))

    return ax


@app.function
def confidence_bar_plot_alt(data=None, ax=None, **kwargs):
    ax = kwargs.get("ax") or plt.gca()

    x = data["mid"]
    y = data["mean_y"].fillna(0)
    n = data["n"].fillna(0)
    widths = data["width"]

    # Error bar math correction: standard error of the mean uses sqrt(n)
    y_errs = data["std_y"].fillna(0) / (np.sqrt(n) + 1e-8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Grab base color passed by FacetGrid (default to Seaborn blue if missing)
    base_color = kwargs.get("color", "C0")

    # 2. Normalize n to an alpha range (e.g., 0.15 to 1.0 so empty/small bins are still visible)
    if n.max() > n.min():
        norm_n = (n - n.min()) / (n.max() - n.min())
    else:
        norm_n = np.ones_like(n)

    # Scale alpha to a visible minimum (e.g., min alpha 0.25, max alpha 1.0)
    alphas = 0.25 + 0.75 * norm_n

    # 3. Convert base_color to RGBA and inject normalized alphas per bar
    base_rgb = mcolors.to_rgb(base_color)
    rgba_colors = np.zeros((len(n), 4))
    rgba_colors[:, :3] = base_rgb  # Set RGB channels
    rgba_colors[:, 3] = alphas  # Set custom alpha per bar

    # Vectorized plotting using RGBA array
    ax.bar(
        x, y, width=widths, color=rgba_colors, yerr=y_errs, edgecolor="black"
    )
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.plot([0, 1], [0, 1], "--", alpha=0.7, color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1))

    return ax


@app.function
def confidence_bar_plot(y_true, y_probs, bins=None, ax=None):
    """Plot calibration reliability diagram."""
    # np.hstack natively flattens both single arrays and lists of arrays
    y_true = np.hstack(y_true)
    y_probs = np.hstack(y_probs)

    if bins is None:
        bins = np.linspace(0, 1, 11)

    bin_stats = compute_bin_stats(y_true, y_probs, bins)

    x = bin_stats["mid"]
    y = bin_stats["mean_y"].fillna(0)
    n = bin_stats["n"].fillna(0)
    widths = bin_stats["width"]

    # Error bar math correction: standard error of the mean uses sqrt(n)
    y_errs = bin_stats["std_y"].fillna(0) / (np.sqrt(n) + 1e-8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Matplotlib's Normalize replaces manual min/max calculations
    colors = plt.cm.Blues(mcolors.Normalize()(n))

    # Vectorized plotting: ax.bar accepts arrays, replacing the for-loop
    ax.bar(x, y, width=widths, color=colors, yerr=y_errs, edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.plot([0, 1], [0, 1], "--", alpha=0.7, color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1))

    return ax


@app.function
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
    bins = pd.qcut(y_prob, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins})

    binned = (
        df.groupby("bin")
        .agg(
            mean_y=("y_true", "mean"),
            std_y=("y_true", "std"),
            mean_conf=("y_prob", "mean"),
            n=("y_true", "count"),
            start=("y_prob", "min"),
            end=("y_prob", "max"),
            width=("y_prob", lambda x: x.max() - x.min()),
        )
        .reset_index()
    )

    return binned


@app.function
def confidence_plot(
    y_true, y_probs, ax=None, label=None, bins=10, print_reference=True
):
    """
    y_true: 1D array of true binary labels (0 or 1)
    y_conf: 1D array of predicted confidence scores (between 0 and 1
    """

    bin_stats = compute_equal_frequency_bin_stats(y_true, y_probs, n_bins=bins)

    if ax is None:
        plt.subplot(1, 2, 1)

    if print_reference:
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.6)

    label = label or "Model calibration"
    ax.plot(bin_stats["mean_conf"], bin_stats["mean_y"], ".-", label=label, linewidth=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


@app.function
def calculate_metrics(run_file, conf_col="confidence", corr_col="correct", n_bins=10):
    mask = run_file[conf_col].notna()

    # Cast to numpy arrays to ensure safe math operations later
    y_true = np.array(run_file.loc[mask, corr_col]).astype(int)
    y_conf = np.array(run_file.loc[mask, conf_col]).astype(float)

    metrics = {}

    # --- Standard Metrics ---
    metrics["ece"] = calculate_ece(
        y_true, y_conf, n_bins
    )  # (Assumes defined elsewhere)
    metrics["brier"] = brier_score_loss(y_true, y_conf)
    metrics["ap_success"] = average_precision_score(y_true, y_conf)
    metrics["ap_errors"] = average_precision_score(1 - y_true, 1 - y_conf)
    metrics["auc_roc"] = roc_auc_score(y_true, y_conf)
    metrics["acc"] = run_file[corr_col].mean()

    # Exponentially penalizes high confidence errors
    metrics["nll"] = log_loss(y_true, y_conf)

    return metrics


@app.function
def calculate_ece(y_correct, y_conf, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        y_correct (np.ndarray): 1D array of binary indicators (1 if the model's
                                prediction was correct, 0 if it was wrong).
        y_conf (np.ndarray): 1D array of predicted confidences (probabilities).
        n_bins (int): Number of bins to divide the [0, 1] interval.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # digitize returns 1 to n_bins+1. Subtract 1 to get 0-indexed bins.
    # We use np.clip to ensure that confidence exactly equal to 1.0 is
    # placed in the last bin instead of falling out of bounds.
    bin_indices = np.clip(np.digitize(y_conf, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    n_total = len(y_correct)

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_accuracy = np.mean(y_correct[bin_mask])
            bin_confidence = np.mean(y_conf[bin_mask])

            # Weighting by the fraction of samples in this bin
            ece += np.abs(bin_accuracy - bin_confidence) * np.sum(bin_mask) / n_total

    return ece


@app.cell
def _():
    prompt_mapping = {
        "cot": "Chain of Thought",
        "multistep": "Multistep",
        "direct": "Direct",
        "top-k": "Top-k",
    }
    metric_mapping = {
        "ece": "Expected Calibration Error (ECE)",
        "brier": "Brier Score",
        "ap_success": "Average Precision (Success)",
        "ap_errors": "Average Precision (Errors)",
        "auc_roc": "AUROC",
        "acc": "Accuracy",
        "nll": "Negative Log-Likelihood (NLL)",
    }
    return metric_mapping, prompt_mapping


@app.cell
def _():
    #color palettes
    dataset_palette ="set2"
    true_false_palette  = {"Correct": "#3CB371", "Incorrect": "#DC143C"}
    return (true_false_palette,)


if __name__ == "__main__":
    app.run()
