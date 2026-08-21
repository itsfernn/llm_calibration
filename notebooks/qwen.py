import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    import matplotlib.pyplot as plt


    import seaborn as sns

    import pandas as pd
    import json
    import re
    import glob
    from pathlib import Path

    from src.metrics import bootstrap_metrics, calculate_metrics
    from src.utils import compute_equal_frequency_bin_stats


@app.cell
def _():
    df = load_data()
    return (df,)


@app.cell
def _(df):
    main_df = df[df["tags"].apply(lambda xs: "main" in xs)]
    prompt_types_df = df[df["tags"].apply(lambda xs: "prompt_types" in xs)]
    thinking_comp_df = df[df["tags"].apply(lambda xs: "thinking_comp" in xs)]
    return main_df, prompt_types_df, thinking_comp_df


@app.cell
def _(get_results_df, main_df):
    with mo.persistent_cache("main_plot_boot_v2"):
        main_plot_df = get_results_df(main_df)
    return (main_plot_df,)


@app.cell
def _(metric_mapping):
    _metric_names = list(metric_mapping.keys())
    metric_dropdown = mo.ui.dropdown(value=_metric_names[0], options=_metric_names)
    metric_dropdown
    return (metric_dropdown,)


@app.cell(hide_code=True)
def _(main_plot_df, metric_dropdown, metric_mapping, plot_metric_vs_size):
    _current_metric = metric_dropdown.value
    _pretty_metric = metric_mapping.get(_current_metric, _current_metric)
    plot_metric_vs_size(
        main_plot_df,
        _current_metric,
        metric_label=_pretty_metric,
        title=f"{_pretty_metric} across Model Sizes",
    )
    return


@app.cell
def _(main_plot_df):
    main_plot_df
    return


@app.cell(hide_code=True)
def _(metric_dropdown):
    mo.md(f"""
    Shows {metric_dropdown.value} against model size for all datasets.
    """)
    return


@app.cell
def _(main_plot_df):
    # Create a figure with 1 row and 2 columns using prefixed variables
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _fig.suptitle("Error Detection (AP) vs. Model Accuracy")

    # --- Left Plot: Verbalized Confidences ---
    _verb_df = main_plot_df[main_plot_df["conf_type"] == "Verbalized Confidence"]
    _ax1.plot([1, 0], "--", color="gray", label="Random Baseline")
    sns.scatterplot(data=_verb_df, x="acc", y="ap_errors", hue="dataset", ax=_ax1, s=80)

    _ax1.set_title("Verbalized Confidences")
    _ax1.set_xlim(0, 1)
    _ax1.set_ylim(0, 1)
    _ax1.set_xlabel("Accuracy")
    _ax1.set_ylabel("Error Detection (AP)")

    # --- Right Plot: Log Probabilities ---
    _log_df = main_plot_df[main_plot_df["conf_type"] == "Log Probabilities"]
    _ax2.plot([1, 0], "--", color="gray", label="Random Baseline")
    sns.scatterplot(data=_log_df, x="acc", y="ap_errors", hue="dataset", ax=_ax2, s=80)

    _ax2.set_title("Log Probabilities")
    _ax2.set_xlim(0, 1)
    _ax2.set_xlabel("Accuracy")

    # Clean up layout and display
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparison of error detection capability (AP) and task accuracy for verbalized confidences (left) and log probabilities (right). Data points represent individual evaluation runs spanning multiple model sizes and datasets (ai2_arc, gsm8k, mmlu) against a random baseline. While verbalized confidences hover slightly above the baseline, log probabilities show a pronounced and consistent advantage in error detection.
    """)
    return


@app.cell
def _(main_plot_df, metric_mapping):
    # Create a figure with 1 row and 2 columns using prefixed variables
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _m = "brier"
    _m_pretty = metric_mapping.get(_m, _m)

    _fig.suptitle(f"{_m_pretty} vs. Model Accuracy")

    # --- Left Plot: Verbalized Confidences ---
    _verb_df = main_plot_df[main_plot_df["conf_type"] == "Verbalized Confidence"]
    sns.scatterplot(data=_verb_df, x="acc", y=_m, hue="dataset", ax=_ax1, s=80)

    _ax1.set_title("Verbalized Confidences")

    _ax1.set_xlabel("Accuracy")
    _ax1.set_ylabel(_m_pretty)

    # --- Right Plot: Log Probabilities ---
    _log_df = main_plot_df[main_plot_df["conf_type"] == "Log Probabilities"]
    sns.scatterplot(data=_log_df, x="acc", y=_m, hue="dataset", ax=_ax2, s=80)

    _ax2.set_title("Log Probabilities")
    _ax2.set_xlabel("Accuracy")

    # Clean up layout and display
    plt.tight_layout()
    _fig
    return


@app.cell
def _(main_df):
    _datasets = main_df["dataset"].unique()
    dataset_dropdown = mo.ui.dropdown(_datasets, value=_datasets[0])
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(
    conf_mapping,
    dataset_dropdown,
    dataset_mapping,
    main_df,
    plot_reliability_diagrams,
):
    _dataset = dataset_dropdown.value
    _data = main_df[main_df["dataset"] == _dataset]

    _conf = "log"
    _title = f"{conf_mapping[_conf]}: {dataset_mapping[_dataset]}"
    _log_fig = plot_reliability_diagrams(_data, conf_col="log_conf", hue="model_size", cmap="flare", title=_title)

    _conf = "verb"
    _title = f"{conf_mapping[_conf]}: {dataset_mapping[_dataset]}"
    _verb_fig = plot_reliability_diagrams(_data, conf_col="verb_conf", hue="model_size", cmap="flare", title=_title)

    mo.hstack((_log_fig, _verb_fig))
    return


@app.function
def get_bootstrap_metrics(df, conf_type="verb", n_bootstrap=1000, n_bins=10):
    metrics = []
    for i, row in df.iterrows():
        o = load_run_data(row)
        mask = o[f"{conf_type}_conf"].notna()
        y_true = np.array(o.loc[mask, "correct"]).astype(int)
        y_conf = np.array(o.loc[mask, f"{conf_type}_conf"]).astype(float)
        m = bootstrap_metrics(
            y_true,
            y_conf,
            ece="discrete" if conf_type == "verb" else "binned",
            n_bootstrap=n_bootstrap,
            n_bins=n_bins,
        )
        m["type"] = row["type"]
        metrics.append(m)
    return pd.DataFrame(metrics)


@app.cell
def _(prompt_types_df):
    with mo.persistent_cache("prompt_types_metrics"):
        prompt_types_metric_df = get_bootstrap_metrics(prompt_types_df, n_bootstrap=1000)
    prompt_types_metric_df
    return (prompt_types_metric_df,)


@app.function
def plot_metric_with_ci(df, metric="acc", title=None):
    plt.figure(figsize=(8, max(4, len(df) * 0.5)), dpi=120)

    # 1. Preserve original DataFrame order
    df_plot = df.reset_index(drop=True)

    # 2. Map colors consistently based on unique types across the dataset
    # (Extracting unique types globally ensures colors never swap when filtering/switching metrics)
    unique_types = list(df['type'].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_types))
    color_map = dict(zip(unique_types, palette))

    y_positions = np.arange(len(df_plot))

    # 3. Plot each prompt in its original order with its fixed color
    for i, row in df_plot.iterrows():
        mean = row[metric]
        xerr_lower = mean - row[f"{metric}_ci_lower"]
        xerr_upper = row[f"{metric}_ci_upper"] - mean
        xerr = [[xerr_lower], [xerr_upper]]

        prompt_type = row['type']
        color = color_map[prompt_type]

        plt.errorbar(
            x=mean,
            y=y_positions[i],
            xerr=xerr,
            fmt='o',
            color=color,
            ecolor=color,
            alpha=0.85,
            elinewidth=2,
            capsize=4,
            markersize=7
        )

    if title is None:
        title = f"{metric.upper()} Comparison Across Prompts"

    plt.yticks(y_positions, df_plot['type'])
    plt.xlabel(metric.upper())
    plt.title(f"{title} (with 95% CI)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Display top item of DataFrame at the top of the plot
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


@app.cell
def _(metric_mapping, prompt_types_metric_df):
    figs=[]
    for metric_key, metric_title in metric_mapping.items():
        if metric_key not in prompt_types_metric_df.columns:
            continue
        # Pass the metric key and label to your plotting function
        fig = plot_metric_with_ci(prompt_types_metric_df, metric=metric_key, title=metric_title)
        figs.append(fig)

    # Render in Marimo as interactive tabs
    mo.vstack(figs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Int seems to slightly outperform the other prompting methods, but according to a n=1000 bootstrap the effect is not statistically significant.
    """)
    return


@app.cell
def _(prompt_types_metric_df):
    prompt_types_metric_df
    return


@app.cell
def _(index_name_mapping, metric_mapping):
    def format_summary_table(df, indices, metrics=["acc", "ece", "brier", "auc_roc", "ap_errors"]):
        formatted_df = pd.DataFrame()
        for i in indices:
            col_name = index_name_mapping[i]
            formatted_df[col_name] = df[i]

        for metric_key in metrics:
            col_name = metric_mapping[metric_key]
            col_values = []
            for _, row in df.iterrows():
                mean_val = row[metric_key]
                se_val = row[f"{metric_key}_se"]  # Reads directly from _se column

                col_values.append(f"{mean_val:.3f} ({se_val:.3f})")

            formatted_df[col_name] = col_values

        return formatted_df

    return (format_summary_table,)


@app.cell
def _(format_summary_table, prompt_types_metric_df):
    format_summary_table(prompt_types_metric_df, indices=["type"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Thinking vs. Non Thinking
    """)
    return


@app.cell
def _(thinking_comp_df):
    thinking_comp_df
    return


@app.cell
def _(thinking_comp_df):
    def get_n_thinking_tokens(row):
        if "n_thinking_tokens" in row:
            return row["n_thinking_tokens"]
        elif "thinking" in row:
            if isinstance(row["thinking"], str):
                return len(row["thinking"].split()) * 1.33

        return 0

    for _i, _row in thinking_comp_df.iterrows():
        run_data = load_run_data(_row)
        n_thinking_tokens = run_data.apply(get_n_thinking_tokens, axis=1).mean()
        thinking_comp_df.loc[_i, "n_thinking_tokens"] = n_thinking_tokens
    return


@app.cell
def _(thinking_comp_df):
    with mo.persistent_cache("thinking_comp_metrics_verb"):
        thinking_comp_boot_metrics_verb = get_bootstrap_metrics(thinking_comp_df, n_bootstrap=1000)
    return (thinking_comp_boot_metrics_verb,)


@app.cell
def _(thinking_comp_boot_metrics_verb, thinking_comp_df):
    thinking_comp_plot_df = pd.concat(
        [
            thinking_comp_df["n_thinking_tokens"].reset_index(drop=True),
            thinking_comp_boot_metrics_verb], 
        axis = 1
    )
    return (thinking_comp_plot_df,)


@app.cell
def _(plot_reliability_diagrams, thinking_comp_df):
    _log_fig = plot_reliability_diagrams(thinking_comp_df, conf_col="log_conf", hue="n_thinking_tokens", cmap="crest", title="Log Probabilites")

    _verb_fig = plot_reliability_diagrams(thinking_comp_df, conf_col="verb_conf", hue="n_thinking_tokens", cmap="crest", title="Verbalized Confidences")

    mo.hstack([_log_fig, _verb_fig])
    return


@app.cell
def _(metric_mapping, thinking_comp_plot_df):
    # 1. Melt the DataFrame to convert metric columns into rows
    _df_melted = thinking_comp_plot_df.melt(
        id_vars=["n_thinking_tokens"],
        value_vars=["acc", "auc_roc"],
        var_name="metric",
        value_name="value",
    )

    _df_melted2 = thinking_comp_plot_df.melt(
        id_vars=[ "acc"],
        value_vars=["ece", "brier"],
        var_name="metric",
        value_name="value",
    )

    # 2. Map the technical metric names to clean, human-readable titles
    _df_melted["metric_label"] = _df_melted["metric"].map(metric_mapping)

    # 3. Create a single figure and draw all lines differentiated by hue
    _fig, _ax = plt.subplots()
    sns.lineplot(
        data=_df_melted,
        x="n_thinking_tokens",
        y="value",
        hue="metric",
        marker="o",
        ax=_ax,
    )

    _fig2, _ax2 = plt.subplots()
    sns.lineplot(
        data=_df_melted2,
        x="acc",
        y="value",
        hue="metric",
        marker="o",
        ax=_ax2,
    )

    _ax.set_title("Metrics vs. Number of Thinking Tokens")
    _ax.set_xlabel("Number of Thinking Tokens")
    _ax.set_ylabel("Value")

    plt.close(_fig)  # Prevents stray auto-displays in marimo

    # Display the single combined figure
    mo.hstack([_fig, _fig2])
    return


@app.cell
def _(metric_mapping, thinking_comp_plot_df):
    _df = thinking_comp_plot_df
    _figs = []

    for _m, _v in metric_mapping.items():
        if _m not in _df.columns:
            continue
        # 1. Create a fresh figure and axis for each metric
        _fig, _ax = plt.subplots()

        # 2. Draw directly onto the created axis
        sns.lineplot(data=_df, x="n_thinking_tokens", y=_m, marker="o", ax=_ax)
        _ax.set_title(f"{_v} vs. Number of Thinking Tokens")

        # 3. Append the Figure object (or close it to prevent stray auto-displays)
        _figs.append(_fig)
        plt.close(_fig)  # Prevents duplicate renderings in marimo

    # Stack the generated figures vertically
    mo.vstack(_figs)
    return


@app.cell
def _(format_summary_table, thinking_comp_plot_df):
    _df = format_summary_table(thinking_comp_plot_df, indices=["n_thinking_tokens"])
    _df["Avg Thinking Tokens"] = _df["Avg Thinking Tokens"].apply(lambda x: f"{x:.0f}")
    _df.sort_values(["Avg Thinking Tokens"], ignore_index=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Improvements across the board. Increased ECE might again be a sign of higher accuracy

    > idea: vary max thinking tokens, map acc vs. ece_df
    """)
    return


@app.function
def load_data():
    from src.utils import repo_root

    paths = glob.glob(
        str(repo_root() / "data" / "qwen" / "**" / "metadata.json"),
        recursive=True,
    )

    rows = []

    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)

            data["folder"] = path.rsplit("/", 1)[0]  # keep folder info
            data["tags"] = data.get("tags") or []
            rows.append(data)

        except Exception as e:
            print(f"Error reading {path}: {e}")

    df = pd.DataFrame(rows)

    df["dataset"] = df["dataset"].fillna("gsm8k")
    df["max_thinking_tokens"] = df["max_thinking_tokens"].fillna(0)

    def get_model_size(model_name):
        match = re.search(r"([\d.]+)\s*B", model_name)
        return float(match.group(1)) if match else None

    df["model_size"] = df["model"].apply(get_model_size)

    return df


@app.function
def load_run_data(row):
    file_path = row["folder"] + "/outputs.jsonl"
    outputs = pd.read_json(file_path, lines=True)

    dataset = row["dataset"]
    if dataset == "gsm8k":
        preprocess_gsm8k(outputs)
    elif dataset == "ai2_arc":
        preprocess_ai2_arc(outputs)
    elif dataset == "cais/mmlu":
        preprocess_mmlu(outputs)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    return outputs


@app.cell
def _(conf_mapping, dataset_mapping):
    # selected_df = main_df[main_df["dataset"] == dataset_dropdown.value].reset_index()
    def get_results_df(df):
        def get_metrics(df, type):
            metrics = []
            for i, row in df.iterrows():
                o = load_run_data(row)
                mask = o[f"{type}_conf"].notna()
                y_true = np.array(o.loc[mask, "correct"]).astype(int)
                y_conf = np.array(o.loc[mask, f"{type}_conf"]).astype(float)
                m = bootstrap_metrics(
                    y_true,
                    y_conf,
                    ece="both" if type == "verb" else "binned",
                    n_bootstrap=1000,
                    n_bins=10,
                )
                metrics.append(m)

            return metrics

        _df = df.reset_index()
        _log_metrics = get_metrics(_df, "log")
        _verb_metrics = get_metrics(_df, "verb")

        _log_metrics_df = pd.DataFrame(_log_metrics)
        _verb_metrics_df = pd.DataFrame(_verb_metrics)

        _log_results = pd.concat([_df, _log_metrics_df], axis=1)
        _log_results["conf_type"] = "log"

        _verb_results = pd.concat([_df, _verb_metrics_df], axis=1)
        _verb_results["conf_type"] = "verb"

        result_df = pd.concat([_log_results, _verb_results], axis=0)


        result_df["dataset"] = (
            result_df["dataset"]
                .map(dataset_mapping)
                .fillna(result_df["dataset"])
        )

        result_df["conf_type"] = (
            result_df["conf_type"]
                .map(conf_mapping)
                .fillna(result_df["conf_type"])
        )

        return result_df

    return (get_results_df,)


@app.cell
def _():
    custom_palette = ["#4A6984", "#D66853", "#619B8A", "#D9A05B"]

    # 2. Apply the palette and style configurations globally
    sns.set_theme(
        style="whitegrid",  # Clean white grid background
        palette=custom_palette,  # Integrates your custom colors
        rc={
            "axes.edgecolor": "#D3D3D3",  # Muted, lighter border lines
            "grid.color": "#EAEAEA",  # Softer, non-distracting gridlines
            "figure.facecolor": "#FFFFFF",  # Crisp white figure background
            "axes.facecolor": "#FFFFFF",  # Crisp white plot canvas
        },
    )

    sns.color_palette(custom_palette)
    return


@app.function
def preprocess_gsm8k(df):
    df["log_conf"] = df["logprobs"].apply(
        lambda xs: np.exp(sum(xs) / len(xs)) if xs else 0.5
    )

    def extract_answer(text):
        match = re.search("####\\s*(\\-?\\d+\\.?\\d*)", text)
        return float(match.group(1)) if match else None

    df["full_answer"] = df["answer"]
    df["answer"] = df["full_answer"].apply(extract_answer)

    def parse_prediction(pred):
        match = re.findall("-?\\d+\\.?\\d*", str(pred))
        return float(match[-1]) if match else None

    df["prediction"] = df["prediction"].apply(parse_prediction)
    df["correct"] = df["answer"] == df["prediction"]


@app.function
def preprocess_mmlu(df):
    df["correct"] = df["answer"] == df["prediction"]

    def get_log_conf(row):
        labels = row["labels"]
        probs = row["label_probs"]
        prediction = row["prediction"][0]

        try:
            idx = labels.index(prediction)
            return probs[idx]
        except:
            print(labels, probs, prediction)

    df["log_conf"] = df.apply(get_log_conf, axis=1)


@app.function
def preprocess_ai2_arc(df):
    df["correct"] = df["answer"] == df["prediction"]

    def get_log_conf(row):
        labels = row["labels"]
        probs = row["label_probs"]
        prediction = row["prediction"][0]

        try:
            idx = labels.index(prediction)
            return probs[idx]
        except:
            print(labels, probs, prediction)

    df["log_conf"] = df.apply(get_log_conf, axis=1)


@app.cell
def _():
    def prepare_eq_bins(selection_df, cor_col="correct", conf_col="confidence"):
        combined_dfs = []

        for i, run in selection_df.iterrows():
            df = load_run_data(run)

            row = compute_equal_frequency_bin_stats(
                y_true=df[cor_col],
                y_prob=df[conf_col],
                n_bins=10
            )

            # Attach run metadata so Seaborn can group by them
            row["unit"] = i
            for col, val in run.items():
                row[col] = [val] * len(row)

            combined_dfs.append(row)

        return pd.concat(combined_dfs, ignore_index=True)


    def plot_reliability_diagrams(
        selection_df, conf_col, hue="model_size", cmap="rocket_r", title=None,
        ax=None, figsize=(7, 6),
    ):
        bins_df = prepare_eq_bins(selection_df, conf_col=conf_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Reference line for perfect calibration (y = x)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

        # Plot lines with standard seaborn call (legend suppressed)
        sns.lineplot(
            data=bins_df,
            x="mean_conf",
            y="mean_y",
            hue=hue,
            hue_norm=(bins_df[hue].min(), bins_df[hue].max()),
            palette=cmap,
            units="unit",
            estimator=None,
            marker="o",
            markersize=8,
            markeredgecolor="none",
            linewidth=3,
            legend=False,  # Suppress default legend
            ax=ax,
            alpha=0.8,
        )

        # Create and add the colorbar (skip if the hue column is constant)
        _hue_lo, _hue_hi = bins_df[hue].min(), bins_df[hue].max()
        if _hue_lo < _hue_hi:
            norm = plt.Normalize(_hue_lo, _hue_hi)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(hue.replace("_", " ").title(), rotation=270, labelpad=15)

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        if title is None:
            ax.set_title("Reliability Diagram")
        else:
            ax.set_title(title)

        ax.grid(True, linestyle=":", alpha=0.6)

        return fig

    return (plot_reliability_diagrams,)


@app.function
def plot_metric_vs_size(
    plot_df,
    metric,
    metric_label=None,
    title=None,
    datasets=("AI2 ARC", "GSM8K", "MMLU"),
    figsize=(14, 4.6),
    ci=True,
):
    """Line plot of a bootstrapped metric vs model size, one panel per dataset.

    Draws Verbalized Confidence and Log Probabilities with 95% CI bands
    (from the ``*_ci_lower`` / ``*_ci_upper`` bootstrap columns).
    """
    colors = {"Verbalized Confidence": "#4A6984", "Log Probabilities": "#D66853"}
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, dpi=110)

    for ax, ds in zip(axes, datasets):
        sub = plot_df[plot_df["dataset"] == ds].sort_values("model_size")
        for conf_type in colors:
            d = sub[sub["conf_type"] == conf_type].dropna(
                subset=[metric, f"{metric}_ci_lower", f"{metric}_ci_upper"]
            )
            color = colors[conf_type]
            if ci:
                ax.fill_between(
                    d["model_size"],
                    d[f"{metric}_ci_lower"],
                    d[f"{metric}_ci_upper"],
                    color=color,
                    alpha=0.25,
                )
            ax.plot(
                d["model_size"],
                d[metric],
                marker="o",
                color=color,
                linewidth=2.5,
                markersize=7,
                label=conf_type,
            )
        ax.set_xscale("log")
        ax.set_title(ds, weight="bold")
        ax.set_xlabel("Model Size (log scale)")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax is axes[0]:
            ax.set_ylabel(metric_label or metric)
            ax.legend(frameon=False)

    if title:
        fig.suptitle(title, weight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


@app.function
def export_thesis_artifacts(main_plot_df, main_df, metric_mapping, out_dir=None):
    """Write the bootstrapped metrics CSVs and thesis figures to results/."""
    from src.utils import repo_root

    root = repo_root()
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    if out_dir is not None:
        tables_dir = Path(out_dir) / "tables"
        figures_dir = Path(out_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Tables ---
    main_plot_df.to_csv(tables_dir / "qwen_main_metrics.csv", index=False)

    eight = main_plot_df[main_plot_df["model_size"] == 8.0].copy()
    eight["model"] = "Qwen3-8B"
    eight.to_csv(tables_dir / "qwen8b_metrics.csv", index=False)

    # --- Metric vs model size figures (one per metric, 3 dataset panels) ---
    for metric in ["ece", "brier", "auc_roc", "ap_errors"]:
        fig = plot_metric_vs_size(
            main_plot_df,
            metric,
            metric_label=metric_mapping[metric],
            title=f"{metric_mapping[metric]} across Model Sizes",
        )
        fig.savefig(figures_dir / f"qwen_{metric}.png", bbox_inches="tight")
        plt.close(fig)

    # --- Reliability diagrams per dataset, all model sizes (verb | log) ---
    dataset_files = {"ai2_arc": "arc", "gsm8k": "gsm8k", "cais/mmlu": "mmlu"}
    dataset_names = {"ai2_arc": "AI2 ARC", "gsm8k": "GSM8K", "cais/mmlu": "MMLU"}
    conf_panels = [
        ("verb_conf", "Verbalized Confidence"),
        ("log_conf", "Log Probabilities"),
    ]
    for raw, short in dataset_files.items():
        data = main_df[main_df["dataset"] == raw]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        for ax, (conf_col, conf_name) in zip(axes, conf_panels):
            plot_reliability_diagrams(
                data,
                conf_col=conf_col,
                hue="model_size",
                cmap="flare",
                title=f"{conf_name}: {dataset_names[raw]}",
                ax=ax,
            )
        fig.suptitle(
            f"Qwen3 Reliability Diagrams - {dataset_names[raw]} (all model sizes)",
            weight="bold",
            fontsize=14,
            y=1.02,
        )
        fig.savefig(figures_dir / f"qwen_model_size_{short}.png", bbox_inches="tight")
        plt.close(fig)

    # --- Qwen3-8B reliability diagrams (3 datasets x 2 conf types) ---
    eight_main = main_df[main_df["model_size"] == 8.0]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13))
    for i, raw in enumerate(dataset_files):
        data = eight_main[eight_main["dataset"] == raw]
        for j, (conf_col, conf_name) in enumerate(conf_panels):
            plot_reliability_diagrams(
                data,
                conf_col=conf_col,
                hue="model_size",
                cmap="flare",
                title=f"{conf_name}: {dataset_names[raw]}",
                ax=axes[i, j],
            )
    fig.suptitle("Qwen3-8B Reliability Diagrams", weight="bold", fontsize=16)
    fig.savefig(figures_dir / "qwen8b_reliability.png", bbox_inches="tight")
    plt.close(fig)

    # --- Error detection (AP) vs accuracy scatter ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Error Detection (AP) vs. Model Accuracy", weight="bold")
    for ax, ct in zip([ax1, ax2], ["Verbalized Confidence", "Log Probabilities"]):
        d = main_plot_df[main_plot_df["conf_type"] == ct]
        ax.plot([1, 0], "--", color="gray", label="Random Baseline")
        sns.scatterplot(data=d, x="acc", y="ap_errors", hue="dataset", ax=ax, s=80)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title(ct)
        ax.legend(frameon=False, title="Dataset")
    ax1.set_ylabel("Error Detection (AP)")
    plt.tight_layout()
    fig.savefig(figures_dir / "qwen_error_vs_acc.png", bbox_inches="tight")
    plt.close(fig)

    return tables_dir, figures_dir


@app.cell(hide_code=True)
def _():
    export_checkbox = mo.ui.checkbox(
        value=True,
        label="Export thesis figures and tables to results/",
    )
    return (export_checkbox,)


@app.cell(hide_code=True)
def _(export_checkbox, export_thesis_artifacts, main_df, main_plot_df, metric_mapping):
    if export_checkbox.value:
        _tables_dir, _figures_dir = export_thesis_artifacts(
            main_plot_df, main_df, metric_mapping
        )
        mo.md(f"Exported to `{_tables_dir}` and `{_figures_dir}`.")
    export_checkbox
    return


@app.cell
def _():
    metric_mapping = {
        "ece": "Expected Calibration Error (ECE)",
        "brier": "Brier Score",
        "ap_success": "Average Precision (Success)",
        "ap_errors": "Average Precision (Errors)",
        "auc_roc": "AUROC",
        "acc": "Accuracy",
        "d_ece": "Discretized ECE (verbalized)",
    }

    index_name_mapping = {
        "type": "Prompt Type",
        "n_thinking_tokens": "Avg Thinking Tokens"
    }

    dataset_mapping = {
        "ai2_arc": "AI2 ARC",
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
        "cais/mmlu": "MMLU",
    }

    conf_mapping = {"verb": "Verbalized Confidence", "log": "Log Probabilities"}
    return conf_mapping, dataset_mapping, index_name_mapping, metric_mapping


@app.cell
def _(Path, df):
    def remove_all_tags(selection):
        for i, row in selection.iterrows():
            file_path = Path(row["folder"]) / "metadata.json"

            if not file_path.exists():
                print(f"Warning: File not found at {file_path}")
                continue

            # 1. Read the existing JSON
            with open(file_path, "r") as f:
                data = json.load(f)

            data["tags"] = []

            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

    def add_tag(selection, tag):
        if selection.empty:
            print("No runs selected!")
            return

        for i, row in selection.iterrows():
            # Path/os.path.join is safer than manual string concatenation
            file_path = Path(row["folder"]) / "metadata.json"

            if not file_path.exists():
                print(f"Warning: File not found at {file_path}")
                continue

            # 1. Read the existing JSON
            with open(file_path, "r") as f:
                data = json.load(f)

            # 2. Safely initialize the tags list if it's missing
            if "tags" not in data or not isinstance(data["tags"], list):
                data["tags"] = []

            # 3. Add the tag if it's not already there
            if tag not in data["tags"]:
                data["tags"].append(tag)

            # 4. Write the modified dictionary back to disk
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

        print(f"Successfully added tag '{tag}' to {len(selection)} runs!")

    # 1. Selection Table (using your 'folder' column)
    run_table = mo.ui.table(
        df,  # Assuming 'df' is your runs dataframe
        selection="multi",
        label="1. Select runs to modify",
    )

    # 2. Tag Input
    tag_input = mo.ui.text(
        placeholder="e.g., baseline, bad_init",
        label="2. Enter tag(s) to ADD (comma-separated)",
    )

    # 3. Action Buttons
    save_button = mo.ui.button(
        label="Add Tags to Selected",
        kind="success",
        on_click=lambda _: add_tag(run_table.value, tag_input.value),
    )

    clear_button = mo.ui.button(
        label="Clear All Tags from Selected",
        kind="danger",
        on_click=lambda _: remove_all_tags(run_table.value),
    )

    # Render everything nicely
    mo.vstack(
        [run_table, tag_input, mo.hstack([save_button, clear_button], justify="start")]
    )
    return


if __name__ == "__main__":
    app.run()
