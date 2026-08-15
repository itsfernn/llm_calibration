import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    from src import plot as pl
    from src.eval import (
        SCALING_METHODS,
        cv_evaluate_runs,
        load_runs,
        run_transfer_grid,
        run_transfer_pooled,
    )


@app.cell
def _():
    method_mapping = {
        "none": "Raw",
        "histogram": "Histogram",
        "isotonic": "Isotonic",
        "platt": "Platt",
        "temperature": "Temperature",
    }
    metric_mapping = {
        "ece": "ECE",
        "brier": "Brier",
        "nll": "NLL",
        "auc_roc": "AUROC",
        "ap_errors": "AP (errors)",
        "ap_success": "AP (success)",
        "acc": "Accuracy",
    }
    prompt_mapping = {
        "direct": "Direct",
        "top-k": "Top-k",
        "multistep": "Multistep",
        "cot": "Chain-of-Thought",
    }
    prompts = ["direct", "top-k", "multistep", "cot"]
    method_palette = {
        "none": "#8C8C8C",
        "histogram": "#E64A8D",
        "isotonic": "#7D3C98",
        "platt": "#00ACC1",
        "temperature": "#9CCC65",
    }
    prompt_colors = {
        "direct": "#4A6984",
        "top-k": "#D66853",
        "multistep": "#619B8A",
        "cot": "#D9A05B",
    }
    return (
        method_mapping,
        method_palette,
        metric_mapping,
        prompt_mapping,
        prompts,
    )


@app.cell(hide_code=True)
def _():
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.edgecolor": "#D3D3D3",
            "grid.color": "#EAEAEA",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        },
    )
    return


@app.cell
def _():
    cwd = Path.cwd()
    if (cwd / "runs-legacy").exists():
        runs_dir = cwd / "runs-legacy"
    else:
        runs_dir = cwd.parent / "runs-legacy"

    runs_df = load_runs(runs_dir)
    mo.md(f"**{len(runs_df)} runs** loaded from `{runs_dir}`")
    return runs_df, runs_dir


@app.cell(hide_code=True)
def _(metric_mapping, runs_df):
    _datasets = list(runs_df["dataset"].unique())
    _models = list(runs_df["model"].unique())

    dataset_dropdown = mo.ui.dropdown(value=_datasets[0], options=_datasets)
    model_dropdown = mo.ui.dropdown(value=_models[0], options=_models)
    metric_dropdown = mo.ui.dropdown(
        value="ece", options=list(metric_mapping.keys())
    )

    mo.hstack(
        [
            dataset_dropdown,
            model_dropdown,
            metric_dropdown,
        ],
        justify="start",
    )
    return dataset_dropdown, metric_dropdown, model_dropdown


@app.cell
def _(runs_df, runs_dir):
    with mo.persistent_cache("scaling_cv_eval_v3folds"):
        fold_metrics_df, summary_df, folds_by_key = cv_evaluate_runs(
            runs_df, runs_dir, n_folds=3, n_bootstrap=100
        )
    return folds_by_key, summary_df


@app.cell
def _(runs_df, runs_dir):
    with mo.persistent_cache("scaling_transfer_v3folds"):
        _transfer = {
            axis: run_transfer_grid(runs_df, runs_dir, axis=axis)
            for axis in ["dataset", "prompt", "model"]
        }
    transfer = _transfer
    return (transfer,)


@app.cell
def _(runs_df, runs_dir):
    with mo.persistent_cache("scaling_transfer_pooled_v3folds"):
        pooled = run_transfer_pooled(runs_df, runs_dir)
    return (pooled,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling Methods Evaluation

    Confidence scaling fits a calibrator on a training split and applies it to unseen
    samples. We evaluate 5 methods — no scaling (raw verbalized confidence), histogram
    binning, isotonic regression, Platt (sigmoid) and temperature scaling — on 48 runs
    (3 multi-hop QA datasets x 4 LLMs x 4 prompt types, ~1000 samples each).

    **Methodology.** Point estimates use **3-fold stratified CV** (fit on ~2/3, evaluate
    on the held-out third; the union of test folds covers the full run). Standard errors
    come from a **combined CV + bootstrap** scheme: each held-out fold is bootstrap-
    resampled and the fold-level bootstrap variances are pooled
    (SE = sqrt(mean fold variance / 3)).

    **Metric note.** All calibration numbers below are the **regular (binned) ECE**. We
    do *not* use the discrete ECE (`d_ece`) that applies only to step-valued verbalized
    confidence — the multistep run's confidence is continuous, so it is treated exactly
    like every other run.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reliability diagrams
    """)
    return


@app.cell
def _(
    dataset_dropdown,
    folds_by_key,
    method_mapping,
    model_dropdown,
    prompt_mapping,
    prompts,
):
    def plot_reliability(folds_by_key, dataset, model, methods, n_bins=15):
        fig, axes = plt.subplots(
            1, len(methods), figsize=(3.2 * len(methods), 3.4),
            sharex=True, sharey=True,
        )
        for ax, method in zip(axes, methods):
            for prompt in prompts:
                folds = folds_by_key[(dataset, model, prompt, method)]
                pl.interploated_confidence_plot(
                    [f["y"] for f in folds],
                    [f["probs"] for f in folds],
                    n_bins=n_bins,
                    error_band="stderr",
                    ax=ax,
                )
                ax.lines[-1].set_label(prompt_mapping[prompt])
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, alpha=0.8)
            ax.set_title(method_mapping[method])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=":", alpha=0.6)
        for ax in axes:
            ax.set_xlabel("Confidence")
        axes[0].set_ylabel("Accuracy")
        fig.suptitle(
            f"{model} on {dataset} — mean of 3 CV folds (shaded: stderr across folds)",
            y=1.03,
        )
        _handles, _labels = axes[0].get_legend_handles_labels()
        fig.legend(
            _handles, _labels,
            loc="lower center", ncol=len(prompts), bbox_to_anchor=(0.5, -0.12),
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        return fig

    fig = plot_reliability(
        folds_by_key, dataset_dropdown.value, model_dropdown.value, SCALING_METHODS
    )
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Curves are averaged over the 3 CV folds: each fold's reliability curve (equal-
    frequency bins) is interpolated on a common grid and then averaged. Raw verbalized
    confidence is heavily overconfident (curve below the diagonal). All scaling methods
    pull the curve towards the diagonal; histogram and isotonic are the most aggressive,
    temperature/Platt the most conservative.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Metrics overview
    """)
    return


@app.cell(hide_code=True)
def _(
    method_mapping,
    method_palette,
    metric_dropdown,
    metric_mapping,
    prompt_mapping,
    prompts,
    summary_df,
):
    _metric = metric_dropdown.value
    _se_col = f"{_metric}_se"

    _plot_df = summary_df.copy()
    _xmap = {p: i for i, p in enumerate(prompts)}
    _plot_df["_x"] = _plot_df["prompt"].map(_xmap)
    rng = np.random.default_rng(42)
    _plot_df["_xj"] = _plot_df["_x"] + rng.uniform(-0.18, 0.18, size=len(_plot_df))

    _fig, _ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=_plot_df,
        x="_xj",
        y=f"{_metric}_mean",
        hue="method",
        hue_order=SCALING_METHODS,
        palette=method_palette,
        size=_se_col,
        sizes=(30, 320),
        alpha=0.65,
        legend=False,
        ax=_ax,
    )

    _handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=8,
            color=method_palette[_m], label=method_mapping[_m],
        )
        for _m in SCALING_METHODS
    ]
    _ax.legend(handles=_handles, title="scaling method", loc="upper right")

    _ax.set_xticks(range(len(prompts)))
    _ax.set_xticklabels([prompt_mapping[p] for p in prompts])
    _ax.set_xlim(-0.5, len(prompts) - 0.5)
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("prompt")
    _ax.set_ylabel(metric_mapping[_metric])
    _ax.set_title(
        f"{metric_mapping[_metric]} by prompt — every run (3 datasets x 4 models), "
        "point size = SE"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(metric_dropdown, metric_mapping, summary_df):
    _metric = metric_dropdown.value

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    _pivot_ds = (
        summary_df.pivot_table(
            index="dataset", columns="method", values=f"{_metric}_mean"
        )
        .reindex(SCALING_METHODS, axis=1)
    )
    _pivot_md = (
        summary_df.pivot_table(
            index="model", columns="method", values=f"{_metric}_mean"
        )
        .reindex(SCALING_METHODS, axis=1)
    )

    sns.heatmap(_pivot_ds, annot=True, fmt=".3f", cmap="Blues", ax=_ax1, cbar=False)
    sns.heatmap(_pivot_md, annot=True, fmt=".3f", cmap="Blues", ax=_ax2, cbar=False)
    _ax1.set_title(f"{metric_mapping[_metric]} — mean over runs")
    _ax2.set_title(f"{metric_mapping[_metric]} — mean over runs")
    _ax1.set_xlabel("Scaling method")
    _ax2.set_xlabel("Scaling method")
    _fig.suptitle(f"{metric_mapping[_metric]} by dataset / model", y=1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    dataset_dropdown,
    method_mapping,
    method_palette,
    metric_dropdown,
    metric_mapping,
    model_dropdown,
    summary_df,
):
    _selected = summary_df[
        (summary_df["dataset"] == dataset_dropdown.value)
        & (summary_df["model"] == model_dropdown.value)
    ]
    _metric = metric_dropdown.value

    _fig, _ax = plt.subplots(figsize=(8, 5.5))
    for _method in SCALING_METHODS:
        _sub = _selected[_selected["method"] == _method]
        if _sub.empty:
            continue
        _ax.errorbar(
            _sub["acc_mean"],
            _sub[f"{_metric}_mean"],
            yerr=_sub[f"{_metric}_se"],
            fmt="o",
            capsize=4,
            ms=8,
            color=method_palette[_method],
            label=method_mapping[_method],
            alpha=0.85,
        )

    _ax.set_xlim(0, 1)
    _ax.set_xlabel("Accuracy")
    _ax.set_ylabel(metric_mapping[_metric])
    _ax.set_title(
        f"{metric_mapping[_metric]} vs accuracy — {model_dropdown.value} on "
        f"{dataset_dropdown.value} (error bars: 1 SE)"
    )
    _ax.legend(title="Scaling method")
    _fig
    return


@app.cell(hide_code=True)
def _(metric_mapping, summary_df):
    def format_summary(df, index_cols, metrics):
        formatted = df.copy()
        for m in metrics:
            formatted[metric_mapping[m]] = formatted.apply(
                lambda r: f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_se']:.3f}", axis=1
            )
        cols = index_cols + [metric_mapping[m] for m in metrics]
        return formatted[cols].round(3)

    formatted = format_summary(
        summary_df,
        ["dataset", "model", "prompt", "method"],
        ["ece", "brier", "nll", "auc_roc", "ap_errors"],
    )
    mo.ui.table(formatted, page_size=15, label="Per-run metrics (mean ± SE)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### What the numbers mean

    - **ECE**: mean |confidence − accuracy|. Lower is better; raw confidence is badly
      miscalibrated (ECE 0.09–0.60 across runs).
    - **Brier / NLL**: proper scoring rules — reward both calibration and sharpness.
    - **AUROC / AP (errors)**: ranking quality of the confidence score. Monotone scaling
      (Platt, temperature, isotonic) preserves the ranking, so these should not change
      much — see the ranking section for the exact cost.
    - **Accuracy** is identical across methods (scaling never changes the prediction).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Effectiveness by prompt — and is it consistent?

    In-domain performance of each method, broken down by prompt. The question: are the
    rankings stable across datasets and across models, or does the "best" method depend
    on the run?
    """)
    return


@app.cell(hide_code=True)
def _(prompt_mapping, prompts, summary_df):
    _fig, _axes = plt.subplots(
        2, len(prompts), figsize=(3.4 * len(prompts), 6.2), sharey=False
    )

    for _j, _prompt in enumerate(prompts):
        _sub = summary_df[summary_df["prompt"] == _prompt]
        _p_ds = (
            _sub.pivot_table(index="dataset", columns="method", values="ece_mean")
            .reindex(SCALING_METHODS, axis=1)
        )
        _p_md = (
            _sub.pivot_table(index="model", columns="method", values="ece_mean")
            .reindex(SCALING_METHODS, axis=1)
        )
        _vmax = max(_p_ds.max().max(), _p_md.max().max())

        sns.heatmap(_p_ds, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=_vmax,
                    ax=_axes[0, _j], cbar=False)
        sns.heatmap(_p_md, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=_vmax,
                    ax=_axes[1, _j], cbar=False)
        _axes[0, _j].set_title(prompt_mapping[_prompt])
        _axes[1, _j].set_xlabel("scaling method")

    _axes[0, 0].set_ylabel("dataset")
    _axes[1, 0].set_ylabel("model")
    for _ax in _axes.flat:
        _ax.tick_params(axis="x", labelrotation=30)
    _fig.suptitle("In-domain mean ECE per scaling method (top: by dataset, bottom: by model)", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(prompts, summary_df):
    def _md_table(df):
        rows = ["| " + " | ".join(df.columns) + " |",
                "|" + "|".join(["---"] * len(df.columns)) + "|"]
        for idx, row in df.iterrows():
            rows.append("| " + str(idx) + " | " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(rows)

    _best_ds = summary_df.loc[summary_df.groupby(["dataset", "prompt"])["ece_mean"].idxmin()]
    _best_md = summary_df.loc[summary_df.groupby(["model", "prompt"])["ece_mean"].idxmin()]

    mo.md(
        "**Best in-domain method (lowest ECE)**\n\n"
        + "**by dataset:**\n\n"
        + _md_table(_best_ds.pivot(index="dataset", columns="prompt", values="method").reindex(prompts, axis=1))
        + "\n\n**by model:**\n\n"
        + _md_table(_best_md.pivot(index="model", columns="prompt", values="method").reindex(prompts, axis=1))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Findings**

    - **In-domain, the aggressive methods win regardless of the prompt.** Histogram and
      isotonic are (near-)best for every prompt, dataset and model; platt trails by
      ~0.01–0.03 and temperature is clearly last (see note below). Any prompt's
      confidence distribution can be realigned in-domain — raw ECE 0.33–0.38 drops to
      ~0.03–0.05 with histogram/isotonic — and which re-mapping method you pick matters
      very little.
    - The absolute differences among histogram / isotonic / platt are minuscule
      (ECE within ~0.01–0.03, vs raw 0.16–0.38). Isotonic edges out histogram on
      direct / top-k / cot (7/12 runs) but they are effectively tied everywhere.
    - **Multistep is the one consistent pattern**: histogram is best on all 3 datasets
      (7/12 runs); its already-calm distribution responds best to aggressive re-mapping.

    **A note on temperature scaling.** Temperature underperforms in-domain
    (ECE ≈ 0.13–0.18 vs 0.03–0.05 for histogram/isotonic), and the reason is
    structural: it is a single-parameter sigmoid anchored at 0.5 that can never move a
    confidence across the 0.5 midline. Because verbalized confidences pile up near 1,
    the optimal temperature explodes and the map collapses everything toward the base
    rate — its in-domain ECE correlates +0.78 with |accuracy − 0.5|, i.e. it is good
    only when accuracy is near 0.5. Fitting it on verbalized confidence is an odd fit:
    the scores must first be converted into "fake" logits, and the data show they do
    not obey the logistic structure temperature assumes. This same mechanism drives its
    transfer behaviour (see the generalizability section).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What does scaling cost? (ranking)

    Calibration re-maps the confidence scores. Monotone maps (Platt, temperature and,
    up to ties, isotonic) preserve the *relative order* of scores, so ranking metrics
    (AUROC, AP) are untouched. Histogram binning replaces each score with its bin mean,
    which is **not** monotone and can shuffle the ranking. Is the drawback measurable?
    """)
    return


@app.cell(hide_code=True)
def _(method_mapping, method_palette, summary_df):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for _ax, _m in zip(_axes, ["auc_roc", "ap_errors"]):
        _raw = summary_df[summary_df["method"] == "none"].set_index("run")[f"{_m}_mean"]
        for _method in SCALING_METHODS[1:]:
            _scaled = summary_df[summary_df["method"] == _method].set_index("run")[f"{_m}_mean"]
            _ax.scatter(_raw, _scaled, s=30, alpha=0.6,
                        color=method_palette[_method], label=method_mapping[_method])
        _lim = (min(_raw.min(), _scaled.min()) - 0.01, 1.01)
        _ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        _ax.set_xlim(_lim)
        _ax.set_ylim(_lim)
        _ax.set_title("AUROC" if _m == "auc_roc" else "AP (errors)")
        _ax.set_xlabel("raw confidence")
        _ax.set_ylabel("scaled confidence")
        _ax.legend()

    _fig.suptitle("Ranking quality: scaled vs raw (identity line = ranking fully preserved)", y=1.03)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Ranking cost**

    - **Platt and temperature lie exactly on the diagonal**: they are strictly
      monotone, so AUROC/AP are preserved to floating-point precision (max |Δ| ≈ 2e-4).
    - **Isotonic** is nearly on the diagonal (monotone non-decreasing), but its
      plateaus tie adjacent scores; AUROC changes by at most ~0.03, usually far less.
    - **Histogram binning** is the only method that genuinely reshuffles the ranking:
      mean AUROC drop ≈ 0.006 (max 0.03), mean AP drop ≈ 0.016 (max 0.07). Small, but it
      is a real, systematic cost.
    - If ranking quality matters, prefer a monotone method (temperature / Platt) —
      you lose nothing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generalizability

    A calibrator is only useful if it works where it was not fitted. We test three
    transfer directions: across **models** (fit on one LLM, apply to another), across
    **datasets** (fit on one dataset, apply to another), and a **precomputed** setting
    (fit on all other datasets pooled, deploy on a new one). Every transfer fits on an
    80% train split of the source and is evaluated on the full target run.
    """)
    return


@app.cell(hide_code=True)
def _(method_mapping, transfer):
    _fig, _axes = plt.subplots(1, 5, figsize=(3.3 * 5, 3.3), sharey=True)

    _x = transfer["model"]
    _vmax = _x["ece"].max()

    for _ax, _method in zip(_axes, SCALING_METHODS):
        _p = (
            _x[_x["method"] == _method]
            .pivot_table(index="src", columns="tgt", values="ece", aggfunc="mean")
            .reindex(sorted(_x["src"].unique()))
            .reindex(sorted(_x["tgt"].unique()), axis=1)
        )
        sns.heatmap(_p, annot=True, fmt=".3f", cmap="Reds_r", vmin=0, vmax=_vmax,
                    ax=_ax, cbar=False)
        _ax.set_title(method_mapping[_method])
        _ax.set_xlabel("target model")
        if _ax is _axes[0]:
            _ax.set_ylabel("source model")
        else:
            _ax.set_ylabel("")
    _fig.suptitle(
        "Cross-model transfer: fit on one model, apply to another (same dataset & prompt). ECE on target.",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Model transfer is the *easier* of the two directions — but no method reliably
    generalizes.** Every transferred method roughly halves the raw out-of-the-box ECE
    (0.31 → 0.11–0.17), so model transfer is never a regression. Relative to their
    *in-domain* fit, histogram/isotonic degrade ~2.8x (to ~0.11) and Platt ~1.7x (to
    ~0.11). Temperature's ratio (≈ 1.05x) is the flattering one — but only because its
    in-domain baseline is already poor (~0.16); in absolute ECE it is actually the
    *worst* transferred method here (0.165 vs 0.107–0.114 for the others).

    Transferring across **datasets** turns out to be the harder axis (all methods land
    at 0.18–0.20 there). So for a fixed model + prompt, the open question is the
    dataset direction — addressed next.
    """)
    return


@app.cell(hide_code=True)
def _(method_mapping, transfer):
    _fig, _axes = plt.subplots(1, 5, figsize=(3.3 * 5, 3.3), sharey=True)

    _x = transfer["dataset"]
    _vmax = _x["ece"].max()

    for _ax, _method in zip(_axes, SCALING_METHODS):
        _p = (
            _x[_x["method"] == _method]
            .pivot_table(index="src", columns="tgt", values="ece", aggfunc="mean")
            .reindex(sorted(_x["src"].unique()))
            .reindex(sorted(_x["tgt"].unique()), axis=1)
        )
        sns.heatmap(_p, annot=True, fmt=".3f", cmap="Reds_r", vmin=0, vmax=_vmax,
                    ax=_ax, cbar=False)
        _ax.set_title(method_mapping[_method])
        _ax.set_xlabel("target dataset")
        if _ax is _axes[0]:
            _ax.set_ylabel("source dataset")
        else:
            _ax.set_ylabel("")
    _fig.suptitle(
        "Cross-dataset transfer (mean over models & prompts): fit on source, apply to all samples of target. ECE on target.",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(method_mapping, method_palette, summary_df, transfer):
    _ind = summary_df[["dataset", "model", "prompt", "method", "ece_mean"]].copy()

    def transfer_table(axis_df, target_cols):
        _t = (
            axis_df.groupby(
                [target_cols["dataset"], target_cols["model"], target_cols["prompt"], "method"]
            )["ece"]
            .mean()
            .rename("transfer_ece")
            .reset_index()
        )
        return _ind.merge(
            _t,
            left_on=["dataset", "model", "prompt", "method"],
            right_on=[
                target_cols["dataset"],
                target_cols["model"],
                target_cols["prompt"],
                "method",
            ],
        )

    _tables = {
        "cross-dataset": transfer_table(
            transfer["dataset"],
            {"dataset": "tgt", "model": "tgt_model", "prompt": "tgt_prompt"},
        ),
        "cross-model": transfer_table(
            transfer["model"],
            {"dataset": "dataset", "model": "tgt", "prompt": "tgt_prompt"},
        ),
    }

    _methods = list(reversed(SCALING_METHODS))
    _rng = np.random.default_rng(7)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for _ax, (_label, _tab) in zip(_axes, _tables.items()):
        for _i, _m in enumerate(_methods):
            _sub = _tab[_tab["method"] == _m]
            _te = _sub["transfer_ece"].values
            _yj = _i + _rng.uniform(-0.28, 0.28, size=len(_te))
            _ax.scatter(_te, _yj, s=9, color=method_palette[_m], alpha=0.28, zorder=1)
            _idm = _sub["ece_mean"].mean()
            _tm = _te.mean()
            _ax.plot(
                [_idm, _tm], [_i, _i],
                color=method_palette[_m], lw=2.4, zorder=2, solid_capstyle="round",
            )
            _ax.scatter([_idm], [_i], s=50, color=method_palette[_m],
                        edgecolor="white", linewidth=1.2, zorder=3, marker="o")
            _ax.scatter([_tm], [_i], s=50, color=method_palette[_m],
                        edgecolor="white", linewidth=1.2, zorder=3, marker="D")
        _ax.set_yticks(range(len(_methods)))
        _ax.set_yticklabels([method_mapping[m] for m in _methods])
        _ax.set_title(_label)
        _ax.set_xlim(0, 0.45)
        _ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    _axes[0].set_xlabel("ECE")
    _axes[1].set_xlabel("ECE")
    _handles = [
        plt.Line2D([], [], color="k", marker="o", linestyle="", markersize=8,
                   label="in-domain (mean)"),
        plt.Line2D([], [], color="k", marker="D", linestyle="", markersize=8,
                   label="transferred (mean)"),
        plt.Line2D([], [], color="k", lw=1.5, alpha=0.3, label="per-run transferred ECE"),
    ]
    _axes[1].legend(handles=_handles, loc="center right", fontsize=8)

    _fig.suptitle(
        "Generalization of each scaling method: in-domain mean ECE (circle) vs "
        "transferred mean ECE (diamond); faint dots = transferred ECE of every run",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(method_palette, summary_df, transfer):
    _acc = (
        summary_df[["dataset", "model", "prompt", "acc_mean"]]
        .drop_duplicates(["dataset", "model", "prompt"])
    )
    _acc_map = {(r.dataset, r.model, r.prompt): r.acc_mean for _, r in _acc.iterrows()}

    def mech(axis_df, tgt_key):
        _t = axis_df[axis_df["method"] == "temperature"].copy()
        _t["tgt_acc"] = _t.apply(lambda r: _acc_map[tgt_key(r)], axis=1)
        _t["dev"] = (_t["tgt_acc"] - 0.5).abs()
        _h = axis_df[axis_df["method"] == "histogram"].copy()
        _h["tgt_acc"] = _h.apply(lambda r: _acc_map[tgt_key(r)], axis=1)
        _h["dev"] = (_h["tgt_acc"] - 0.5).abs()
        return _t, _h

    _t_ds, _h_ds = mech(
        transfer["dataset"],
        lambda r: (r["tgt"], r["tgt_model"], r["tgt_prompt"]),
    )
    _t_md, _h_md = mech(
        transfer["model"],
        lambda r: (r["dataset"], r["tgt"], r["tgt_prompt"]),
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for _ax, (_t, _h, _label) in zip(
        _axes, [(_t_ds, _h_ds, "cross-dataset"), (_t_md, _h_md, "cross-model")]
    ):
        _ax.scatter(_h["dev"], _h["ece"], s=14, color=method_palette["histogram"],
                    alpha=0.3, label="histogram (contrast)")
        _ax.scatter(_t["dev"], _t["ece"], s=22, color=method_palette["temperature"],
                    alpha=0.65, label="temperature")
        _xb = np.linspace(_t["dev"].min(), _t["dev"].max(), 50)
        _sl, _ic = np.polyfit(_t["dev"], _t["ece"], 1)
        _ax.plot(_xb, _sl * _xb + _ic, color=method_palette["temperature"], lw=2, ls="--")
        _c = np.corrcoef(_t["dev"], _t["ece"])[0, 1]
        _ax.annotate(f"r = {_c:+.2f}", xy=(0.03, 0.87), xycoords="axes fraction", fontsize=11)
        _ax.set_title(_label)
        _ax.set_xlabel("|target accuracy - 0.5|")
        _ax.set_xlim(0, 0.35)
        _ax.grid(True, linestyle=":", alpha=0.5)
    _axes[0].set_ylabel("transferred ECE")
    _axes[1].legend(loc="upper left", fontsize=8)

    _fig.suptitle(
        "Why temperature 'transfers': it degenerates toward a fixed ~0.5, so it works "
        "only when the target's accuracy is near 0.5 (histogram has no such dependence)",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(prompts, summary_df, transfer):
    _x = transfer["dataset"]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    _summary = (
        _x.pivot_table(index="src_prompt", columns="method", values="ece", aggfunc="mean")
        .reindex(prompts)
        .reindex(SCALING_METHODS, axis=1)
    )
    _in_domain = (
        summary_df.pivot_table(index="prompt", columns="method", values="ece_mean", aggfunc="mean")
        .reindex(prompts)
        .reindex(SCALING_METHODS, axis=1)
    )
    sns.heatmap(_summary, annot=True, fmt=".3f", cmap="Reds_r", ax=_ax1, cbar=False)
    _ax1.set_title("mean cross-dataset transfer ECE by prompt")
    _ax1.set_ylabel("prompt")
    _ax1.set_xlabel("scaling method")

    _pen = (_summary / _in_domain).round(2)
    sns.heatmap(_pen, annot=True, fmt=".2f", cmap="viridis_r", ax=_ax2, cbar=False)
    _ax2.set_title("transfer ECE / in-domain ECE (1.0 = no transfer loss)")
    _ax2.set_xlabel("scaling method")

    _fig.suptitle(
        "Cross-dataset transfer per prompt: temperature is the only method whose transfer loss is ~1x for every prompt",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(prompts, transfer):
    _x = transfer["dataset"]

    def _md_table(df):
        rows = ["| " + " | ".join(df.columns) + " |",
                "|" + "|".join(["---"] * len(df.columns)) + "|"]
        for idx, row in df.iterrows():
            rows.append("| " + str(idx) + " | " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(rows)

    _best = (
        _x.groupby(["src_model", "src_prompt", "method"])["ece"]
        .mean()
        .reset_index()
    )
    _best = _best.loc[_best.groupby(["src_model", "src_prompt"])["ece"].idxmin()]
    _raw = _x[_x["method"] == "none"].groupby(["src_model", "src_prompt"])["ece"].mean().rename("raw OOB")

    _merged = _best.set_index(["src_model", "src_prompt"]).join(_raw).reset_index()
    _tbl = _merged.pivot(index="src_model", columns="src_prompt", values="method").reindex(prompts, axis=1)
    _ece = _merged.copy()
    _ece["ece"] = _ece.apply(
        lambda r: f"{r['ece']:.3f} (raw {r['raw OOB']:.3f})", axis=1
    )
    _ece = _ece.pivot(index="src_model", columns="src_prompt", values="ece").reindex(prompts, axis=1)

    mo.md(
        "**Best cross-dataset transfer method for each fixed (model, prompt)**\n\n"
        + _md_table(_tbl)
        + "\n\n…and the ECE it achieves on the target (raw out-of-the-box ECE in parens):\n\n"
        + _md_table(_ece)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Is there a (prompt + scaling) combination that transfers well between datasets?**

    **Not really — and this is the honest conclusion.** All methods land at
    ~0.16–0.20 ECE when transferred across datasets (vs raw 0.33–0.38 for
    direct/top-k/cot), far from the 0.03–0.05 of a same-dataset fit. No method
    reliably generalizes.

    **The apparent exception — temperature — is an artifact.** Temperature's *average*
    transfer loss looks small (~1.1x of its in-domain value), but that is because its
    in-domain baseline is already poor (it collapses toward a fixed ~0.5). Looked at
    per run, the transfer effect genuinely varies: **~25% of runs improve under
    temperature transfer, ~75% degrade, spanning −0.08 to +0.12 ECE**. The sign is not
    random — temperature's transferred ECE correlates **+0.83** with |target accuracy −
    0.5|: it "works" only when the target's accuracy happens to be near 0.5. In
    absolute terms temperature is *not* the best transfer method either: it ties
    histogram/isotonic for dataset transfer (~0.18) and is the worst for model transfer.

    - When you may pick the method *per (model, prompt)*, the best transfer method varies
      (isotonic, platt or temperature) — but the winner is only ~0.02 ECE better than
      the others on average, and the choice is data-dependent (fragile).
    - **Multistep is special**: it is already well calibrated raw (ECE 0.16 in-domain,
      0.12–0.14 transferred), so *no scaling at all* is often the best "transfer" for it.
      Whatever you do, never calibrate multistep with a histogram fitted on another
      prompt — it inflates ECE to 0.16–0.34.
    """)
    return


@app.cell(hide_code=True)
def _(pooled, prompts, transfer):
    _single = transfer["dataset"].pivot_table(
        index="src_prompt", columns="method", values="ece", aggfunc="mean"
    ).reindex(prompts).reindex(SCALING_METHODS, axis=1)
    _pooled_t = pooled.pivot_table(
        index="prompt", columns="method", values="ece", aggfunc="mean"
    ).reindex(prompts).reindex(SCALING_METHODS, axis=1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    sns.heatmap(_single, annot=True, fmt=".3f", cmap="Reds_r", ax=_ax1, cbar=False)
    _ax1.set_title("single-source transfer (mean over sources)")
    _ax1.set_ylabel("prompt")
    _ax1.set_xlabel("scaling method")
    sns.heatmap(_pooled_t, annot=True, fmt=".3f", cmap="Reds_r", ax=_ax2, cbar=False)
    _ax2.set_title("pooled transfer (fit on all other datasets)")
    _ax2.set_xlabel("scaling method")

    _fig.suptitle("Precomputed calibrator: fit on other datasets, deploy on the target (mean ECE by prompt)", y=1.04)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Precomputed calibrator (fit once on other datasets, deploy on a new one)**

    Pooling the other datasets' runs (same model & prompt, ~2000 samples) before fitting
    gives a ready-to-deploy calibrator, but it does **not** rescue transfer: every method
    still lands at ~0.15–0.19 ECE on the target, ~3–5x its in-domain value. Temperature
    looks best by the *ratio* metric (pooled-transfer ≈ within 10% of a same-dataset
    fit) — again because its in-domain baseline is already ~0.15.

    **Practical takeaway.** If you must precompute a calibrator to fix out-of-the-box
    calibration on a new dataset, there is **no method that reliably gets you there** —
    expect ~0.15–0.20 ECE regardless of choice. Temperature is not the safe universal
    choice it first appeared to be: its low *average* transfer loss hides a
    target-dependent effect (good only when the target's accuracy is near 0.5), and in
    absolute ECE it ties or loses to the others. The first-order lesson is that
    calibration must be **fitted on the target domain**; the choice of transferred
    method is a second-order concern.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Details

    - Data: `runs-legacy/`, labels = `gpt_eval`, confidence = `confidence`
      (verbalized).
    - 3-fold stratified CV, `seed=42`; scaling fitted on the training folds only.
    - Combined bootstrap: 100 resamples per held-out fold (300 per run/method), pooled;
      SE of the CV mean = sqrt(mean fold bootstrap variance / 3).
    - Metric: regular (binned) ECE everywhere; discrete ECE is only for step-valued
      verbalized confidence and is not used.
    - Transfer: calibrator fitted on an 80% train split of the source run(s), applied to
      the full target run. Pooled transfer pools all source runs with the same model &
      prompt.
    """)
    return


if __name__ == "__main__":
    app.run()
