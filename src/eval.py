"""Cross-validated evaluation of confidence scaling methods and transfer analysis.

Everything here is built on top of ``src.scaling`` (the scaling methods themselves),
``src.metrics`` (metric definitions + bootstrap) and ``src.plot`` (reliability plots).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from . import scaling
from .metrics import calculate_metrics

SCALING_METHODS = ["none", "histogram", "isotonic", "platt", "temperature"]

METRIC_COLUMNS = [
    "acc",
    "brier",
    "nll",
    "ece",
    "auc_roc",
    "ap_success",
    "ap_errors",
]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Load ``data.json`` from ``runs_dir`` and return it as a DataFrame."""
    with open(runs_dir / "data.json") as f:
        import json

        data = json.load(f)
    return pd.DataFrame(data)


def load_run_df(runs_dir: Path, run: pd.Series) -> pd.DataFrame:
    """Read the CSV of a single run (from ``load_runs``)."""
    df = pd.read_csv(runs_dir / run["csv_file"])
    df["dataset"] = run["dataset"]
    df["model"] = run["model"]
    df["prompt"] = run["prompt"]
    return df


def apply_scaling(method: str, conf_train, conf_test, y_train) -> np.ndarray:
    """Fit a scaling method on train confidences and apply it to test confidences."""
    if method == "none":
        return np.asarray(conf_test, dtype=float)
    if method == "histogram":
        return scaling.histogram_scaling(conf_train, conf_test, y_train)
    if method == "isotonic":
        return scaling.isotonic_scaling(conf_train, conf_test, y_train)
    if method == "platt":
        return scaling.platt_scaling(conf_train, conf_test, y_train)
    if method == "temperature":
        return scaling.temperature_scaling(conf_train, conf_test, y_train)
    raise ValueError(f"Unknown scaling method: {method!r}")


def cv_folds(conf, y, method: str, n_folds: int = 5, seed: int = 42):
    """Stratified K-fold CV for one run: fit scaling on train, apply on test.

    Returns a list of dicts ``{"probs": calibrated test confidences, "y": test labels}``.
    """
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(conf, y):
        probs = apply_scaling(method, conf[train_idx], conf[test_idx], y[train_idx])
        folds.append(
            {"probs": np.asarray(probs, dtype=float), "y": y[test_idx].astype(int)}
        )
    return folds


def fold_metrics(folds, n_bins: int = 10) -> list[dict]:
    """Per-fold point metrics (one dict per fold)."""
    return [
        calculate_metrics(f["y"], f["probs"], ece="binned", n_bins=n_bins)
        for f in folds
    ]


def _bootstrap_fold(fold_probs, fold_y, n_bootstrap, seed, n_bins):
    """Bootstrap-resample one test fold and recompute metrics on each resample."""
    rng = np.random.default_rng(seed)
    n = len(fold_y)
    rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = calculate_metrics(fold_y[idx], fold_probs[idx], ece="binned", n_bins=n_bins)
        rows.append(m)
    return pd.DataFrame(rows)


def cv_evaluate(
    conf,
    y,
    method: str,
    n_folds=5,
    seed=42,
    n_bootstrap=100,
    n_jobs=-1,
):
    """Full CV evaluation of one (run, method): per-fold metrics + combined bootstrap.

    Returns ``(folds, metric_rows, boot_pooled, summary)``.
    """
    folds = cv_folds(conf, y, method, n_folds=n_folds, seed=seed)
    metric_rows = fold_metrics(folds)
    boot_folds = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_fold)(f["probs"], f["y"], n_bootstrap, seed + i, 10)
        for i, f in enumerate(folds)
    )
    boot_pooled = pd.concat(boot_folds, ignore_index=True)
    summary = summarize_cv(metric_rows, boot_pooled)
    summary["n"] = len(y)
    return folds, metric_rows, boot_pooled, summary


def cv_evaluate_runs(
    runs_df: pd.DataFrame,
    runs_dir: Path,
    methods=SCALING_METHODS,
    n_folds: int = 5,
    n_bootstrap: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
):
    """Run the full CV evaluation over every run in ``runs_df`` and every method.

    All bootstrap tasks are batched into a single ``Parallel`` call to avoid per-call
    pool overhead.

    Returns ``(fold_metrics_df, summary_df, folds_by_key)``:
    - ``fold_metrics_df``: one row per (run, method, fold) with the fold's metrics.
    - ``summary_df``: one row per (run, method) with ``<metric>_mean``, ``<metric>_se``,
      ``<metric>_se_fold`` and ``<metric>_cv_std`` columns.
    - ``folds_by_key``: dict mapping ``(dataset, model, prompt, method)`` to the list of
      fold dicts (``probs`` + ``y``) for reliability diagrams.
    """
    runs = list(runs_df.iterrows())

    # 1. Build all folds and per-fold point metrics (fit + apply only)
    fold_rows = []
    fold_tasks = []  # (run_idx, method_idx, fold_idx, fold)
    folds_by_key = {}
    for i, (_, run) in enumerate(runs):
        df = load_run_df(runs_dir, run)
        conf, y = df["confidence"].values, df["gpt_eval"].values
        for j, method in enumerate(methods):
            folds = cv_folds(conf, y, method, n_folds=n_folds, seed=seed)
            folds_by_key[(run["dataset"], run["model"], run["prompt"], method)] = folds
            for k, (f, m) in enumerate(zip(folds, fold_metrics(folds))):
                row = {
                    "run": run["ID"],
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "prompt": run["prompt"],
                    "method": method,
                    "fold": k,
                    "n": len(f["y"]),
                }
                row.update(m)
                fold_rows.append(row)
                fold_tasks.append((i, j, k, f))

    fold_metrics_df = pd.DataFrame(fold_rows)

    # 2. Bootstrap within every fold, batched into one parallel call
    base_seed = seed
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_fold)(
            f["probs"], f["y"], n_bootstrap, base_seed + t, 10
        )
        for t, (_, _, _, f) in enumerate(fold_tasks)
    )

    # 3. Aggregate per (run, method)
    n_runs = len(runs)
    summary_rows = []
    for i, (_, run) in enumerate(runs):
        for j, method in enumerate(methods):
            sel = fold_metrics_df[
                (fold_metrics_df["run"] == run["ID"])
                & (fold_metrics_df["method"] == method)
            ]
            nf = len(sel)
            if nf == 0:
                continue
            boot_parts = []
            for k in range(nf):
                # task index = i * len(methods) * n_folds + j * n_folds + k
                t = i * len(methods) * n_folds + j * n_folds + k
                boot_parts.append(results[t])
            boot_pooled = pd.concat(boot_parts, ignore_index=True)
            summary = summarize_cv(
                sel[METRIC_COLUMNS].to_dict("records"), boot_pooled
            )
            summary.update(
                {
                    "run": run["ID"],
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "prompt": run["prompt"],
                    "method": method,
                    "n": int(sel["n"].sum()),
                }
            )
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    return fold_metrics_df, summary_df, folds_by_key


def summarize_cv(metric_rows: list[dict], boot_pooled: pd.DataFrame):
    """Combine per-fold point estimates and the pooled bootstrap into one summary row.

    - ``*_mean``: mean of the per-fold point estimates (the reported value).
    - ``*_se``: standard error of that mean, ``sqrt(mean_i Var_boot_i / n_folds)``
      (bootstrap within each fold, pooled).
    - ``*_se_fold``: bootstrap SE at the fold level (n = n_samples / n_folds).
    - ``*_cv_std``: naive standard deviation across the (few) folds.
    """
    fold_df = pd.DataFrame(metric_rows)
    n_folds = len(fold_df)

    summary = {}
    for col in METRIC_COLUMNS:
        if col not in fold_df.columns:
            continue
        fold_vals = fold_df[col]
        boot_vals = boot_pooled[col]
        summary[f"{col}_mean"] = fold_vals.mean()
        summary[f"{col}_se"] = float(np.sqrt(boot_vals.var(ddof=1) / n_folds))
        summary[f"{col}_se_fold"] = float(np.sqrt(boot_vals.var(ddof=1)))
        summary[f"{col}_cv_std"] = fold_vals.std(ddof=1)
    summary["n_folds"] = n_folds
    return summary


def transfer_metrics(
    src_conf,
    src_y,
    tgt_conf,
    tgt_y,
    method: str,
    fit_frac: float = 0.8,
    seed: int = 42,
    n_bins: int = 10,
):
    """Fit a scaling method on a train split of the source run, apply it to ALL target
    samples and evaluate metrics on the full target run.

    Returns ``(metrics_dict, probs)``.
    """
    src_conf = np.asarray(src_conf, dtype=float)
    src_y = np.asarray(src_y, dtype=int)
    tgt_conf = np.asarray(tgt_conf, dtype=float)
    tgt_y = np.asarray(tgt_y, dtype=int)

    rng = np.random.default_rng(seed)
    n_train = int(fit_frac * len(src_conf))
    train_idx = rng.choice(len(src_conf), size=n_train, replace=False)
    probs = apply_scaling(
        method, src_conf[train_idx], tgt_conf, src_y[train_idx]
    )
    probs = np.asarray(probs, dtype=float)
    metrics = calculate_metrics(tgt_y, probs, ece="binned", n_bins=n_bins)
    return metrics, probs


def run_transfer_grid(
    runs_df: pd.DataFrame,
    runs_dir: Path,
    axis: str = "dataset",
    methods=SCALING_METHODS,
    fit_frac: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate transfer of every scaling method between all pairs of runs that differ
    only along ``axis`` (dataset, model or prompt).

    For each (src, tgt) pair the other attributes (model, prompt, ...) match, so the
    result can be aggregated over those with a mean. Returns a long DataFrame with one
    row per (src, tgt, method) containing every metric in METRIC_COLUMNS plus the run
    attributes (``dataset`` is the dataset shared by both runs, so prompt/model grids
    can also be sliced by dataset).
    """
    records = []
    for _, src in runs_df.iterrows():
        for _, tgt in runs_df.iterrows():
            if src[axis] == tgt[axis]:
                continue
            others_match = all(
                col not in (axis, "ID", "csv_file")
                and src[col] == tgt[col]
                for col in runs_df.columns
                if col not in (axis, "ID", "csv_file")
            )
            if not others_match:
                continue

            src_df = load_run_df(runs_dir, src)
            tgt_df = load_run_df(runs_dir, tgt)
            for method in methods:
                metrics, _ = transfer_metrics(
                    src_df["confidence"].values,
                    src_df["gpt_eval"].values,
                    tgt_df["confidence"].values,
                    tgt_df["gpt_eval"].values,
                    method,
                    fit_frac=fit_frac,
                    seed=seed,
                )
                rec = {
                    "src": src[axis],
                    "tgt": tgt[axis],
                    "method": method,
                    "dataset": src["dataset"],
                    "src_model": src["model"],
                    "src_prompt": src["prompt"],
                    "tgt_model": tgt["model"],
                    "tgt_prompt": tgt["prompt"],
                }
                rec.update({k: metrics[k] for k in METRIC_COLUMNS})
                records.append(rec)

    return pd.DataFrame(records)


def transfer_metrics_pooled(
    src_dfs: list[pd.DataFrame],
    tgt_conf,
    tgt_y,
    method: str,
    fit_frac: float = 0.8,
    seed: int = 42,
    n_bins: int = 10,
):
    """Fit a scaling method on a train split of POOLED source runs and apply it to ALL
    target samples.

    ``src_dfs`` is a list of run DataFrames (from ``load_run_df``) that are pooled
    before the fit — this simulates precomputing a calibrator on whatever data you have
    and deploying it on a new (target) dataset.
    """
    src_conf = np.concatenate([df["confidence"].values for df in src_dfs])
    src_y = np.concatenate([df["gpt_eval"].values for df in src_dfs])

    src_conf = np.asarray(src_conf, dtype=float)
    src_y = np.asarray(src_y, dtype=int)
    tgt_conf = np.asarray(tgt_conf, dtype=float)
    tgt_y = np.asarray(tgt_y, dtype=int)

    rng = np.random.default_rng(seed)
    n_train = int(fit_frac * len(src_conf))
    train_idx = rng.choice(len(src_conf), size=n_train, replace=False)
    probs = apply_scaling(method, src_conf[train_idx], tgt_conf, src_y[train_idx])
    probs = np.asarray(probs, dtype=float)
    metrics = calculate_metrics(tgt_y, probs, ece="binned", n_bins=n_bins)
    return metrics, probs


def run_transfer_pooled(
    runs_df: pd.DataFrame,
    runs_dir: Path,
    methods=SCALING_METHODS,
    fit_frac: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-one-out pooled transfer across datasets.

    For every run (target), fit each scaling method on the pooled runs of the SAME
    model and prompt but OTHER datasets, then apply to the full target run. Returns a
    long DataFrame with one row per (src_dataset, tgt_dataset, model, prompt, method)
    containing every metric in METRIC_COLUMNS.
    """
    records = []
    for _, tgt in runs_df.iterrows():
        tgt_df = load_run_df(runs_dir, tgt)
        pool = runs_df[
            (runs_df["dataset"] != tgt["dataset"])
            & (runs_df["model"] == tgt["model"])
            & (runs_df["prompt"] == tgt["prompt"])
        ]
        if pool.empty:
            continue
        src_dfs = [load_run_df(runs_dir, s) for _, s in pool.iterrows()]
        src_label = "+".join(sorted(pool["dataset"].unique()))
        for method in methods:
            metrics, _ = transfer_metrics_pooled(
                src_dfs,
                tgt_df["confidence"].values,
                tgt_df["gpt_eval"].values,
                method,
                fit_frac=fit_frac,
                seed=seed,
            )
            rec = {
                "src": src_label,
                "tgt": tgt["dataset"],
                "method": method,
                "dataset": tgt["dataset"],
                "model": tgt["model"],
                "prompt": tgt["prompt"],
            }
            rec.update({k: metrics[k] for k in METRIC_COLUMNS})
            records.append(rec)

    return pd.DataFrame(records)


def aggregate_transfer(transfer_df: pd.DataFrame, by: list[str], metric: str):
    """Aggregate a transfer grid to a pivot table of ``metric`` over ``by`` columns."""
    return (
        transfer_df.pivot_table(index="src", columns="tgt", values=metric, aggfunc="mean")
        .reindex(sorted(transfer_df["src"].unique()), axis=0)
        .reindex(sorted(transfer_df["tgt"].unique()), axis=1)
    )
