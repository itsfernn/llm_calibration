#!/usr/bin/env python3
"""Prepare the thesis table CSVs for the Qwen3 white-box results.

Reads the raw bootstrapped metrics CSV written by the qwen.py notebook
(results/tables/qwen_main_metrics.csv, one row per run x confidence stream,
with both the binned ECE ("ece") and the discrete ECE ("d_ece")) and produces
the two table CSVs consumed by csv_to_latex_table.py:

- qwen8b_metrics.csv    : Qwen3-8B head-to-head (verbalized vs. log probs per dataset)
- qwen_main_metrics.csv : full grid over model sizes (verb/log as subgroups)

In both tables ECE is reported per stream with the definition appropriate to
its support (discrete for verbalized confidence, binned for log probabilities),
mirroring the convention used throughout the thesis.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "tables" / "qwen_main_metrics.csv"


def per_stream_ece(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    verb = df["conf_type"] == "Verbalized Confidence"
    df.loc[verb, "ece"] = df.loc[verb, "d_ece"]
    df.loc[verb, "ece_se"] = df.loc[verb, "d_ece_se"]
    return df


def main() -> None:
    raw = pd.read_csv(RAW)
    raw["model_size"] = raw["model_size"].astype(float)
    raw = per_stream_ece(raw)
    raw["model"] = raw["model"].str.replace("Qwen/", "", regex=False)

    # --- Qwen3-8B head-to-head table (rows = confidence stream) ---
    keep8 = [
        "dataset", "model", "conf_type", "acc", "acc_se", "ece", "ece_se",
        "brier", "brier_se", "auc_roc", "auc_roc_se",
        "ap_errors_norm", "ap_errors_norm_se",
    ]
    eight = raw[raw["model_size"] == 8.0].copy()
    eight["model"] = "Qwen3-8B"
    eight = eight.sort_values(["dataset", "conf_type"])
    eight[keep8].to_csv(ROOT / "results" / "tables" / "qwen8b_metrics.csv", index=False)

    # --- Full grid appendix table (rows = model size) ---
    keepA = [
        "dataset", "conf_type", "model", "acc", "acc_se", "ece", "ece_se",
        "brier", "brier_se", "auc_roc", "auc_roc_se",
    ]
    main = raw.sort_values(["dataset", "model_size", "conf_type"])
    main[keepA].to_csv(ROOT / "results" / "tables" / "qwen_main_metrics.csv", index=False)

    print(f"wrote qwen8b_metrics.csv ({len(eight)} rows) and qwen_main_metrics.csv ({len(main[keepA])} rows)")


if __name__ == "__main__":
    main()
