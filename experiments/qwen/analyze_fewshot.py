"""Analysis of the GSM8K few-shot post-training comparison (base vs. aligned).

Compares the verbalized-confidence behavior of a post-trained checkpoint
(e.g. Qwen/Qwen3-8B) against its non-RLHF base counterpart (Qwen3-8B-Base)
under the identical plain-text few-shot protocol of gsm8k_fewshot.py.

The question: does preference-based post-training cause the tie-at-1.0
overconfidence signature seen in the main grid (Section 5.1 of the thesis)?

Usage:
    python experiments/qwen/analyze_fewshot.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import bootstrap_metrics  # noqa: E402
from src.utils import compute_equal_frequency_bin_stats, repo_root  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reliability_figure(run: dict, path: Path) -> None:
    """Reliability diagram (verbalized + log-prob panels) for one run."""
    df = run["df"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (conf_col, title) in zip(
        axes, [("verb_conf", "Verbalized Confidence"), ("log_conf", "Log Probabilities")]
    ):
        valid = df["prediction"].notna() & df[conf_col].notna()
        sub = df[valid]
        if len(sub) >= 10:
            bins = compute_equal_frequency_bin_stats(
                y_true=sub["correct"], y_prob=sub[conf_col], n_bins=10
            )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
            ax.plot(bins["mean_conf"], bins["mean_y"], "o-", linewidth=2, markersize=6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle=":", alpha=0.5)
    model = run["meta"]["model"]
    fig.suptitle(f"{model}", weight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Preprocessing (mirrors the notebook's preprocess_gsm8k)
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_conf"] = df["logprobs"].apply(
        lambda xs: np.exp(sum(xs) / len(xs)) if xs else 0.5
    )

    def extract_answer(text):
        match = re.search(r"####\s*(\-?\d+\.?\d*)", text)
        return float(match.group(1)) if match else None

    df["answer"] = df["answer"].apply(extract_answer)

    def parse_prediction(pred):
        match = re.findall(r"-?\d+\.?\d*", str(pred))
        return float(match[-1]) if match else None

    df["prediction"] = df["prediction"].apply(parse_prediction)
    df["correct"] = df["answer"] == df["prediction"]
    return df


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(
        glob.glob(
            str(repo_root() / "data" / "qwen" / "gsm8k_fewshot" / "**" / "metadata.json"),
            recursive=True,
        )
    ):
        with open(path) as f:
            meta = json.load(f)
        run_dir = Path(path).parent
        outputs = pd.read_json(run_dir / "outputs.jsonl", lines=True)
        runs.append({"meta": meta, "df": preprocess(outputs), "run_dir": run_dir})
    return runs


def label(run: dict) -> str:
    model = run["meta"]["model"]
    return "base" if model.endswith("-Base") else "aligned"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def run_metrics(run: dict, n_bootstrap=1000) -> dict:
    df = run["df"]
    out = {"run": run["meta"]["run"] if "run" in run["meta"] else run["meta"]["timestamp"],
           "model": run["meta"]["model"], "n": len(df)}

    out["n_parsed_verb"] = int(df["verb_conf"].notna().sum())
    out["n_parsed_pred"] = int(df["prediction"].notna().sum())
    out["frac_verb_conf"] = df["verb_conf"].notna().mean()

    # distribution of verbalized confidence
    vc = df["verb_conf"].dropna()
    if len(vc):
        out["mean_verb_conf"] = vc.mean()
        out["frac_conf_ge_0999"] = (vc >= 0.999).mean()
        out["n_unique_verb_conf"] = vc.round(2).nunique()

    # accuracy and correctness gap (on rows with a parsed prediction)
    pred_ok = df[df["prediction"].notna()]
    if len(pred_ok):
        acc = pred_ok["correct"].mean()
        out["acc"] = acc
        out["n_correct"] = int(pred_ok["correct"].sum())
        if len(vc):
            out["conf_minus_acc"] = vc.mean() - acc

    # metric suite for each channel (bootstrap SEs); correctness is only
    # meaningful on rows with a parsed prediction
    valid_pred = df["prediction"].notna()
    for channel, conf_col, ece_type in [
        ("verb", "verb_conf", "discrete"),
        ("log", "log_conf", "binned"),
    ]:
        mask = valid_pred & df[conf_col].notna()
        y = np.asarray(df.loc[mask, "correct"], dtype=int)
        c = np.asarray(df.loc[mask, conf_col], dtype=float)
        if len(y) < 50:
            out[f"{channel}_n"] = len(y)
            continue
        m = bootstrap_metrics(y, c, ece=ece_type, n_bootstrap=n_bootstrap)
        for k in ["ece", "brier", "auc_roc", "ap_errors", "ap_errors_norm"]:
            out[f"{channel}_{k}"] = m[k]
            out[f"{channel}_{k}_se"] = m[f"{k}_se"]
        out[f"{channel}_n"] = len(y)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    runs = load_runs()
    if not runs:
        print("no runs found in data/qwen/gsm8k_fewshot/")
        return

    rows = [run_metrics(r) for r in runs]
    table = pd.DataFrame(rows)

    # readable printout
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table.to_string(index=False))

    out_csv = repo_root() / "results" / "tables" / "qwen_fewshot_comp.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    # reliability figures
    fig_dir = repo_root() / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        fig_path = fig_dir / f"qwen_fewshot_{label(run)}_reliability.png"
        reliability_figure(run, fig_path)
        print(f"saved {fig_path}")
    print("\n=== notes ===")
    for _, r in table.iterrows():
        if r["frac_verb_conf"] < 0.5:
            print(f"{r['model']}: LOW FORMAT COMPLIANCE "
                  f"({r['frac_verb_conf']:.1%} of rows carry verbalized confidence) "
                  f"- not comparable")
        else:
            print(
                f"{r['model']}: acc={r.get('acc', float('nan')):.1%}, "
                f"mean verb conf={r.get('mean_verb_conf', float('nan')):.2f}, "
                f"frac conf>=0.999={r.get('frac_conf_ge_0999', float('nan')):.2f}, "
                f"n unique conf={r.get('n_unique_verb_conf', 0)}, "
                f"conf-acc gap={r.get('conf_minus_acc', float('nan')):.2f}, "
                f"verb AUROC={r.get('verb_auc_roc', float('nan')):.3f}, "
                f"log AUROC={r.get('log_auc_roc', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()
