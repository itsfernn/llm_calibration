import marimo

__generated_with = "0.23.14"
app = marimo.App()

with app.setup:
    import glob
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    from src import metrics


@app.cell
def _(mo):
    mode = mo.ui.dropdown(["verb", "log"], value="verb")
    mode
    return (mode,)


@app.cell
def _(df, mode):
    ## plot metrics with seaborn
    plot_metrics(df, mode.value)
    return


@app.cell
def _(df, mode):
    results = create_bins(df)
    plot_calibration(results, method=mode.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell
def _():
    from src.utils import repo_root

    paths = glob.glob(
        str(
            repo_root()
            / "data"
            / "qwen"
            / "gsm8k"
            / "additional"
            / "**"
            / "metadata.json"
        ),
        recursive=True,
    )

    rows = []

    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)

            data["folder"] = path.rsplit("/", 1)[0]  # keep folder info
            rows.append(data)

        except Exception as e:
            print(f"Error reading {path}: {e}")

    df = pd.DataFrame(rows)

    def get_model_size(model_name):
        match = re.search(r"([\d.]+)\s*B", model_name)
        return float(match.group(1)) if match else None

    df = df[(df["thinking"] == True) | (df["type"] == "float")]

    df = df.set_index("type")
    df = df.sort_index()
    process_df(df)
    return (df,)


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
def process_df(df):
    for t, row in df.iterrows():
        file = row["folder"] + "/outputs.jsonl"
        output_file = pd.read_json(file, lines=True)
        preprocess_gsm8k(output_file)
        for method in ["verb", "log"]:
            mask = output_file[f"{method}_conf"].notna()
            y_true = output_file.loc[mask, "correct"]
            y_conf = output_file.loc[mask, f"{method}_conf"]
            ece_values = metrics.bootstrap_metrics(
                output_file["correct"],
                output_file[f"{method}_conf"],
                ece="discrete" if method == "verb" else "binned",
                n_bootstrap=10,
            )
            df.loc[t, f"{method}_ece"] = ece_values["ece"]
            df.loc[t, f"{method}_ece_se"] = ece_values["ece_se"]
            df.loc[t, f"{method}_brier"] = metrics.brier_score_loss(y_true, y_conf)
            df.loc[t, f"{method}_ap"] = metrics.average_precision_score(y_true, y_conf)
            df.loc[t, f"{method}_roc_auc"] = metrics.roc_auc_score(y_true, y_conf)
        df.loc[t, "acc"] = output_file["correct"].mean()
        df.loc[t, "num_samples"] = output_file.shape[0]


@app.cell
def _():
    sns.set_theme(style="whitegrid", context="talk")
    return


@app.function
def plot_metrics(df, type):
    cols = [
        c for c in df.columns if c.startswith(f"{type}_") and (not c.endswith("_se"))
    ] + ["acc"]
    plot_df = (
        df[cols]
        .reset_index(names="type")
        .melt(id_vars="type", var_name="metric", value_name="value")
    )
    plot_df["metric"] = plot_df["metric"].str.replace(f"{type}_", "", regex=False)
    title = "Verbalized Confidence" if type == "verb" else "Log Probability"
    ax = sns.barplot(data=plot_df, x="metric", y="value", hue="type")
    plt.title(title)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    return ax


@app.function
def create_bins(df):
    records = []
    for t, row in df.iterrows():
        file = row["folder"] + "/outputs.jsonl"
        output_file = pd.read_json(file, lines=True)
        preprocess_gsm8k(output_file)
        rec = {
            "model": row["model"],
            "type": t,
            "acc": output_file["correct"].mean(),
            "num_samples": len(output_file),
        }
        for method in ["verb", "log"]:
            conf_col = f"{method}_conf"
            mask = output_file[conf_col].notna()
            y_true = output_file.loc[mask, "correct"].to_numpy()
            y_conf = output_file.loc[mask, conf_col].to_numpy()
            if len(y_conf) == 0:
                rec[f"{method}_bin_conf"] = [np.nan] * 10
                rec[f"{method}_bin_acc"] = [np.nan] * 10
            else:
                cal_df = (
                    pd.DataFrame({"conf": y_conf, "correct": y_true})
                    .sort_values("conf")
                    .reset_index(drop=True)
                )
                cal_df["pool"] = pd.qcut(np.arange(len(cal_df)), q=10, labels=False)
                pool_stats = (
                    cal_df.groupby("pool")
                    .agg(avg_conf=("conf", "mean"), avg_acc=("correct", "mean"))
                    .reindex(range(10))
                )
                rec[f"{method}_bin_conf"] = pool_stats["avg_conf"].tolist()
                rec[f"{method}_bin_acc"] = pool_stats["avg_acc"].tolist()
        records.append(rec)
    summary = pd.DataFrame.from_records(records)
    return summary


@app.function
def plot_calibration(df, method="verb"):
    data = df.reset_index()
    data = data.explode([f"{method}_bin_conf", f"{method}_bin_acc"])
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = "flare"
    sns.lineplot(
        data=data,
        x=f"{method}_bin_conf",
        y=f"{method}_bin_acc",
        hue="type",
        markers="o",
        legend=True,
        palette=cmap,
        ax=ax,
    )
    ## set x and y labels
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    title = "Verbalized Calibration" if method == "verb" else "Log Probability"
    ax.set_title(title)
    plt.tight_layout()
    return ax


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
