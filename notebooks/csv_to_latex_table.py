#!/usr/bin/env python3
"""Convert a metrics CSV into a LaTeX longtable for the thesis.

The table is grouped by a "section" column (e.g. dataset, bold header) and a
"subgroup" column (e.g. model, italic header), with one row per prompt. The
best value of each metric within each subgroup is bolded.

Defaults reproduce `verb_metrics_full.csv` -> `verb_metrics_full.tex`.

Usage:
    python csv_to_latex_table.py verb_metrics_full.csv --output verb_metrics_full.tex
    python csv_to_latex_table.py verb_metrics_full.csv --columns ece brier nll --output subset.tex
    python csv_to_latex_table.py human_eval.csv --section dataset --subgroup model --label col --output out.tex
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    name: str                 # column name in the CSV
    label: str                # LaTeX header label
    direction: str            # "min" or "max"  -> which value is bolded
    ndigits: int = 3          # decimals for estimate and SE
    as_percent: bool = False  # multiply by 100 and append "\%"


def fmt_value(metric: Metric, value: float, percent_sign: bool = True) -> str:
    if metric.as_percent:
        suffix = "\\%" if percent_sign else ""
        return f"{value * 100:.{metric.ndigits}f}{suffix}"
    return f"{value:.{metric.ndigits}f}"


DEFAULT_METRICS = [
    Metric("acc", "Acc. $\\uparrow$", "max", ndigits=1, as_percent=True),
    Metric("ece", "ECE $\\downarrow$", "min"),
    Metric("brier", "Brier $\\downarrow$", "min"),
    Metric("nll", "NLL $\\downarrow$", "min"),
    Metric("auc_roc", "AUROC $\\uparrow$", "max"),
]

# CSV column used as the row label, plus how CSV values map to display labels
# and the order in which rows are shown.
DEFAULT_PROMPT_COL = "prompt"
DEFAULT_PROMPT_ORDER = ["Direct", "Top-k", "Chain of Thought", "Multistep"]
DEFAULT_PROMPT_LABELS = {
    "Direct": "Direct",
    "Top-k": "Top-$k$",
    "Chain of Thought": "CoT",
    "Multistep": "Multi-Step",
}

DEFAULT_SECTION_COL = "dataset"   # bold large header
DEFAULT_SUBGROUP_COL = "model"    # italic subheader
DEFAULT_SE_SUFFIX = "_se"         # metric SE column = metric name + suffix


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def group_column_value(value) -> str:
    text = str(value).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
    return text


def format_row(metric: Metric, row: pd.Series, se_suffix: str) -> str:
    est = fmt_value(metric, float(row[metric.name]))
    se = fmt_value(metric, float(row[metric.name + se_suffix]), percent_sign=False)
    return f"{est} ({se})"


def build_table(
    df: pd.DataFrame,
    *,
    metrics: list[Metric],
    section_col: str,
    subgroup_col: str,
    prompt_col: str,
    prompt_order: list[str],
    prompt_labels: dict[str, str],
    se_suffix: str,
    caption: str,
    label: str,
    note: str,
) -> str:
    n_cols = 1 + len(metrics)  # prompt label + one column per metric
    col_spec = "l" + "c" * len(metrics)

    header_row = " & ".join(["\\textbf{Prompt}"] + [f"\\textbf{{{m.label}}}" for m in metrics])

    lines: list[str] = []
    lines.append(f"\\begin{{longtable}}{{{col_spec}}}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:{label}}} \\\\")
    lines.append("\\toprule")
    lines.append(header_row + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append(header_row + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    section_order = list(dict.fromkeys(df[section_col].tolist()))
    subgroup_order = list(dict.fromkeys(df[subgroup_col].tolist()))
    prompt_order = [p for p in prompt_order if p in set(df[prompt_col])] + [
        p for p in dict.fromkeys(df[prompt_col].tolist()) if p not in prompt_order
    ]

    first_section = True
    for section in section_order:
        section_df = df[df[section_col] == section]

        if not first_section:
            lines.append(f"\\cmidrule{{1-{n_cols}}}")
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{\\large {group_column_value(section)}}}}} \\\\")
        lines.append(f"\\cmidrule{{1-{n_cols}}}")
        first_section = False

        for i, subgroup in enumerate(subgroup_order):
            subgroup_df = section_df[section_df[subgroup_col] == subgroup]
            if subgroup_df.empty:
                continue

            if i > 0:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_column_value(subgroup)}}}}} \\\\")

            # best (min or max) estimate per metric within this subgroup
            best = {
                m.name: subgroup_df[m.name].max() if m.direction == "max"
                else subgroup_df[m.name].min()
                for m in metrics
            }

            for prompt in prompt_order:
                row = subgroup_df[subgroup_df[prompt_col] == prompt]
                if row.empty:
                    continue
                row = row.iloc[0]
                cells = [prompt_labels.get(prompt, group_column_value(prompt))]
                for m in metrics:
                    val = format_row(m, row, se_suffix)
                    if abs(float(row[m.name]) - best[m.name]) < 1e-12:
                        val = f"\\textbf{{{val}}}"
                    cells.append(val)
                lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append(
        "\\multicolumn{" + str(n_cols) + "}{l}{\\footnotesize{\\textit{Note:} "
        + note + "}} \\\\"
    )
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_metrics(names: list[str] | None) -> list[Metric]:
    if names is None:
        return DEFAULT_METRICS
    by_name = {m.name: m for m in DEFAULT_METRICS}
    result = []
    for n in names:
        if n not in by_name:
            sys.exit(f"unknown metric column: {n!r} (known: {list(by_name)})")
        result.append(by_name[n])
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a metrics CSV into a LaTeX longtable."
    )
    parser.add_argument("csv", type=Path, help="input CSV file")
    parser.add_argument("--output", "-o", type=Path, help="output .tex file (default: stdout)")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="metric columns to include, e.g. 'acc ece brier nll auc_roc' (default: all)",
    )
    parser.add_argument("--section", default=DEFAULT_SECTION_COL,
                        help=f"column used for the bold section headers (default: {DEFAULT_SECTION_COL})")
    parser.add_argument("--subgroup", default=DEFAULT_SUBGROUP_COL,
                        help=f"column used for the italic subgroup headers (default: {DEFAULT_SUBGROUP_COL})")
    parser.add_argument("--label", default=DEFAULT_PROMPT_COL,
                        help=f"column used for row labels (default: {DEFAULT_PROMPT_COL})")
    parser.add_argument("--se-suffix", default=DEFAULT_SE_SUFFIX,
                        help=f"suffix that names SE columns (default: {DEFAULT_SE_SUFFIX!r})")
    parser.add_argument("--caption", default=None,
                        help="table caption (default: auto-generated)")
    parser.add_argument("--note", default=None,
                        help="footnote text under the table (default: auto-generated)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.csv)
    metrics = resolve_metrics(args.columns)

    section_label = args.section.replace("_", " ")
    subgroup_label = args.subgroup.replace("_", " ")
    caption = args.caption or (
        f"Full calibration and discrimination metrics for all "
        f"{section_label}--{subgroup_label}--{args.label} combinations."
    )
    note = args.note or (
        "Values are Estimate (SE). Bold indicates the best value for each "
        f"metric within each {subgroup_label}--{section_label} subgroup."
    )
    label = args.output.stem if args.output is not None else args.csv.stem

    tex = build_table(
        df,
        metrics=metrics,
        section_col=args.section,
        subgroup_col=args.subgroup,
        prompt_col=args.label,
        prompt_order=DEFAULT_PROMPT_ORDER,
        prompt_labels=DEFAULT_PROMPT_LABELS,
        se_suffix=args.se_suffix,
        caption=caption,
        label=label,
        note=note,
    )

    if args.output is None:
        print(tex, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
