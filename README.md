# Calibration of Verbalized Confidence — Experiments & Evaluation

Active repository for the master thesis on calibration of verbalized confidence in
LLMs. Covers data extraction (MHQA + local Qwen runs), post-hoc confidence scaling
methods (histogram, isotonic, Platt, temperature), and their evaluation with
cross-validation, bootstrapped uncertainty, and transfer analysis.

The original experiment code is archived separately in `code-legacy/`
(repo `code/`), kept for paper reproduction.

## Layout

```
src/            Reusable library used by the notebooks
  metrics.py    Single source of truth for metrics: calculate_metrics (ece=
                "binned" | "discrete" | "both"), bootstrap_metrics (+ CIs)
  scaling.py    Post-hoc scaling methods (histogram, isotonic, Platt, temperature, LR)
  plot.py       Reliability diagrams and calibration plots
  eval.py       CV evaluation of scaling methods + transfer analysis (src/eval.py)
  utils.py      Bin statistics, repo_root() path helper

data/
  mhqa/         MHQA experiment outputs: 48 runs (CSV per run + data.json metadata),
                copied from the legacy repo (3 datasets x 4 models x 4 prompts).
                Labels: gpt_eval (LLM-as-judge), confidence (verbalized).
  qwen/         Raw outputs of local Qwen model runs (metadata.json + outputs.jsonl)
                for GSM8K, AI2 ARC, MMLU (incl. old/ with earlier runs).
  raw/          Input datasets (2WikiMultihopQA, copied locally). Gitignored.

experiments/
  mhqa/         Data-extraction pipeline for the MHQA experiments, ported from the
                legacy repo: run.py + utils/ (chat, datasets, processor, gpt_eval,
                metrics, plotting). Run from this directory: python run.py --help.
  qwen/         Data-extraction scripts for local Qwen models (gsm8k.py, ai2_arc.py,
                mmlu.py, run.py). Outputs go to experiments/qwen/out_runs/.

notebooks/      Active analysis notebooks (marimo)
  Scaling Methods Evaluation.py   CV + bootstrap eval of all scaling methods, transfer
  Verbalized Conf Analysis.py     Verbalized-confidence analysis of the MHQA runs
  compareThinking.py              Thinking vs. non-thinking prompts (GSM8K)
  qwen.py                         Model-size analysis + run tagging UI (Qwen runs)
  legacy/                         Archived Jupyter notebooks (not versioned)

results/
  tables/       Generated metric CSVs (verb_metrics_full, *_metrics_summary,
                human_eval*, ...) used by scripts/csv_to_latex_table.py for thesis tables
  figures/      Generated plots

scripts/
  csv_to_latex_table.py   Convert a metrics CSV into a LaTeX longtable (thesis tables)
```

## Setup

Requires Python 3.14 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # installs everything (incl. experiment deps: langchain, wandb, torch, transformers)
cp .env.example .env   # (optional) API keys for data extraction
```

All dependencies (evaluation library, MHQA extraction, Qwen extraction, dev tools)
are in the default dependency set, so a plain `uv sync` gives a fully working env.

## Running the notebooks

```bash
uv run marimo edit notebooks/Scaling\ Methods\ Evaluation.py
```

All notebooks resolve data paths relative to the repo root, so they work regardless
of where marimo is launched from.

## Regenerating results

- MHQA run data → `experiments/mhqa/run.py` (needs API keys)
- Qwen run data → `experiments/qwen/run.py` (needs local model checkpoints)
- Metric tables → `scripts/csv_to_latex_table.py results/tables/verb_metrics_full.csv ...`
- Figures → the marimo notebooks save plots into `results/figures/`

## Data provenance & known quirks

- `data/mhqa/` CSVs are identical to `code-legacy/csv/main/` (verified byte-for-byte).
- Five GSM8K runs in `data/qwen/gsm8k/additional/` (20260725_*) are missing a `type`
  field in their metadata — they show up as NaN groups in `compareThinking.py`.
- The legacy Jupyter notebooks in `notebooks/legacy/` are kept for reference only and
  are not versioned; they point at `data/mhqa` and can be run from their own directory.
