# Progress Notes — Scaling Methods Evaluation

Goal: a single clean marimo notebook that evaluates all scaling methods (none, histogram,
isotonic, platt, temperature) on all runs (3 datasets x 4 models x 4 prompts), with
uncertainty estimates (CV + bootstrap), reliability diagrams, an effectiveness-by-prompt
analysis, a ranking-cost analysis, and a generalizability (transfer) analysis.

## Data

- `data/mhqa/data.json` -> 48 runs, ~1000 samples each (HotpotQA, 2WikiMultihopQA,
  MuSiQue; gpt-4o-mini, gpt-4o, Llama-3.3-70B, DeepSeek-V3; direct, top-k, multistep, cot).
- Labels `gpt_eval`, confidence `confidence` (verbalized). src/metrics.py, src/plot.py,
  src/scaling.py untouched.

## Method

- **3-fold stratified CV** (seed 42), fit scaling on training folds, evaluate on the
  held-out third (union of test folds covers the full run). Chosen over 5-fold: larger
  test folds -> more stable fold-level variance estimate, ~40% less bootstrap compute,
  still plenty of data to fit each method.
- **Combined CV + bootstrap SEs**: bootstrap-resample each held-out fold (model frozen),
  pool fold distributions; SE = sqrt(mean fold bootstrap variance / 3). 100 resamples per
  fold -> 300 per (run, method).
- **Regular (binned) ECE everywhere** — d_ece (discrete ECE, only for step-valued
  verbalized confidence) is NOT used; multistep's confidence is continuous and treated
  like every other run.
- Transfer: fit on an 80% train split of the source run(s), apply to the FULL target run.
  Three directions: cross-model, cross-dataset (single source), and **pooled source**
  (fit on all other datasets' runs with the same model+prompt, deploy on the target).

## src/eval.py (new)

- `cv_folds`, `fold_metrics`, `_bootstrap_fold`, `cv_evaluate`, `cv_evaluate_runs`
  (+ `folds_by_key` for reliability diagrams), `summarize_cv`, `transfer_metrics`,
  `run_transfer_grid` (now also records the shared `dataset` for prompt/model axes),
  `transfer_metrics_pooled`, `run_transfer_pooled`, `aggregate_transfer`.
- `pure_bootstrap_se` removed (its only use, the CV-vs-bootstrap comparison, was scrapped).

## Key findings (3-fold, verified)

- Raw verbalized confidence is overconfident: ECE 0.09–0.60 (mean 0.31). Multistep is the
  best calibrated prompt raw (0.16 vs 0.33–0.38 for direct/top-k/cot).
- In-domain: histogram ≈ 0.03–0.05, isotonic ≈ 0.03–0.05, platt ≈ 0.05–0.08, temperature
  ≈ 0.13–0.18. **Consistency is prompt-dependent**: multistep -> histogram on all 3
  datasets (7/12 runs); direct/top-k/cot are inconsistent (isotonic wins most often but
  flips between histogram/isotonic/platt across runs).
- Ranking cost: platt/temperature strictly monotone -> AUROC/AP preserved to ~2e-4;
  isotonic ~preserved (plateaus, max AUROC Δ ~0.03); histogram genuinely changes ranking
  (mean AUROC drop 0.006, max 0.03; mean AP drop 0.016, max 0.07).
- **Cross-model** is the *easier* axis: every transferred method halves raw ECE
  (0.31 -> 0.11–0.17); histogram/isotonic degrade ~2.8x vs in-domain, platt ~1.7x,
  **temperature ~1.05x**.
- **Cross-dataset** is the harder axis: histogram/isotonic degrade ~4–5x vs in-domain and
  can be worse than doing nothing (e.g. histogram MuSiQue->HotpotQA 0.254 vs raw 0.232).
- **Temperature's transfer behaviour is base-rate driven — this changed the
  conclusion.** Its transferred ECE correlates +0.83 with |target accuracy − 0.5|
  (dataset transfer); it degenerates toward predicting a fixed ~0.5, so it "works"
  only when the target's accuracy is near 0.5. Per run the transfer effect varies
  (25% improve, 75% degrade, −0.08 to +0.12); the low mean loss is a cancellation,
  not consistency. In-domain its ECE also correlates +0.78 with |acc − 0.5|.
- In absolute ECE, temperature is NOT the best transfer method: tied with
  histogram/isotonic for dataset transfer (~0.18), and the worst for model transfer
  (0.165 vs 0.107–0.114). The "~1.1x" robustness was an artifact of its high
  in-domain baseline.
- **Consequence**: no scaling method reliably fixes OOB calibration by transfer;
  expect ~0.15–0.20 ECE regardless of method. Calibration must be fitted on the
  target domain.

## Marimo gotchas (important)

- A trailing `return X` in a marimo `.py` cell function is STRIPPED by the parser — cell
  output comes from the last *expression* statement. Figure cells must end with a bare
  `fig`/`_fig` (an Artist), NOT `return fig`. `fig.tight_layout()` returns None, so it
  must not be the last statement. (Debugged via marimo source: `_ast/parse.py`
  `has_return` handling.)
- `mo.persistent_cache` caches the cell's *defs* (assigned variables), so data cells work
  with assignments + a stripped return.

## Notebook structure (Scaling Methods Evaluation.py)

1. Setup, mappings, theme, data, selectors (dataset/model/metric)
2. Cached: cv_eval (3 folds), transfer grids, pooled transfer
3. Header + methodology + ECE note
4. Reliability diagrams (unchanged interpolated plots, now 3 folds; legend once)
5. Metrics overview: prompt-based scatter (all runs, x=prompt, size=SE, color=method,
   color-only legend), dataset/model heatmaps, dropdown-driven metric-vs-accuracy
   scatter, per-run summary table
6. Effectiveness by prompt & consistency: 2x4 heatmaps (by dataset / by model),
   best-method tables, findings + temperature "fake logits" note
7. Ranking cost: raw-vs-scaled AUROC/AP identity scatter + numbers
8. Generalizability: cross-model heatmap, cross-dataset overview heatmaps,
   **condensed dumbbell** (per method: mean in-domain vs transferred ECE, per-run
   transfer dots as faint strip; cross-dataset + cross-model side by side),
   cross-dataset by prompt (summary + transfer-loss heatmaps), best-transfer table
   per (model,prompt), pooled precomputed calibrator
9. Details

## Plotting notes

- Method colors use a distinct palette (grey/magenta/purple/cyan/lime), separate from
  the muted prompt palette (dusty blue/terracotta/sage/gold).
- Marimo quirk: figure cells must end with a bare `fig`/`_fig` expression (a trailing
  `return` is stripped by the parser); `_fig.tight_layout()` returns None and must not
  be the last statement.
- The dropdown-driven metric-vs-accuracy scatter was accidentally dropped in an edit
  and later restored — always verify cell count/figure count after edits.

## Validation

- `marimo check` clean; `marimo export session` runs all cells -> 9 figures + 2 tables,
  no console warnings/errors. Session snapshot in notebooks/__marimo__/session/.
- Caches (v3folds) in notebooks/__marimo__/cache/; stale 5-fold caches removed.
