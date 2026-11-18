# Experiment: Rolling temporal evaluation windows

Branch: `experiment/sliding-window-history-vs-disjoint-1day`

## Important clarification

The existing history builder `src/data/build_history_configs.py` **already** constructs a sliding 24-hour history `[t-24h,t)` for each evaluation event. The existing `src/data/build_split.py` instead creates a **single disjoint chronological 70/15/15 train/validation/test split**. This experiment changes the **evaluation split protocol**, not the per-event 1-day history definition.

## Proposed comparison

- **Baseline:** fixed chronological 70/15/15 train/validation/test split; frozen V4 and checkpoints remain untouched.
- **Experimental:** rolling temporal folds, default 7-day train / 1-day validation / 1-day test, advanced by 1 day each fold; optional embargo between sets.
- Folds can overlap **across folds**, but train, validation and test **never overlap within the same fold**.
- Rolling folds are not statistically independent; aggregate metrics must account for overlapping requests.
- Do not use test labels to tune hyperparameters or retrain a frozen policy.

## Generate fold artifacts

```bash
git switch experiment/sliding-window-history-vs-disjoint-1day
PYTHONPATH=. python -m unittest tests.test_rolling_evaluation_splits -v
PYTHONPATH=. python -m src.data.build_rolling_evaluation_splits \
  --source data/processed/phase7/history_KFull.parquet \
  --train-days 7 --validation-days 1 --test-days 1 --step-days 1 \
  --output data/processed/experiments/sliding_window_1day
```

Output: `fold_000/{train,validation,test}.parquet` etc., per-fold manifests and `fold_manifest.json`. The generator does not modify baseline artifacts under `data/processed/splits`.

**Status:** branch, generator and unit tests committed; local dataset run and recommender retraining/evaluation have **not** yet been executed. Rolling-fold AUC and history-saving numbers are not yet available. Frozen V4 trained on baseline data must not be presented as a newly trained rolling-window model. A fair policy/model comparison requires a separate per-fold training/evaluation protocol with strict leakage controls.
