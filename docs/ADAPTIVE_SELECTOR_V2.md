# Adaptive History Selector V2

## Purpose

V2 replaces V1's direct `request features -> K class` classifier with a
quality-prediction layer and a separately calibrated cost policy:

```text
History structure features
          |
          v
Multi-output quality predictor
  -> estimated per-request log-loss for each K
          |
          v
Cost-aware policy
  -> argmin_k predicted_loss(k) + lambda * normalized_history_cost(k)
          |
          v
Selected K and frozen recommender score
```

V1 remains unchanged and is the direct-classification baseline.

## What is new

- Request-time features describe history composition, concentration, recency,
  head/tail overlap, novelty, and adjacent repetition.
- The quality target is now the per-request log-loss difference
  `loss(K) - loss(Full)`. Full is the exact zero reference, so the model
  focuses on the relative quality change caused by truncating history rather
  than predicting each model's noisy absolute loss independently.
- Targets are winsorized at the 1st/99th percentiles using predictor-fit data
  only. ExtraTrees and RandomForest are compared on an inner split of the
  predictor-fit partition; the lower-MAE model is refit on that full partition.
- Validation is split into predictor-fit and policy-calibration subsets.
  The scalar `lambda` is selected on the second subset by minimizing
  `calibration_logloss + budget_weight * normalized_mean_history_cost`.
  AUC is reported as an evaluation metric, not used as a hard feasibility gate.
  This removes the previous no-feasible-lambda fallback.
- Test is held out for the final evaluation only.

The cost penalty uses each candidate K's interaction count divided by the
mean Full-history length in the calibration split. This makes aggregate cost
proportional to total history interactions; it does not give every request
equal weight regardless of its history length.

## Run

From the repository root, after Phase 8 frozen checkpoints and the frozen video
vocabulary are available:

```bash
python src/policy/train_adaptive_selector_v2.py
```

Useful options:

```bash
python src/policy/train_adaptive_selector_v2.py \
  --budget-weight 0.005 \
  --calibration-fraction 0.30 \
  --trees 500 \
  --lambda-max 0.50 \
  --lambda-steps 101
```

The default budget weight controls the trade-off between validation logloss
and normalized history-interaction budget. It is an explicit utility parameter,
not a claim that per-request log-loss equals AUC.

## Outputs

- `models/adaptive_history_selector_v2.joblib`
- `data/processed/phase9_v2/adaptive_v2_metrics.csv`
- `data/processed/phase9_v2/adaptive_v2_test_decisions.parquet`
- `data/processed/phase9_v2/adaptive_v2_lambda_calibration.csv`
- `data/processed/phase9_v2/adaptive_v2_feature_importance.csv`
- `data/processed/phase9_v2/adaptive_v2_quality_diagnostics.csv`
- `data/processed/phase9_v2/adaptive_v2_model_selection.csv`

The test decisions Parquet additionally stores:
- frozen-model probabilities for every K;
- actual per-request binary log-loss for every K;
- the selected policy's actual log-loss;
- a post-hoc oracle-best K and its actual loss;
- per-request actual-loss regret relative to that oracle.

The oracle fields are strictly post-hoc diagnostics. They use test labels after the policy has already been selected and must never be used as selector inputs, training targets, or tuning signals. The quality diagnostics CSV reports MAE, RMSE, and R-squared between predicted and actual per-request log-loss, separately for validation calibration and held-out test.

## Interpretation limits

- Per-request binary log-loss is a supervised proxy for the quality predictor;
  it is not per-request AUC.
- Final recommendation quality is reported using global test ROC-AUC.
- Interaction reduction is a history-budget measurement. It is not evidence
  of measured serving latency, cloud spending, energy savings, or serverless
  speedup.
- The current experiment still evaluates frozen K-specific checkpoints. It is
  not yet an end-to-end online serving implementation or a reproduction of
  BLOOM-FaaS.
