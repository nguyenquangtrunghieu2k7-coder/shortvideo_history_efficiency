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
- An `ExtraTreesRegressor` predicts the per-request binary log-loss for each
  frozen K-specific recommender.
- Validation is split into predictor-fit and policy-calibration subsets.
  The predictor is fit only on the first subset. The scalar `lambda` is chosen
  on the second subset by finding the smallest interaction budget whose AUC is
  within the configured tolerance of the Full model.
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
  --max-auc-drop 0.01 \
  --calibration-fraction 0.30 \
  --trees 500 \
  --lambda-max 0.50 \
  --lambda-steps 101
```

The default AUC tolerance is 0.01 absolute ROC-AUC. It is a policy constraint,
not a claim that per-request log-loss equals AUC.

## Outputs

- `models/adaptive_history_selector_v2.joblib`
- `data/processed/phase9_v2/adaptive_v2_metrics.csv`
- `data/processed/phase9_v2/adaptive_v2_test_decisions.parquet`
- `data/processed/phase9_v2/adaptive_v2_lambda_calibration.csv`
- `data/processed/phase9_v2/adaptive_v2_feature_importance.csv`

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
