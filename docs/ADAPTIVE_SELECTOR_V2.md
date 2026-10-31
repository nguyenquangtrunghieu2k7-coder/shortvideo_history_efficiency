# Adaptive History Selector V2

## Purpose

V2 predicts, for each request and each truncated history length, the
probability that using that K keeps per-request binary log-loss within a
configured tolerance of the Full-history model:

```text
History structure features
          |
          v
Multi-output quality-acceptance classifier
  -> P(loss(K) <= loss(Full) + max_logloss_regret)
          |
          v
Shortest-acceptable-history policy
  -> choose the smallest K whose acceptance probability clears a threshold
  -> use Full if no truncated K clears the threshold
          |
          v
Validation calibration
  -> maximize AUC - budget_weight * normalized_history_cost
```

V1 remains unchanged and is the direct-classification baseline.

## What is new

- Request-time features describe history composition, concentration, recency,
  head/tail overlap, novelty, and adjacent repetition.
- For each request and truncated K, the binary target is
  `loss(K) <= loss(Full) + max_logloss_regret`. The default regret tolerance
  is 0.05, matching the V1 oracle's initial tolerance and enabling a direct
  conceptual comparison.
- ExtraTrees, RandomForest, HistGradientBoosting, and a prior-only
  DummyClassifier are compared on an inner split of predictor-fit validation
  data using macro average precision. The selected classifier is refit on the
  full predictor-fit partition. If the mean baseline wins, the run explicitly
  warns that request-specific feature signal has not been demonstrated.
- Validation is split into predictor-fit and policy-calibration subsets.
  On the calibration subset, the probability threshold is selected by
  maximizing `validation_AUC - budget_weight * normalized_mean_history_cost`.
  The policy chooses the shortest K whose predicted acceptance probability
  meets that threshold; Full is the safe fallback. K=0 remains eligible.
- The calibration objective directly includes global AUC, the final ranking
  metric, instead of assuming that lower log-loss necessarily means higher AUC.
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
  --max-logloss-regret 0.05 \
  --calibration-fraction 0.30 \
  --trees 500 \
  --threshold-steps 101
```

The default budget weight controls the trade-off between validation ROC-AUC
and normalized history-interaction budget. The regret tolerance defines the
binary acceptance target; it is not a per-request AUC target.

## Outputs

- `models/adaptive_history_selector_v2.joblib`
- `data/processed/phase9_v2/adaptive_v2_metrics.csv`
- `data/processed/phase9_v2/adaptive_v2_test_decisions.parquet`
- `data/processed/phase9_v2/adaptive_v2_threshold_calibration.csv`
- `data/processed/phase9_v2/adaptive_v2_feature_importance.csv`
- `data/processed/phase9_v2/adaptive_v2_quality_diagnostics.csv`
- `data/processed/phase9_v2/adaptive_v2_model_selection.csv`

The quality diagnostics CSV reports per-K acceptance prevalence, predicted
acceptance rate, average precision, ROC-AUC, and Brier score on validation
calibration and held-out test. The model-selection CSV records the inner
validation comparison between the tree classifiers and the prior-only
baseline. Feature importance is measured by permutation using macro average
precision on the policy-calibration subset.

The test decisions Parquet additionally stores:
- predicted acceptance probabilities for every truncated K;
- frozen-model probabilities for every K;
- actual per-request binary log-loss for every K;
- the selected policy's actual log-loss;
- a post-hoc oracle-best K and its actual loss;
- per-request actual-loss regret relative to that oracle.

The oracle fields are strictly post-hoc diagnostics. They use test labels after
the policy has already been selected and must never be used as selector inputs,
training targets, or tuning signals.

## Interpretation limits

- Per-request acceptable-vs-Full classification is a supervised proxy; it is
  not a per-request AUC prediction. Global AUC is optimized only during
  validation policy calibration.
- Final recommendation quality is reported using global test ROC-AUC.
- Interaction reduction is a history-budget measurement. It is not evidence
  of measured serving latency, cloud spending, energy savings, or serverless
  speedup.
- The current experiment still evaluates frozen K-specific checkpoints. It is
  not yet an end-to-end online serving implementation or a reproduction of
  BLOOM-FaaS.
