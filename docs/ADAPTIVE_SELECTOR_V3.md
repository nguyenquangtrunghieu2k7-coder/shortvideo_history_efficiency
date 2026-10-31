# Adaptive History Selector V3

## Research objective

V3 formulates adaptive history selection as:

\[
\min_{\pi} \mathbb{E}[C(K_i)]
\quad \text{subject to} \quad
AUC(\pi) \ge AUC(Full)-\delta
\]

where \(\pi\) selects a history budget \(K_i\) for each request, \(C(K_i)\) is the number of history interactions processed, and \(\delta\) is the maximum acceptable global AUC drop.

The project non-inferiority margin is \\(\\delta=0.03\\). Use `--max-auc-drop 0.03` explicitly for all project V3 runs. This is a validation constraint, not a promise that test AUC will satisfy it.

## Why V3 replaces V2

V2 predicted whether a truncated model's *per-request log-loss* was close to Full and then thresholded the predicted probabilities. This introduced two mismatches:
- a single binary outcome makes per-request log-loss noisy as a supervision target;
- per-request log-loss acceptance is not the same objective as global ranking AUC.

V3 removes the acceptance classifier and its probability threshold. It learns a ranking-aligned proxy and evaluates the actual global AUC of every candidate policy during validation.

## Method

### 1. Request-time features

Use the stored history-length metadata and at most the most recent 100 interaction IDs. Features describe recent unique ratio, entropy, concentration, and repetition at several bounded suffix windows. The selector does not scan the entire history to calculate its request-time features. No label, future interaction, or model output is used as an inference feature. A production implementation should maintain history length as metadata and fetch only the bounded suffix before choosing whether to retrieve more history.

### 2. Ranking-aligned uplift target

For each request and each truncated K, compare its score against the opposite-class **Full-model** scores:
- for a positive request, how much of the negative Full scores does this score outrank?
- for a negative request, how much of the positive Full scores outranks this score?

The difference from that request's Full-reference contribution is its AUC-contribution uplift target. Ties count as one half.

This is a **supervised proxy**, not an exact additive decomposition when multiple positive and negative requests change actions simultaneously. The actual global AUC is always computed for the complete candidate policy during calibration and certification.

### 3. Predictor selection

Compare ExtraTrees, Random Forest, HistGradientBoosting, and a mean-only baseline using a nested split inside the development validation partition. Select by macro MAE over the seven uplift targets, with best-action agreement as a secondary diagnostic. Refit the selected model on the development partition only.

### 4. Policy search and baselines

Compare:
- all fixed K policies: 0, 1, 5, 10, 20, 50, 100, Full;
- transparent history-length heuristics;
- transparent history-diversity heuristics;
- learned per-request policies maximizing predicted AUC uplift minus a penalty on normalized interaction cost.

On the policy-tuning partition, calculate each candidate policy’s actual global AUC. Screen candidates by the point-estimate constraint AUC(policy) ≥ AUC(Full) − δ, then choose the lowest-cost candidate (learned policies must beat the best fixed-K policy; transparent heuristics may also win). Bootstrap intervals on this tuning partition are diagnostics, not the final gate. The selected candidate is evaluated once on the separate certification partition.

### 5. Independent certification

Validation is split into three disjoint roles:
- development: fit/select the uplift predictor;
- policy tuning: choose the policy and cost penalty;
- independent certification: evaluate the selected policy once.

The candidate passes only if the 90% paired-bootstrap lower confidence bound for AUC(policy)-AUC(Full) is at least \(-\delta\). If certification fails, V3 falls back to Full. The output records both the candidate policy and the final policy used for test, so a fallback cannot be mistaken for a successful adaptive policy. This prevents a policy selected from many candidates from being certified on the same observations that selected it.

**Protocol caveat:** Phase 8 used the validation split for checkpoint selection/early stopping. The three V3 partitions are independent of one another for selector fitting, policy tuning, and certification, but they are not a completely untouched dataset relative to the already-selected Phase 8 checkpoints. The held-out test set remains the final untouched evaluation set. Its bootstrap interval is post-hoc inference and never changes the selected policy.

## Run

After Phase 8 checkpoints and the frozen video vocabulary are available:

\`\`\`bash
python src/policy/train_adaptive_selector_v3.py --max-auc-drop 0.03
\`\`\`

Optional controls:

\`\`\`bash
python src/policy/train_adaptive_selector_v3.py \\
  --max-auc-drop 0.03 \\
  --lambda-steps 41 \\
  --bootstrap-repeats 200
\`\`\`

For a faster smoke run, use fewer trees, fewer lambda steps, and at least 20 bootstrap repeats; do not report smoke-run metrics as final results.

## Outputs

- \`models/adaptive_history_selector_v3.joblib\`
- \`data/processed/phase9_v3/adaptive_v3_metrics.csv\`
- \`data/processed/phase9_v3/adaptive_v3_validation_policy_frontier.csv\`
- \`data/processed/phase9_v3/adaptive_v3_test_decisions.parquet\`

## Interpretation limits

- AUC-contribution uplift is a proxy target. Policy selection uses the actual global AUC.
- Bootstrap confidence intervals are empirical uncertainty estimates, not guarantees.
- History interactions are a computational-work proxy. They are not measured latency, cloud spending, energy, or Bloom-FaaS cost.
- The current experiment runs frozen K-specific models to generate counterfactual scores offline. Bounded feature extraction reduces selector-side inspection to at most 100 recent IDs, but the current offline data loader still materializes histories; it does not yet demonstrate online serving or actual Bloom-FaaS integration.
- A V3 outcome that falls back to Full is a valid negative result: it means this experiment did not certify a cheaper policy under the chosen AUC constraint. Do not relax the constraint after inspecting test results.


## One-shot evaluation of the selected \\(\\delta=0.03\\) policy

The current selected tuning candidate is \\(\\lambda=0.006235507341\\). Its 5,000-repeat tuning bootstrap lower bound is approximately \\(-0.023824\\), which satisfies the project margin \\(-0.03\\). The frontier CSV may still show `ci_feasible_on_tuning=False` because it was generated in a run configured with `--max-auc-drop 0.01`; interpret the numeric CI bound against the intended \\(\\delta=0.03\\) threshold instead.

After the model bundle, test-decision parquet, and validation frontier from that run are present locally, evaluate this fixed policy once on the held-out test set without retraining:

\`\`\`bash
python src/policy/evaluate_adaptive_v3_test_policy.py \\
  --lambda-penalty 0.006235507341 \\
  --max-auc-drop 0.03 \\
  --bootstrap-repeats 5000 \\
  --bootstrap-seed 2026
\`\`\`

This reuses the saved model and all per-K test predictions. It does not reselect \\(\\lambda\\) from test results. Treat the output as the final test evaluation; do not adjust the policy after inspecting it. The earlier certification split was already inspected for a different candidate, so it must not be described as an untouched independent confirmation for this policy.
