# Adaptive History Selector V3

## Research objective

V3 formulates adaptive history selection as:

\[
\min_{\pi} \mathbb{E}[C(K_i)]
\quad \text{subject to} \quad
AUC(\pi) \ge AUC(Full)-\delta
\]

where \(\pi\) selects a history budget \(K_i\) for each request, \(C(K_i)\) is the number of history interactions processed, and \(\delta\) is the maximum acceptable global AUC drop.

The default non-inferiority margin is \(\delta=0.01\). This is a validation constraint, not a promise that test AUC will satisfy it.

## Why V3 replaces V2

V2 predicted whether a truncated model's *per-request log-loss* was close to Full and then thresholded the predicted probabilities. This introduced two mismatches:
- a single binary outcome makes per-request log-loss noisy as a supervision target;
- per-request log-loss acceptance is not the same objective as global ranking AUC.

V3 removes the acceptance classifier and its probability threshold. It learns a ranking-aligned proxy and evaluates the actual global AUC of every candidate policy during validation.

## Method

### 1. Request-time features

Use only the available, temporally prior interaction history. Features describe history length, unique ratio, entropy, concentration, recent novelty, head-tail overlap, and repetition. No label, future interaction, or model output is used as an inference feature.

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

For each candidate policy, calculate global validation AUC and a paired, class-stratified bootstrap confidence interval for AUC(policy)-AUC(Full). Among feasible policies, choose the lowest interaction cost. Keep a learned policy only if it beats the best feasible fixed-K policy; otherwise use the fixed-K fallback. Heuristics are also allowed to win if they reduce cost further.

### 5. Independent certification

Validation is split into three disjoint roles:
- development: fit/select the uplift predictor;
- policy tuning: choose the policy and cost penalty;
- independent certification: evaluate the selected policy once.

The policy passes only if the 90% paired-bootstrap lower confidence bound for AUC(policy)-AUC(Full) is at least \(-\delta\). If certification fails, V3 falls back to Full. This prevents a policy selected from many candidates from being certified on the same observations that selected it.

The held-out test set is used only after policy selection and independent validation certification. Its bootstrap interval is post-hoc inference and never changes the selected policy.

## Run

After Phase 8 checkpoints and the frozen video vocabulary are available:

\`\`\`bash
python src/policy/train_adaptive_selector_v3.py
\`\`\`

Optional controls:

\`\`\`bash
python src/policy/train_adaptive_selector_v3.py \\
  --max-auc-drop 0.01 \\
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
- The current experiment runs frozen K-specific models to generate counterfactual scores offline. It does not yet demonstrate online serving or actual Bloom-FaaS integration.
- A V3 outcome that falls back to Full is a valid negative result: it means this experiment did not certify a cheaper policy under the chosen AUC constraint. Do not relax the constraint after inspecting test results.
