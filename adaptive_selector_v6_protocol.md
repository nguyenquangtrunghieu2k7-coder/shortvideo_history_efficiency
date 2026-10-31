# Adaptive History Selector V6 — Validation LCB90-constrained Pareto Knee

## 1. Objective

V6 retains V5's Pareto knee selection, but strengthens candidate feasibility. A candidate must satisfy both a point-estimate AUC constraint and a one-sided lower-confidence-bound constraint on the validation policy-tuning partition.

## 2. Data and leakage controls

- Keep the existing user-disjoint development / policy-tuning split and inner group-disjoint uplift-model fit / selection split.
- Fit the uplift predictor using development users only.
- Evaluate all candidate policies and their bootstrap intervals on the policy-tuning users only.
- Do not use held-out test labels, scores, AUC, cost, or bootstrap results in selection.
- Freeze the selected policy before one-time held-out test evaluation.

## 3. Candidate family

Use the V5 candidate family unchanged: fixed K in {0, 1, 5, 10, 20, 50, 100, Full} and learned lambda policies on the predeclared grid (default 41 values including zero).

## 4. Feasibility constraints

Let Full be the reference and let Δ = AUC(policy) − AUC(Full).

A candidate is feasible only if both conditions hold on the validation policy-tuning partition:

1. Point estimate: Δ ≥ −δ.
2. Bootstrap lower confidence bound: LCB90(Δ) ≥ −δ.

Here δ = 0.03 by default. LCB90 is the 5th percentile of a class-stratified paired bootstrap distribution of Δ. Each bootstrap replicate resamples positive and negative requests separately with replacement, using the same sampled request indices for policy and Full (paired comparison). The implementation uses 1,000 replicates by default and a fixed, recorded seed.

Full is the reference and should have Δ = 0 and LCB90 = 0. If no non-Full candidate meets both constraints, Full remains the safe fallback.

**Interpretation:** this is a stricter validation selection guard than V5. It does not guarantee that the unknown population AUC drop is below δ; it requires the observed validation bootstrap lower bound to clear the margin.

## 5. Selection among feasible candidates

Among candidates satisfying both constraints, construct the two-objective Pareto frontier (maximize AUC and history saving) and select its orientation-invariant geometric knee using the same V5 algorithm and 0.01 minimum knee-distance threshold. If no stable knee exists, use Full fallback.

## 6. Test evaluation

After policy freeze, evaluate the selected policy once on held-out test. Report AUC, logloss, history interactions, selected-K distribution, and paired class-stratified bootstrap 90% CI versus Full. Test CI is post-selection reporting only and must not be used to change V6.

## 7. Important statistical qualification

The validation LCB is computed while comparing a predeclared finite candidate family on the same policy-tuning partition. It is a conservative candidate filter, but it is not a fully independent confirmatory guarantee after searching many candidates. For a formal non-inferiority claim, reserve an additional untouched confirmation set (or use a predeclared nested/selection-adjusted procedure) and evaluate the frozen policy there. The held-out test remains the final untouched evaluation in this experiment.

## 8. Interpretation boundaries

History interactions are a computational-work proxy, not measured cloud cost, latency, or energy. The experiment remains offline frozen-checkpoint evaluation, not online Bloom-FaaS serving. Bloom-FaaS effects must be measured separately in an integrated system experiment.
