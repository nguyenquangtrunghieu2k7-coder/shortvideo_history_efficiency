# Adaptive History Selector V4 — Balanced Trade-off Protocol

## Objective

V4 retains the frozen recommender checkpoints, request-level AUC-contribution uplift predictor, user-disjoint validation development/tuning split, and held-out test protocol from V3. Only the policy-selection objective changes.

First apply the validation quality constraint:

\[
AUC(\pi) \ge AUC(Full)-\delta, \quad \delta=0.03.
\]

For every candidate policy, compute normalized quality retention and history saving:

\[
Q(\pi)=clip\left(\frac{AUC(\pi)-[AUC(Full)-\delta]}{\delta},0,1\right)
\]

\[
S(\pi)=clip\left(1-\frac{HistoryCost(\pi)}{HistoryCost(Full)},0,1\right).
\]

Then maximize the weighted geometric utility among feasible candidates:

\[
U(\pi)=Q(\pi)^{0.6}S(\pi)^{0.4}.
\]

The default weights are configurable but must be positive and sum to one. Full remains a feasible fallback and has zero history saving, hence zero utility. If all feasible non-Full candidates have zero quality headroom, the tie-break may select Full.

## Selection and tie-breaks

1. Evaluate Fixed-K and learned-lambda candidates on the policy-tuning validation subset only.
2. Exclude candidates with point-estimate AUC below the Full AUC minus the configured tolerance.
3. Maximize balanced utility.
4. Tie-break by higher validation AUC, then lower history cost.
5. Freeze the selected policy, weights, predictor, and normalization scale.
6. Evaluate the held-out test once. Test metrics and bootstrap intervals are post-hoc only and must not change policy selection.

## Interpretation limits

History interactions are a computational-work proxy, not measured cloud cost, latency, or energy. This is an offline frozen-checkpoint experiment, not online Bloom-FaaS serving. The uplift target remains a request-level proxy; the global AUC constraint is measured directly on the validation tuning subset.
