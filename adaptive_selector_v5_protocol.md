# Adaptive History Selector V5 — Pareto Knee-point Protocol

## 1. Research objective

V5 selects a history policy that balances recommendation quality and behavioral-history efficiency. It does not minimize history cost at any quality price, and it does not maximize AUC regardless of history cost.

For policy \(\pi\), define:

- Quality: \(Q(\pi)=AUC(\pi)\).
- History saving: \(S(\pi)=1-HistoryCost(\pi)/HistoryCost(Full)\).

Both objectives are maximized.

## 2. Data protocol and leakage controls

V5 inherits the V4 data and model protocol:

1. Use the same frozen recommender checkpoints, feature construction, candidate actions, user-disjoint validation split, and uplift-predictor training procedure as V4.
2. Keep predictor development and policy-tuning users disjoint. Keep the inner uplift-model fit and model-selection groups disjoint.
3. Use only the validation policy-tuning partition to calculate candidate policy AUC, history cost, feasibility, Pareto dominance, and the knee point.
4. Do not use held-out test labels, predictions, AUC, history cost, or bootstrap results to select or modify the policy.
5. Freeze the selected policy and its parameters before evaluating the held-out test split once.

The test bootstrap confidence interval is post-selection inference only. It must not trigger a policy change.

## 3. Candidate generation

Generate the same candidate family as V4:

- Fixed-K policies for K = 0, 1, 5, 10, 20, 50, 100, Full.
- Learned policies generated from the V4 uplift predictor over the predeclared lambda grid (default 41 values, including lambda = 0).

No candidate grid or threshold may be changed after inspecting test results.

## 4. Feasibility constraint

Retain the predeclared V4 validation quality constraint:

\[
AUC(\pi) \ge AUC(Full)-\delta, \qquad \delta=0.03.
\]

Feasibility is evaluated using the point estimate on the validation policy-tuning partition. Full is always included as a feasible reference candidate. The threshold is fixed before test evaluation.

## 5. Pareto frontier construction

For each feasible candidate, calculate:

\[
Q_i=AUC(\pi_i), \qquad
S_i=1-\frac{HistoryCost(\pi_i)}{HistoryCost(Full)}.
\]

A candidate is Pareto-dominated if another feasible candidate has both:

- quality greater than or equal to its quality; and
- history saving greater than or equal to its history saving;

with at least one strict inequality.

Keep only non-dominated candidates. Keep Full in the candidate set; it provides the quality-oriented endpoint (normally zero history saving).

For duplicate (Q, S) coordinates, retain one deterministic representative: higher AUC first, then lower history cost, then lexicographically smaller policy identifier. Record how many duplicates were collapsed.

## 6. Knee-point detection

Order the distinct Pareto points by increasing history saving. Identify:

- A: the quality-oriented endpoint (maximum AUC; deterministic tie-break by lower history cost).
- B: the history-saving endpoint (maximum history saving; deterministic tie-break by higher AUC).

Normalize each objective over the Pareto frontier:

\[
x_i=\frac{S_i-S_{min}}{S_{max}-S_{min}}, \qquad
y_i=\frac{Q_i-Q_{min}}{Q_{max}-Q_{min}}.
\]

For each interior point, calculate its signed perpendicular distance to the endpoint chord:

\[
d_i=\frac{(B_x-A_x)(P_{iy}-A_y)-(B_y-A_y)(P_{ix}-A_x)}
{\sqrt{(B_x-A_x)^2+(B_y-A_y)^2}}.
\]

The sign only describes which side of the chord the point lies on. It depends on frontier orientation and is **not** a criterion for accepting or rejecting a knee. Define the geometric knee score as:

\[
K_i=|d_i|.
\]

Select the interior point with the largest (K_i). Tie-break by higher validation AUC, then higher history saving, then lexicographically smaller policy identifier. Retain both signed distance and absolute knee score in the frontier output for diagnostics.

A maximum distance alone can still force a knee onto an almost-linear frontier. Therefore, if the maximum normalized distance is below 0.01, treat the knee as not reliably identifiable and use Full fallback. This threshold is defined on normalized objective coordinates and is independent of held-out test results.

### Degenerate or non-identifiable cases

- If fewer than three distinct Pareto points remain, a knee is not identifiable.
- If either objective has zero range, normalization is undefined and a knee is not identifiable.
- If the maximum absolute perpendicular distance is below 0.01, the frontier is treated as nearly linear and no stable knee is declared.
- In any non-identifiable case, select Full as the conservative fallback and record the reason.

The implementation must validate finite inputs, positive Full history cost, and valid AUC-drop tolerance. It must handle duplicate points and floating-point tolerances explicitly.

## 7. Validation outputs

Save a complete candidate table and a Pareto-frontier table containing at least:

- policy identifier, policy type, lambda or fixed K;
- validation AUC and AUC delta versus Full;
- history cost ratio, history saving, and mean history interactions;
- feasibility flag and Pareto-frontier membership;
- normalized quality and saving coordinates;
- signed knee distance and absolute knee score;
- selected-for-test flag and knee/fallback status.

Also save the selected policy metadata, endpoint identifiers, number of feasible candidates, number of Pareto points, duplicate count, and fallback reason if applicable.

## 8. Held-out test evaluation

After freezing the validation-selected policy, apply it unchanged to test. Report:

- AUC and logloss;
- mean and total history interactions;
- history reduction and cost ratio versus Full;
- selected-K distribution;
- paired, class-stratified bootstrap 90% CI for AUC(policy) - AUC(Full).

The confidence interval is descriptive/post-hoc and cannot be used to retune V5.

## 9. Interpretation boundaries

History interactions are a proxy for computational work, not a direct measurement of cloud monetary cost, energy, or end-to-end latency. The experiment uses offline inference with frozen recommender checkpoints; it is not evidence of online serving or Bloom-FaaS performance.

Bloom-FaaS integration must be evaluated as a separate system-level component. Do not attribute additional savings, latency improvements, or energy reductions to Bloom unless those quantities are measured in an integrated experiment.

## 10. Comparison with earlier versions

Report V3, V4, and V5 using their own frozen validation-selected policies and the same held-out test split/protocol where applicable. Do not compare V5 validation results against V3/V4 test results. V5's main methodological change is the validation selection rule: a Pareto knee rather than minimum feasible history cost (V3) or weighted geometric utility (V4).
