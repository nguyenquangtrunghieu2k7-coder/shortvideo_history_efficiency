# Research Roadmap: Robust, Preference-Conditioned Adaptive History Selection

**Status:** Proposed research direction (not an implemented or validated result)  
**Date:** 2026-11-17  
**Project:** Adaptive History Efficiency for Short-Video Recommendation  
**Scope:** Research-grade methodology with enterprise-oriented serving and configurable policies.

## 1. Motivation and current baseline

The project currently selects a history budget K from {0, 1, 5, 10, 20, 50, 100, Full} using frozen recommendation checkpoints, request/history features, uplift prediction and validation-calibrated policy selection. The baseline recommender is a mean-pooled history model; reported held-out Full AUC is approximately **0.614246**. V4 reports AUC **0.596878** with **23.84%** fewer processed history interactions, while V6's more conservative confidence-bound feasibility test falls back to Full. These are existing reported results, **not newly reproduced measurements**.

The immediate scientific challenge is not simply finding the smallest feasible K. With few feasible policies, Pareto-knee heuristics become poorly identified, and conservative feasibility can result in Full fallback. An enterprise partner also needs meaningful behavior under changing quality/cost/risk preferences and temporal shifts.

**Central research objective:** develop and evaluate a **risk-aware, preference-conditioned adaptive history selection framework** that finds the best defensible quality–efficiency trade-off for the current operating constraints, measures uncertainty and stability, and explicitly reports infeasibility rather than claiming guaranteed savings.

## 2. Research questions

- **RQ1 — Recommendation quality:** Can stronger history modeling improve Full AUC and reduce the quality gap at shorter K?
- **RQ2 — Preference adaptation:** Can one calibrated policy family respond to changing business weights and constraints without retraining recommendation checkpoints for every preference?
- **RQ3 — Robustness:** How reliably do feasibility, utility, selected K and operating costs generalize across rolling temporal windows, seeds, and datasets?
- **RQ4 — End-to-end efficiency:** Do adaptive history selection and Bloom-assisted retrieval improve *measured* latency/compute/memory after accounting for policy inference and filtering overhead?

## 3. Design principles and invariants

1. Preserve existing V3/V4/V5/V6 checkpoints, results, and code paths as immutable comparison baselines.
2. Never tune on the final held-out test set; separate recommender training, selector development, calibration, and final evaluation.
3. Use a consistent vocabulary, target, candidate universe, timestamp convention, and evaluation cohort within each comparison.
4. Compare fixed K, Full, V4/V5/V6, and new policies under identical data splits and scoring protocols.
5. Report both point estimates and uncertainty; no guarantee of feasibility across every dataset or preference.
6. History interactions processed are a **work proxy**, not automatically latency, cloud spend, or energy savings.
7. Bloom membership positives are *possible matches*, requiring exact verification; Bloom cannot be assumed to accelerate a pipeline without benchmarking.
8. All experiments must retain configuration, code commit, data fingerprint, checkpoint identifiers, seeds, split manifests, metrics, and audit results. Large joblib/checkpoint artifacts may remain off GitHub; maintain hashes and reproducible artifact manifests.

## 4. Recommendation-quality track

### 4.1 Diagnose before replacing the baseline
Audit label definition (e.g., long_view), prevalence, candidate sampling, temporal leakage, feature availability at prediction time, per-user/per-history-length performance, score distributions, calibration and confidence intervals. Establish simple popularity/metadata baselines and learning curves.

### 4.2 Candidate model upgrades
- Stronger video/candidate features and metadata, only if available before the prediction timestamp.
- Recency- or time-aware weighted pooling.
- Candidate-aware attention (DIN-style) over history.
- Sequential baselines (e.g., GRU4Rec/SASRec) where data volume and order support them.
- Controlled hyperparameter tuning and regularization.

### 4.3 Two comparison protocols
- **A: K-specific checkpoints** (existing approach): each K has its own trained model.
- **B: Shared checkpoint:** a model trained to handle variable history lengths, optionally with randomized truncation, evaluated at every K. This helps isolate the impact of K from checkpoint differences.

Rebuild score matrices and recalibrate selectors whenever recommender checkpoints change. Do not compare a new recommender's quality to an old selector's metrics as if only one factor changed.

## 5. Policy research: beyond a Pareto knee

### 5.1 Separate feasibility from preference optimization

Let `pi` denote a request-level history-selection policy, `A(pi)` its AUC, `C(pi)` measured cost or history-work proxy, and `A_F` Full AUC. A configurable feasibility set can be defined as:

```text
F(theta) = {pi:
  LCB_(1-alpha)[A(pi)-A_F] >= -epsilon,
  C(pi) <= B                 [optional, when a real budget is specified],
  latency(pi) <= L_max       [optional, when measured]
}
```

The confidence-bound criterion is a **proposed operating mode**, not a universally valid guarantee; bootstrap construction, dependence structure, and out-of-distribution behavior must be audited. Also report a point-estimate-only feasibility mode to compare against V4. Full is the reference safety fallback for relative AUC-drop constraints, but may not satisfy independent latency/budget limits; flag that case as **infeasible**, not as a successful fallback.

Within feasible candidates, choose according to business preferences:

```text
pi*(theta) = argmax_(pi in F(theta))
  [w_q * Q(pi) + w_s * S(pi) - w_r * R(pi)]
```

Here Q, S and R require explicit definitions, scaling, and validation. Risk R must not double-count the same uncertainty already encoded as a hard constraint without justification. An alternative is to optimize quality and savings with a separate risk budget. A weighted sum is an **engineering baseline**, not itself a novel research contribution.

### 5.2 Expand the policy space, not just the knee heuristic
**Stage A — dense/continuous calibration:** sweep lambda, constraint thresholds, and preference vectors; deduplicate identical request-level action assignments; evaluate the actual global AUC for each candidate. A dense lambda grid does not imply a continuous set of distinct policies.

**Stage B — randomized/mixture policies:** mix two policies at the request level using an explicitly seeded assignment probability rho. Evaluate the realized mixed predictions and bootstrap AUC directly. **AUC is not generally linear in rho**. Report variance across mixture seeds and practical serving reproducibility.

**Stage C — learned preference-conditioned selector:** learn `pi_phi(K | x, theta)`, where x contains strictly pre-decision bounded-cost features and theta describes operating preferences. Train/calibrate across sampled preferences; evaluate on held-out preferences and time folds. Compare against simpler calibrated-policy libraries to justify complexity.

The goal is not to eliminate Full fallback at any cost. If reduced histories cannot preserve ranking quality, fallback is a valid, reportable outcome.

### 5.3 Preference stability and update regularization
Study neighboring preference vectors and policy transitions. Optional switching penalty:

```text
pi_t = argmax_(pi in F_t)
  [U_(theta_t)(pi) - gamma * D(pi, pi_(t-1))]
```

Define D as action-switch rate or cost change on a fixed reference cohort. Hard feasibility constraints take precedence over stability. Discrete actions may create legitimate discontinuities; report them rather than promising smoothness everywhere.

### 5.4 Metrics
- Test AUC, delta vs Full, paired user-aware/bootstrap CI, violation frequency.
- Mean/median/p95 selected K, history interaction saving, action histogram.
- Preference sweep: utility, feasibility rate, action-switch rate, cost elasticity, regret against a clearly defined candidate oracle (diagnostic only).
- Temporal: mean and worst-fold AUC delta, cost, feasibility, selection variance.
- Calibration: predicted uplift vs realized aggregate AUC; subgroup diagnostics and out-of-time drift.

**Important:** V4's per-request AUC contribution/uplift targets are surrogate labels relative to a Full-score reference; they are not an additive decomposition of globally mixed-policy AUC. Final policy decisions must be evaluated with actual combined scores and global AUC.

## 6. Temporal and multi-dataset experiments

Replace sole reliance on a fixed 1:1:1-day split with a configurable rolling-origin engine:
- Fixed temporal split (reference).
- Rolling 1-day and 3-day windows **where sufficient data exists**.
- Expanding training windows and fixed-length moving training windows.
- Different history lookback horizons and recency-vs-long-term selection.
- Temporal gaps/embargo where feature construction or label horizons require them.

For every fold, document timestamp boundaries, training/validation/selector/calibration/test roles, user overlap assumptions, checkpoint provenance, and leakage checks. If the dataset is too short for credible independent folds, reduce fold count or use another dataset; do not fabricate coverage.

Potential additional datasets to assess for schema and licensing fit: KuaiRec, MicroLens, and a non-short-video dataset such as MovieLens as an out-of-domain stress test. **These are candidates, not confirmed integrated datasets.** Dataset adapters must specify entity IDs, event chronology, label mapping, observation windows, and permitted pre-event features. Avoid pooling incomparable AUC values across datasets.

## 7. Bloom and end-to-end integration

### 7.1 Contract to establish before coding
Explicitly define:
- What exact IDs/keys are inserted into each Bloom filter and the scope of that set (global, user, shard, time window).
- Which records are probed and why a negative result avoids expensive downstream work.
- Whether selected K changes filter construction, retrieval, or probe work.
- How MAYBE results are exact-verified; handling false positives and preventing false negatives from incorrect data partitioning.
- Versioning, hash-function compatibility, filter refresh, data staleness, and serving correctness.

Bloom filtering does not automatically help top-K history retrieval. If the workload lacks a selective membership/join step, Bloom may add overhead with no benefit. The existing local Bloom benchmark reports substantial rejection but slower wall-clock time than exact-set lookup; integration must be justified by a measured bottleneck.

### 7.2 Minimum comparison matrix
- Full recommender without Bloom.
- Fixed-K recommender without Bloom.
- Adaptive selector without Bloom.
- Bloom-only pipeline at an explicitly defined workload.
- Adaptive selector + Bloom, with identical correctness checks.

Measure data access, feature extraction, selector inference, Bloom build/merge/probe/exact verification, recommendation inference, serialization/network (when applicable), throughput, p50/p95 latency, peak memory, and total resource usage. Keep offline filter construction and online serving costs separate. Only claim end-to-end acceleration if measurements support it.

## 8. Proposed milestones and acceptance gates

| Milestone | Deliverables | Acceptance gate |
|---|---|---|
| **M0: Audit and freeze** | Baseline manifest, quality/leakage audit, split and checkpoint registry | Existing metrics traceable; no unintended baseline modifications |
| **M1: Offline E2E harness** | Dataset/split adapter, unified evaluation API, Bloom contract, Full/fixed-K/V4/Bloom comparisons | Identical inputs and verified outputs; component timings; correctness tests |
| **M2: Recommender improvement** | Stronger model ablations, shared vs K-specific training | Fair matched splits; reproducible AUC and confidence intervals |
| **M3: Policy engine** | Dense calibration, feasible preference optimizer, optional mixture/conditioned selector | Preference sweep, infeasibility reports, comparisons vs V4/V5/V6 |
| **M4: Robustness** | Rolling-window and sensitivity suite | Worst-fold/mean metrics, uncertainty, action stability and regression tests |
| **M5: Generalization and deployment evidence** | Multi-dataset adapters, runtime profiling, reproducibility package | Cross-dataset results, limitations, verified end-to-end cost claims |

The order can overlap in implementation, but M0's measurement and data contracts are prerequisites for credible comparisons.

## 9. Suggested experiment matrix

| Dimension | Candidate settings |
|---|---|
| Recommender | mean pooling; time-aware; candidate-aware; sequence model |
| Checkpoint protocol | K-specific; shared variable-K |
| History policy | Full; fixed-K; V4; V5; V6; new preference-aware |
| Quality tolerance | epsilon sweep including 0.01, 0.02, 0.03 (illustrative) |
| Preference weights | quality weights 0.1 to 0.9; cost weights complement; risk modes separately |
| Policy search | sparse lambda; dense lambda; mixture; conditional selector |
| Temporal split | reference; rolling; expanding; lookback sensitivity |
| Dataset | KuaiRand first; others after adapter audits |
| Bloom | disabled; enabled at a verified membership/join bottleneck |

Use staged experiments rather than the full Cartesian product. First isolate one variable, then test interactions for the most promising settings. Reserve final holdout(s) for locked comparisons.

## 10. Artifact structure (proposed)

```text
configs/experiments/{dataset}/{experiment_id}.yaml
src/evaluation/{temporal_splits,metrics,bootstrap,robustness}.py
src/policy/{preference_optimizer,policy_library,conditional_selector}.py
src/integration/{history_retrieval,bloom_adapter,serving_pipeline}.py
results/{experiment_id}/{manifest,metrics,decisions,timings,audits}
Documents/RESEARCH_ROADMAP_PREFERENCE_ROBUST_HISTORY_SELECTION.md
```

These paths are **proposed**, not claims that files already exist.

## 11. Open decisions and non-goals

**Decisions to resolve before implementation:** operational cost unit (interactions vs CPU/latency), quality tolerance and statistical confidence, whether serving uses K-specific or shared checkpoints, Bloom's actual target join/membership bottleneck, temporal fold feasibility, and availability/licensing of additional datasets.

**Non-goals:** guaranteed AUC improvement, universal savings, forcing a Pareto knee when there is no meaningful knee, claiming Bloom speedups from rejection ratio alone, or treating off-GitHub joblib artifacts as missing experiments.

## 12. Immediate next steps

1. Freeze current metrics/configs and register local joblib/checkpoint hashes.
2. Audit Full AUC by label, time, user activity and history length; establish stronger-but-fair recommender baselines.
3. Write the E2E data/serving contract and test whether Bloom actually removes a costly operation.
4. Implement a small preference-aware policy library with feasibility and explicit Full/infeasible outcomes before a neural conditional selector.
5. Build rolling-window evaluation and weight-sweep reports; expand datasets only after the first reproducible results.

---
**Document intent:** an evolving research plan and engineering decision record. Proposed algorithms, paths and expected benefits are hypotheses until verified experimentally.
