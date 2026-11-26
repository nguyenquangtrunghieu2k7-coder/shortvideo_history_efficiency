# Research progress checkpoint — from initial E2E to Adaptive V2

**Date:** 2026-11-18  
**Repository:** `nguyenquangtrunghieu2k7-coder/shortvideo_history_efficiency`  
**Working branch:** `experiment/sliding-window-history-vs-disjoint-1day`  
**Frozen baseline branch:** `feature/e2e-integration-contract-20261117` (**DO NOT MODIFY**)  
**Status:** Experimental development and diagnostic validation; **no demonstrated adaptive Pareto superiority**.

## 1. Research objective and invariants

The primary question is whether **request-level adaptive history selection** can meet a recommendation-quality requirement with **lower total resource cost** than the best feasible Fixed-K policy. The refresh schedules S0–S3 are an experimental testbed, **not** the central scientific claim.

Non-negotiable rules:

1. Preserve frozen Phase 7 preprocessing, vocabulary, original checkpoints, original Adaptive V4, original E2E integration contract and frozen baseline branch.
2. Keep training, validation/model selection, calibration, and test roles explicit. Do not optimize using test outcomes. Previously opened test results are exploratory and cannot serve as pristine confirmatory evidence for subsequent changes.
3. Compare all adaptive methods with **strong Fixed-K alternatives**, especially Fixed-50 and Fixed-100, using identical request cohorts and identical recommender weights when isolating selector effects.
4. Distinguish selected history interactions, inference-loop time, policy feature/prediction overhead, retrieval and index costs, CPU/memory, and actual end-to-end resource cost. **No inference of production savings from mean K alone.**
5. Report negative results and confidence intervals. An interval containing zero does **not** prove equivalence or non-inferiority.
6. Do not label a validation-selected policy as independently validated on the same data. No production approval until a separate, defensible evaluation protocol is satisfied.

## 2. Initial frozen E2E and integration findings

- Initial E2E combined frozen K-specific recommenders, Adaptive V4, and in-memory Bloom-assisted sharded history retrieval.
- Bloom is an in-memory exact-lookup prefilter with exact verification; it is **not** proof of production distributed I/O performance. The initial M2.3-style retrieval materialized **Full history first** and only then sliced to K. Thus lower K did not establish K-aware retrieval savings.
- Frozen test size: **6,455 requests**. Strict event-time history reconstruction previously matched **6,455/6,455**, with zero event-time OOV under the frozen contract.
- For scheduled history experiments, **5,254 complete-case OOV-free requests** (799 users) were shared across the 36 configurations. The 1,201 excluded requests limit generalization.
- Schedules: S0 24h, S1 12h, S2 6h, S3 3h; 8 Fixed-K actions (0, 1, 5, 10, 20, 50, 100, Full) plus frozen Adaptive V4.
- Under S2, frozen independent-K checkpoints yielded: Fixed-Full **AUC 0.620709, mean K 539.03**; Fixed-100 **AUC 0.591193, mean K 92.01**; Adaptive V4 **AUC 0.604799, mean K 405.79**. Adaptive V4 was **not** established as the best quality–cost option.
- Across schedules, Adaptive V4 saved approximately 23–26% of selected history interactions vs Full but lost point-estimate AUC. Detailed matrix and uncertainty: `Documents/Policy_Refresh_36_Configuration_Results_20261118.md`.
- Resource profiling found substantial V4 model-prediction overhead and methodological limitations in one-off inference timings. Resource results are diagnostic, not production-grade.

## 3. Shared checkpoint architectural ablation

**Script:** `src/integration/shared_checkpoint_policy_ablation.py`.

A single frozen Full checkpoint was reused across all K (history truncation only). S2 complete-case results:

| Policy | AUC | Mean K | Inference loop (one-off) |
|---|---:|---:|---:|
| Fixed-100 | 0.605310 | 92.01 | ~0.048 s |
| Fixed-Full | 0.620709 | 539.03 | ~0.338 s |
| Adaptive V4 | 0.608277 | 405.79 | ~0.193 s |

Measured V4 feature construction ~0.139 s and policy-model prediction ~2.121 s for 5,254 requests. These timings are single-run diagnostics; device/batching/warmup affect comparisons. **Shared inference reduced checkpoint-switching overhead, but V4 policy overhead overwhelmed the model-only savings in this setup.** Fixed-100 remained a strong competitor.

## 4. Shared Variable-K recommender V1

**Script:** `src/integration/train_shared_variable_k.py`.  
**Local checkpoint:** `models/experimental_shared_variable_k.pth` (not assumed committed to GitHub).

- Initialized from frozen Full checkpoint without altering it.
- Trained on **30,118 TRAIN** requests, with per-example randomized K over 0,1,5,10,20,50,100,Full.
- Selected checkpoint on **6,454 VALIDATION** requests, using unweighted mean AUC over eight K.
- Training: seed 42, requested 12 epochs; **best epoch 10**, best mean validation AUC **0.6027885983**.
- Epoch 10 validation AUC: K0 0.591837; K1 0.539161; K5 0.567254; K10 0.598347; K20 0.605380; K50 0.633946; K100 0.638366; Full 0.648018.
- The checkpoint is trained to support variable K but **is not evidence of adaptive selection superiority**. Low K=1/5 performed poorly. Training loss continued to decrease while validation metrics fluctuated.

**Evaluation script:** `src/integration/evaluate_shared_variable_k.py`. Existing validation/test comparisons for frozen V4 **transferred** to shared checkpoint:

| Comparison | Validation ΔAUC (95% user-cluster CI) | Test ΔAUC (95% user-cluster CI) |
|---|---|---|
| V4-transfer − Fixed-100 | +0.011691 [−0.006284,+0.027825] | +0.002825 [−0.009629,+0.014292] |
| V4-transfer − Fixed-Full | +0.002038 [−0.015967,+0.017905] | −0.011224 [−0.021731,+0.000274] |

All intervals cross zero; no confirmed superiority or equivalence. **The test has been viewed**: future development cannot claim it remains an untouched final holdout. These event-time frozen-history comparisons must not be conflated with S2 scheduled complete-case results.

## 5. Lightweight selector V1 — negative result

**Scripts:** `src/integration/train_lightweight_selector_v1.py`, `src/integration/audit_lightweight_frontier.py`, `src/integration/audit_fixed50_uncertainty.py`.

- Multi-output Ridge predicts per-request **log-loss advantage relative to Full**, not true per-request AUC gain.
- User-disjoint fit/calibration within validation: **2,088 calibration requests / 321 users**.
- Penalty selected on the calibration subset, with an AUC tolerance of 0.01 relative to Full. This is a **selection-biased diagnostic** because the shared checkpoint itself was also selected on validation.
- Chosen V1 penalty 0.01; AUC **0.646087**, mean K **119.539**, mean interaction proxy **0.207903**. This proxy is **not** 79.2% end-to-end savings.

**Same-cohort fixed-K frontier:**

| Policy | Calibration AUC | Mean K |
|---|---:|---:|
| Fixed-50 | **0.653491** | **48.429** |
| Fixed-100 | 0.650204 | 93.208 |
| Fixed-Full | 0.654331 | 576.858 |
| Lightweight V1 | 0.646087 | 119.539 |

Both Fixed-50 and Fixed-100 dominate Lightweight V1 by **point-estimate AUC and mean K**. Fixed-50 saves **~91.60% selected history interactions** vs Full on this cohort, **not** 91.60% total cost.

Fixed-50 paired user-cluster uncertainty (1,000 bootstrap draws):
- Fixed-50 − Full: ΔAUC **−0.000840**, 95% CI **[−0.024070,+0.022042]**.
- Fixed-50 − Fixed-100: ΔAUC **+0.003287**, 95% CI **[−0.014162,+0.021154]**.

Both intervals contain zero. **No statistical equivalence or non-inferiority claim is justified.**

## 6. Adaptive V2 Fixed-50-first gate — negative result

**Script:** `src/integration/train_adaptive_v2_gate.py`.  
**Local artifact:** `models/experimental_adaptive_v2_gate.joblib` (not assumed committed).

- Fixed-50 used as default/reference; candidate actions 0,10,20,50,100,Full.
- Ridge predicts log-loss advantage vs Fixed-50; penalty discourages extra interactions.
- Calibration tolerance: **AUC no more than 0.005 below Fixed-50**, measured on same validation calibration subset.
- Outcome: **`NO_GAIN_OVER_FIXED50_PROXY`**. Selected penalty 0.001; AUC **0.648927**, ΔAUC vs Fixed-50 **−0.004564**; mean K **109.107** vs Fixed-50 **48.429** (**2.253×** as many interactions).
- Action counts on 2,088 requests: K0 623, K10 0, K20 78, K50 407, K100 456, Full 524.
- Thus Fixed-50 dominates selected V2 in point-estimate AUC and mean K. Additional feature/policy overhead further weakens the adaptive cost case, but **total E2E costs were not measured for V2**.
- This is a negative diagnostic result, not evidence that adaptive selection is impossible in general.

## 7. Scientific interpretation and next research gate

**Current conclusion:** Shared Variable-K V1 is a functional experimental recommender; neither lightweight V1 nor Fixed-50-first V2 establishes an adaptive quality–cost Pareto improvement. **Fixed-50 is the strongest current low-history reference on the reported calibration cohort**, though its AUC equivalence to Full is not proven.

**Next steps (not yet completed):**

1. Conduct an **oracle upper-bound diagnostic on development data only**, using labels solely to estimate whether request-specific choice has potential. Oracle is not deployable and must never be presented as a learned policy.
2. Rebuild a clean **user-disjoint development protocol**: train selector on proper out-of-fold predictions, calibrate on separate users, and reserve independent evaluation; account for the fact that Shared V1 checkpoint was selected on existing validation and existing test has been viewed.
3. Investigate why short histories (especially K1/K5) degrade quality, whether current request-time features predict useful K changes, and whether the log-loss surrogate is aligned with ranking AUC.
4. Build a **K-aware retrieval path** (fetch selected suffix rather than materializing Full), then measure feature extraction, policy inference, history access, model inference, CPU, memory, and warmed repeated latency under consistent hardware/protocol.
5. Compare adaptive candidates against **all feasible Fixed-K policies**, with user-cluster uncertainty and declared quality tolerance; preserve all negative findings.

## 8. Reproduction entry points and evidence locations

- `src/integration/policy_refresh_matrix.py` — frozen 36-cell policy/schedule matrix.
- `src/integration/benchmark_policy_refresh_resources.py` — diagnostic resource matrix.
- `src/integration/shared_checkpoint_policy_ablation.py` — shared frozen Full architectural ablation.
- `src/integration/train_shared_variable_k.py` — shared Variable-K training.
- `src/integration/evaluate_shared_variable_k.py` — paired shared-checkpoint evaluation; `reports/experiments/shared_variable_k_{validation,test}_evaluation.{json,csv,npz}` generated locally.
- `src/integration/train_lightweight_selector_v1.py` — lightweight V1; local JSON/joblib.
- `src/integration/audit_lightweight_frontier.py` — same-cohort Fixed-K frontier.
- `src/integration/audit_fixed50_uncertainty.py` — paired bootstrap.
- `src/integration/train_adaptive_v2_gate.py` — Fixed-50-first V2; local JSON/joblib.
- Frozen and experimental model artifacts under `models/` must be checked locally; their existence in GitHub is **not asserted**.

### Explicitly prohibited interpretations

- “Adaptive V2 is efficient because mean K is lower than Full” — false relative to Fixed-50 and ignores overhead.
- “Fixed-50 and Full are statistically equivalent” — not established.
- “Validation-selected V1/V2 is independently verified on validation” — false.
- “Already viewed test is pristine for new tuning decisions” — false.
- “Bloom in-memory benchmark proves production savings” — false.

**Decision checkpoint:** Preserve frozen artifacts, keep Shared V1 as experimental backbone, keep Fixed-50 as the primary low-history baseline, archive V1/V2 as negative results, and advance only after oracle/development-protocol and actual cost audits.
