# Policy × Refresh Schedule — Experimental Results (Exploratory)

**Experiment:** Frozen Fixed-K versus Adaptive V4 under scheduled 24-hour history windows  
**Branch:** `experiment/sliding-window-history-vs-disjoint-1day`  
**Status:** `EXPLORATORY_POLICY_REFRESH_MATRIX` — **not final scientific acceptance**  
**Evidence:** User-provided terminal output from `python -m src.integration.policy_refresh_matrix` (18 November 2026).  
**Reproduction:** `PYTHONPATH=. python -m src.integration.policy_refresh_matrix --batch-size 512 --bootstrap 400`

## 1. Experimental contract

- **Total frozen test requests:** 6,455.
- **Common OOV-free evaluation cohort:** 5,254 requests (81.3943% of full test); same requests and labels across all 36 configurations.
- **Refresh schedules:** S0 = 24 h, S1 = 12 h, S2 = 6 h, S3 = 3 h.
- **Policies:** Fixed K = 0, 1, 5, 10, 20, 50, 100, Full, and frozen Adaptive V4.
- **Model and preprocessing:** Frozen video vocabulary, K-specific checkpoints, V4 policy, and standard logs. No retraining or test-time policy tuning.
- **Quality:** ROC-AUC (higher is better).
- **History cost:** Mean selected history interactions per evaluated request (lower is better). This is **not** total resource cost: refresh workload, feature extraction, memory, and serving cost are distinct.
- **Uncertainty:** 400 paired user-cluster bootstrap replicates for Adaptive V4 minus Fixed-Full, by schedule.

Prior strict reconstruction audit: **6,455 / 6,455** event-time histories match frozen Phase 7; event-time OOV = **0**. Scheduled histories can contain OOV, so this experiment uses the same 5,254 requests that are OOV-free under **all four schedules**. The results **cannot automatically generalize** to the 1,201 excluded requests.

## 2. Full 36-configuration matrix

`Mean K` is the actual mean number of interactions used, after truncation by available history length, **not** the nominal fixed K. All values are transcribed from the reported terminal output.

| Schedule | Policy | ROC-AUC | Mean selected history length |
|:--|:--|--:|--:|
| S0 (24h) | Fixed-0 | 0.584472 | 0.00 |
| S0 (24h) | Fixed-1 | 0.511416 | 0.96 |
| S0 (24h) | Fixed-5 | 0.543062 | 4.80 |
| S0 (24h) | Fixed-10 | 0.555760 | 9.55 |
| S0 (24h) | Fixed-20 | 0.564159 | 18.92 |
| S0 (24h) | Fixed-50 | 0.561282 | 46.35 |
| S0 (24h) | Fixed-100 | 0.587414 | 89.82 |
| S0 (24h) | Fixed-Full | **0.606683** | 516.61 |
| S0 (24h) | Adaptive-V4 | 0.580544 | 383.56 |
| S1 (12h) | Fixed-0 | 0.584472 | 0.00 |
| S1 (12h) | Fixed-1 | 0.512515 | 0.97 |
| S1 (12h) | Fixed-5 | 0.545345 | 4.85 |
| S1 (12h) | Fixed-10 | 0.555304 | 9.66 |
| S1 (12h) | Fixed-20 | 0.552557 | 19.17 |
| S1 (12h) | Fixed-50 | 0.577776 | 46.96 |
| S1 (12h) | Fixed-100 | 0.589106 | 91.01 |
| S1 (12h) | Fixed-Full | **0.621637** | 529.61 |
| S1 (12h) | Adaptive-V4 | 0.595179 | 395.94 |
| S2 (6h) | Fixed-0 | 0.584472 | 0.00 |
| S2 (6h) | Fixed-1 | 0.507204 | 0.98 |
| S2 (6h) | Fixed-5 | 0.535172 | 4.90 |
| S2 (6h) | Fixed-10 | 0.542825 | 9.76 |
| S2 (6h) | Fixed-20 | 0.546868 | 19.38 |
| S2 (6h) | Fixed-50 | 0.576562 | 47.47 |
| S2 (6h) | Fixed-100 | 0.591193 | 92.01 |
| S2 (6h) | Fixed-Full | **0.620709** | 539.03 |
| S2 (6h) | Adaptive-V4 | 0.604799 | 405.79 |
| S3 (3h) | Fixed-0 | 0.584472 | 0.00 |
| S3 (3h) | Fixed-1 | 0.510539 | 0.99 |
| S3 (3h) | Fixed-5 | 0.533779 | 4.92 |
| S3 (3h) | Fixed-10 | 0.541459 | 9.79 |
| S3 (3h) | Fixed-20 | 0.545136 | 19.44 |
| S3 (3h) | Fixed-50 | 0.565346 | 47.74 |
| S3 (3h) | Fixed-100 | 0.583897 | 92.73 |
| S3 (3h) | Fixed-Full | **0.618858** | 546.41 |
| S3 (3h) | Adaptive-V4 | 0.597932 | 418.83 |

## 3. Adaptive V4 versus Fixed-Full

AUC deltas below are calculated from the displayed six-decimal AUC values (therefore approximate); confidence bounds are transcribed exactly.

| Schedule | Fixed-Full AUC | Adaptive V4 AUC | Δ AUC (V4 − Full) | V4 mean K | Full mean K | Approx. history saving | 95% paired user-cluster bootstrap CI |
|:--|--:|--:|--:|--:|--:|--:|:--|
| S0 (24h) | 0.606683 | 0.580544 | −0.026139 | 383.56 | 516.61 | 25.75% | [−0.047162, −0.008031] |
| S1 (12h) | 0.621637 | 0.595179 | −0.026458 | 395.94 | 529.61 | 25.24% | [−0.045701, −0.006630] |
| S2 (6h) | 0.620709 | 0.604799 | −0.015910 | 405.79 | 539.03 | 24.72% | [−0.033160, +0.001109] |
| S3 (3h) | 0.618858 | 0.597932 | −0.020926 | 418.83 | 546.41 | 23.35% | [−0.038339, −0.006010] |

**Interpretation:** Fixed-Full has higher point-estimate AUC than Adaptive V4 under **every** schedule. V4 saves approximately 23–26% of selected history interactions. For S0, S1, and S3, the reported 95% bootstrap intervals for V4 − Full lie strictly below zero; S2's interval crosses zero. This **does not establish equivalence/non-inferiority** for S2; a predeclared non-inferiority margin and suitable test would be required.

**Critical baseline:** Fixed-0 achieves AUC **0.584472** using **zero** history interactions for all schedules. Adaptive V4 at S0 has *lower* AUC (0.580544) while using 383.56 interactions/request. Thus, **S0 Adaptive V4 is dominated by Fixed-0** for the two objectives (maximize AUC, minimize selected history). This must be acknowledged when assessing the value of the adaptive selector.

## 4. What the quality–history relationship actually shows

1. **More history is not uniformly better for small K:** Fixed-1 AUC is only 0.507–0.513, well below Fixed-0 at 0.584472. Some intermediate K values are also non-monotonic.
2. **Full history produces the highest AUC in every schedule:** 0.606683 (S0), 0.621637 (S1), 0.620709 (S2), 0.618858 (S3). The highest point-estimate quality is **S1 Fixed-Full**, not S2 Adaptive V4.
3. **Adaptive V4 is a quality–history compromise, not the quality winner:** V4 uses ~384–419 interactions/request versus ~517–546 for Full, but sacrifices AUC.
4. **Refresh frequency interacts with policy:** Fixed-Full is highest at S1 (12h); Adaptive V4 is highest at S2 (6h). Therefore, the refresh interval preferred by a policy cannot be inferred from V4-only results.
5. **Fixed-100 is a meaningful low-history competitor:** Its AUC is 0.587414 / 0.589106 / 0.591193 / 0.583897 across S0–S3, using only ~90–93 interactions/request. It should be explicitly compared against V4 at an equal history budget or via Pareto analysis.

## 5. Pareto interpretation and a reporting caveat

The runner returned the following **quality–selected-history-only** non-dominated entries:

```text
S0:Fixed-0       S0:Fixed-100     S0:Fixed-Full
S1:Fixed-0       S1:Fixed-100     S1:Fixed-Full
S1:Adaptive-V4
S2:Fixed-0       S2:Fixed-100     S2:Adaptive-V4
S3:Fixed-0
```

**Do not treat this raw list as a verified canonical Pareto frontier.** There is a clear issue in the code's tie handling: all four Fixed-0 points have identical AUC and zero history cost, so they are all listed. More importantly, the reported list appears **internally inconsistent**: S0 Fixed-Full (AUC 0.606683, mean K 516.61) is dominated by S1 Fixed-Full (AUC 0.621637, mean K 529.61) **only if** history cost is ignored; with history cost included neither dominates the other. However, S0 Fixed-100 (AUC 0.587414, mean K 89.82) is dominated by S1 Fixed-100 (AUC 0.589106, mean K 91.01) **only if** history cost is ignored; with history cost included neither dominates the other. The main unresolved issue is that the Pareto calculation uses exact unrounded total counts, while this report displays rounded means; the raw scores and costs should be used to recompute and independently validate the frontier. **Do not claim statistical Pareto dominance from point estimates alone.**

A complete deployment Pareto must consider **at least three distinct objectives**: maximize ROC-AUC, minimize selected-history/retrieval cost, and minimize refresh processing cost. Refresh-age/SLA may add a fourth constraint or objective. The existing refresh workload numbers are **observed-job interaction proxies**, not production CPU/GPU/latency measurements; the cohort differs from the 5,254-request quality sample.

## 6. Scientific status and next steps

**Supported now:** strict Phase 7 parity, frozen vocabulary contract on event-time histories, controlled common-cohort 36-cell matrix, descriptive quality–history trade-off, and paired user-cluster intervals for Adaptive V4 versus Fixed-Full.

**Not yet supported:** causal attribution to refresh alone on the full test cohort; V4 non-inferiority to Full; superiority over Fixed-100 at matched cost; calibrated production resource savings; a statistically validated 3D Pareto frontier; or a test-independent policy/refresh decision rule.

**Next analyses required:**

1. Independently recompute quality–history Pareto with explicit tolerance and tie rules, and validate the result against the 36-cell CSV.
2. Plot the **36-point AUC versus mean selected history** scatter, distinguishing refresh schedule and policy; overlay exact Pareto frontier, label Fixed-0, Fixed-100, Full, and V4.
3. Run paired user-cluster comparisons for key policy alternatives (V4 versus Fixed-100; S1 Full versus S2 Full; S2 V4 versus S1/S3 V4), with multiplicity awareness.
4. Add measured resource accounting (refresh construction CPU time, memory, selected-history retrieval and model inference), keeping refresh and serving costs separate.
5. Specify quality floor, history budget, refresh budget and freshness SLA **before** choosing a deployment operating point. Any selection made after inspecting held-out test results is exploratory and requires independent confirmation.

## 7. Artifact provenance

- Runner: `src/integration/policy_refresh_matrix.py`.
- Raw metrics: `reports/experiments/policy_refresh_matrix.json` and `.csv` on the experimenter's local machine.
- Paired predictions: `reports/experiments/policy_refresh_matrix.npz` on the experimenter's local machine.
- Existing diagnostic plots: `reports/experiments/figures/` (prior experiment; not the 36-policy quality–history plot).
- This Markdown file records **observed terminal output**, not a separately rerun or independently verified computation.

**Frozen baseline branch must not be modified.**
