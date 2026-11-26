# M1 — End-to-End Integration Contract (Draft v0.1)

**Date:** 2026-11-17  
**Branch:** `feature/e2e-integration-contract-20261117`  
**Status:** Proposed integration specification; implementation and benchmarks pending.

## 1. Purpose

Connect the existing adaptive-history policy, history retrieval, optional Bloom-assisted membership/join, frozen recommendation inference, and evaluation **without changing the established V3–V6 research baselines**. The primary acceptance criterion is correct, reproducible, apples-to-apples end-to-end measurement—not a presupposed speedup.

## 2. Verified current module boundaries

- `src/policy/train_adaptive_selector_v4.py` loads Full validation/test histories, builds bounded-cost history features, scores all K-specific frozen recommender checkpoints, calibrates the V4 policy on a validation tuning partition, then evaluates on test.
- `src/policy/adaptive_v4_policy.py` defines K actions `[0,1,5,10,20,50,100,"Full"]` and selection/evaluation utilities.
- `src/data/history.py` stores raw video IDs in history artifacts, with a frozen raw-ID-to-model-index vocabulary; model index 0 is padding.
- `src/bloom_pipeline/pipeline.py` performs local partition/build/OR-merge/probe, then calculates Bloom correctness against an exact set.
- `src/bloom_pipeline/kuairand_experiment.py` builds a **global union** of history video IDs and probes event video IDs; this does **not** currently filter a request's selected history and does not demonstrate integrated serving speedup.
- Frozen checkpoints and joblib artifacts may be stored locally rather than committed; their absence from GitHub is not evidence that experiments are missing.

## 3. Logical data flow

```text
EvaluationRequest (user_id, candidate_video_id, event_time, label [evaluation only])
      |
      +--> bounded pre-decision history features --> frozen V4 policy --> selected K
      |                                                            |
      +--> raw history retrieval ------------------------------> newest-K slice
                                                                   |
                                            [optional verified Bloom join]
                                                                   |
                                   raw IDs --> frozen vocab --> padded model IDs
                                                                   |
                                         frozen K-specific recommender checkpoint
                                                                   |
                                             score, timings, selected K, audits
                                                                   |
                                      global AUC, saving, latency, Bloom metrics
```

**Important:** the existing K-specific architecture requires the model checkpoint to match the chosen K. A later shared-checkpoint experiment is a distinct protocol.

## 4. Interface contracts

### 4.1 EvaluationRequest

Required fields: `evaluation_id` (unique), `user_id`, `candidate_video_id`, `event_time_ms`, `history_raw_ids` (ordered oldest to newest), `label` (evaluation only). Preserve original order and IDs; no target leakage.

### 4.2 PolicyDecision

```json
{
  "evaluation_id": "example",
  "selected_k": 20,
  "policy_version": "v4",
  "policy_artifact_sha256": "required-in-real-run",
  "feature_schema_version": "required-in-real-run"
}
```

`selected_k` must belong to the frozen action set. `Full` means all history permitted by the current reconstruction contract, not unrestricted all-time data.

### 4.3 SelectedHistory

`selected_history_raw_ids = history_raw_ids[-K:]` for integer K > 0, `[]` for K=0, and the entire permitted history for Full. Verify `len(selected_history) == min(K, len(history))` for integer K. The history-cost proxy counts selected interactions, including repeated IDs.

### 4.4 BloomJoinContract (optional, must be justified)

Inputs: **exactly defined** build-side key set, probe records, key namespace, filter configuration, snapshot/version, and exact verifier.

Outputs: definite-NO records, MAYBE records, exact-confirmed matches, false positives, false negatives, build/merge/probe/exact-verification timing, bytes, and cardinality.

**Correctness:** Bloom MUST NOT drop a true member. Every MAYBE requiring semantic correctness MUST be exact-verified. Compare resulting exact joins with an exact-only reference. If the target workload has no costly membership/join operation, set Bloom mode to `disabled` and report `not_applicable` rather than inserting a meaningless Bloom stage.

**Open decision:** whether the genuine expensive operation is video-metadata join, candidate eligibility, history-store shard membership, or another lookup. The existing global union-of-history-IDs probe is a standalone benchmark, not a justified serving contract.

### 4.5 PredictionRecord

Required: `evaluation_id`, `label`, `selected_k`, `history_length_before`, `history_length_after`, `score`, `checkpoint_id`, `policy_id`, `bloom_mode`, and stage timings. Scores must be finite and aligned by evaluation_id, never by implicit row order alone.

## 5. Experiment modes and comparisons

1. **FULL_EXACT:** Full history, exact/no Bloom, Full checkpoint.
2. **FIXED_K_EXACT:** each K with matching checkpoint, exact/no Bloom.
3. **V4_EXACT:** frozen V4 decisions, exact/no Bloom.
4. **BLOOM_ONLY:** a documented join workload with Bloom and exact verification; may be benchmark-only if no valid recommender join is found.
5. **V4_BLOOM:** V4 plus Bloom at the **same validated join**, when a meaningful join is identified.

Keep the same test requests, frozen checkpoints, vocab and data version across comparable modes. Do not infer a V4_BLOOM speedup from an unrelated Bloom-only benchmark.

## 6. Required checks

### Correctness gates
- Split provenance and no test-set calibration.
- History chronology: strictly eligible pre-event interactions; explicit tie policy.
- Frozen vocabulary mapping and padding semantics unchanged.
- Policy action in allowed K set; correct truncation and order.
- Same request IDs and labels in all compared modes.
- With Bloom enabled, exact-confirmed outputs equal exact-only join outputs; **zero false negatives**.
- With identical histories and checkpoint, Bloom-on and Bloom-off recommendation scores match within stated floating-point tolerance.
- No silent fallback if checkpoint/joblib/manifest is missing; fail closed and identify missing artifact.

### Measurement gates
- Per-stage timings: features, policy, history fetch/slice, Bloom build (offline), Bloom probe, exact verification, preprocessing, model inference.
- Warm-up and repeated runs; p50/p95 latency and throughput; peak memory when available.
- End-to-end speedup relative to the same exact baseline, **including added overhead**.
- History savings = 1 - sum(selected history lengths)/sum(Full history lengths); do not label it CPU/latency savings.
- Global AUC and paired uncertainty, plus policy action distribution.

## 7. Suggested implementation seams (new files; not yet implemented)

```text
src/integration/contracts.py
src/integration/history_selector.py
src/integration/bloom_join.py
src/integration/e2e_runner.py
src/evaluation/e2e_metrics.py
tests/test_integration_contracts.py
tests/test_integration_bloom_equivalence.py
configs/experiments/kuairand/e2e_v4.yaml
```

Favor adapters around existing tested functions instead of copying V4 training logic. Separate offline calibration from frozen-policy inference. Do not use test labels inside selection or feature construction.

## 8. Local artifacts required for the first real run

- Processed KuaiRand evaluation events/history, vocabulary, and relevant splits.
- All frozen K-specific recommender checkpoints used by V4.
- V4 selector joblib **plus metadata needed to reconstruct the frozen selected policy** (mode, lambda/fixed K, cost scale, feature names, checkpoint hashes, training code version).
- Exact source/snapshot for the Bloom join, once its actual role is established.

If any artifact is unavailable, run isolated contract tests and synthetic correctness checks, but **do not report empirical AUC or latency**.

## 9. Execution order

1. Verify artifact schemas and define exact join target.
2. Implement deterministic exact-only Full/fixed-K/V4 E2E runner and compare scores/AUC to existing frozen baseline.
3. Implement optional Bloom adapter with exact verification and equivalence tests.
4. Measure identical workloads with Bloom off/on and report any regression honestly.
5. Only after M1 correctness passes: recommender improvements, rolling windows, preference robustness, multi-dataset evaluation.

## 10. M1 exit criteria

- A reproducible CLI run produces manifest, per-request decisions/scores, correctness report and aggregate metrics.
- Exact-only runner reproduces reference predictions within tolerance on the same artifacts.
- Bloom integration, **if applicable**, produces equivalent exact-verified outputs and no false negatives.
- Timings account for selector and Bloom overhead; report both improvements and regressions.
- No modification to validated V3–V6 baselines.

**Decision gate:** If Bloom cannot be tied to a meaningful serving membership/join workload, keep it as a separate experimental module and do not claim V4+Bloom is a useful integrated optimization.
