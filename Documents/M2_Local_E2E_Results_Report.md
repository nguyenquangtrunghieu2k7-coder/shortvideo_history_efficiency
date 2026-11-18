# M2 Technical Progress and Local End-to-End Evaluation Report

**Project:** Adaptive History Efficiency for Short-Video Recommendation  
**Milestone:** M2 — Persistent History Retrieval, Local Serving, Reliability, and Frozen-V4 Integration  
**Report date:** 18 November 2026  
**Implementation:** `feature/e2e-integration-contract-20261117`  
**Evaluation environment:** Local macOS machine; Python/Conda; KuaiRand held-out evaluation data  
**Status:** **LOCAL E2E VALIDATED** (offline Bloom-assisted inference and HTTP-to-V4 inference are independently validated; no claim of one combined Bloom+HTTP request path)

## Executive summary

M2 extends the earlier proof of concept (PoC) beyond an in-memory Bloom-filter and history-selection experiment into an **integrity-checked, persistent, authenticated, concurrent local retrieval service**, with independent correctness, stability, and frozen-recommender inference-parity evidence.

The PoC established that a frozen V4 adaptive policy and Bloom-assisted, sharded **in-memory** history retrieval could be combined with frozen recommender inference. M2 adds immutable SQLite snapshots, deterministic shard routing, Bloom metadata, atomic signed activation, a local HTTP/1.1 API, full-record equivalence testing, reliability/fault-injection tests, and an HTTP-to-frozen-V4 E2E acceptance gate.

**Principal outcomes:**

- **6,455** real held-out evaluation histories persisted in an **8-shard** snapshot, ID `0e43ec79835648358b09b304ec3596cd`.
- **M2.3G PASS:** **98,760 HTTP requests** across three consecutive full-data stability runs, **zero failures**, **8 concurrent workers**; **28/28** local regression tests passed in the reported run.
- **M2.3H PASS:** **6,455 HTTP-selected history requests**, **zero history mismatches**, **maximum absolute prediction difference = 0.0** against the offline exact-retrieval path.
- Frozen-V4 test AUC is **0.5968776837585248** on both offline and HTTP paths. The V4 policy reduces selected history interactions from **3,633,082** to **2,766,778** (**23.844878810882886% reduction**).
- The separately re-executed offline Bloom-assisted E2E run reports **zero Bloom false negatives**, **217 false positives** (resolved by exact lookup), and full retrieval equivalence.

These results establish a **reproducible local research demonstration**, not a public cloud/FaaS deployment or production latency claim.

## 1. Starting point: what the PoC already demonstrated

The pre-M2 PoC had two relevant capabilities:

1. **Adaptive history selection and frozen recommendation.** A validation-frozen V4 policy chooses an action from `{0, 1, 5, 10, 20, 50, 100, Full}`; frozen K-specific recommender checkpoints evaluate the selected history. The held-out evaluation reported AUC **0.5968776837585248** and history saving **0.23844878810882886**.
2. **Bloom-assisted in-memory sharded retrieval.** `src/integration/e2e_v4_bloom.py` builds Bloom filters over **evaluation-history IDs** (not video IDs), probes shards, and verifies Bloom MAYBE outcomes with exact dictionary lookup. This runner checks equivalence against the source histories before frozen recommender inference.

**PoC boundary:** The offline runner uses Python dictionaries and in-memory shard fan-out, preloads Full histories to compute V4 features, and does not measure disk-backed or HTTP retrieval. A Bloom filter is a *negative filter*, not a replacement for exact lookup; false positives are expected and verified.

## 2. M2 additions relative to the PoC

| Capability | Earlier PoC | M2 implementation and evidence |
|---|---|---|
| History storage | In-memory Python dictionaries | Immutable, read-only, **8-shard SQLite snapshots** with manifest and content hashes |
| History lookup | In-memory fan-out and exact dictionary checks | Deterministic SHA-256 shard routing; indexed exact lookup and optional Bloom-assisted shard pruning in the persistent store |
| Data integrity | Offline source comparison | Snapshot file hashes, schema/record checks, shard routing and Bloom validation; corrupted data fails closed at verification |
| Activation | No persisted active-snapshot contract | **HMAC-SHA256 signed `ACTIVE.json`**, manifest digest, atomic pointer replacement, pinned active snapshot |
| API | No network transport in PoC E2E | Bearer-authenticated **HTTP/1.1 loopback API**: `/health`, `/ready`, `/history?key=...&k=...` |
| Concurrency | Local simulation | Concurrent read-only SQLite and HTTP workers; keep-alive connection reuse |
| Correctness testing | Offline exact/Bloom equality | Independent SQLite truth vs HTTP, Full/0/1/5/out-of-range K, missing keys, repeated full-data tests |
| Reliability | Prototype-level tests | Fault injection, signed activation checks, restart/pinning tests, repeated 8-worker stability gate |
| Recommender integration | In-memory Bloom → frozen recommender | **HTTP-selected histories → frozen V4 recommender**, independently compared with offline exact inference |
| Evidence | Prototype experiment outputs | Machine-readable M2.3F/G/H JSON reports and reproducible gate commands |

### 2.1 Persistent snapshot design (M2.3A–C)

`src/integration/m23_persistent_store.py` builds immutable snapshots from pairs of `(evaluation_id, history_ids)`. It assigns each key to a shard using the first eight bytes of SHA-256 of the key, modulo shard count, and stores the ordered history as a JSON payload in a SQLite table with a primary-key index. Each shard has a Bloom filter and recorded metadata. A manifest contains schema version, snapshot ID, shard inventory, record counts and file checksums.

Read-only `HistoryStore` supports `indexed_exact`, `sharded_exact`, and `sharded_bloom` retrieval modes. Bloom MAYBE is followed by exact SQLite lookup; a missing/corrupt Bloom index can fall back to exact lookup. Full snapshot verification checks routing, counts, integrity and Bloom false negatives against stored keys.

The earlier local direct-SQLite concurrency benchmark reported **32,275 requests**, **8 workers**, indexed-exact **p95 ≈ 0.956 ms**, and **≈ 9,767 requests/s**. This was a warm-cache, direct SQLite experiment, **not** an HTTP serving measurement; its timing is not directly comparable to the HTTP benchmark below.

### 2.2 Atomic activation and authenticated local serving (M2.3D–E)

`src/integration/m23_activation.py` verifies a candidate snapshot before atomically replacing a signed active pointer. The HMAC key is supplied through the environment, not stored in Git. The API verifies the signed active snapshot at startup and **pins that snapshot until restart**. The server is bound to loopback (`127.0.0.1`), requires a bearer token, uses read-only SQLite lookups and returns the selected most-recent K interactions.

The API exposes `/ready` to identify its pinned snapshot and `/history` to retrieve Full or nonnegative K. Missing histories return 404. This is **local HTTP without TLS**, not a remotely exposed service. Startup verification does not prove continuous integrity after startup.

### 2.3 Full-data correctness, keep-alive and stability (M2.3F–G)

`src/integration/m23f_api_gate.py` constructs **independent ground truth directly from SQLite shard rows** and tests every real history at five K values: `Full`, `0`, `1`, `5`, and `len(history)+3`. It also tests missing keys expecting HTTP 404. For **6,455** histories, this gives `6,455 × 5 + 645 = 32,920` requests per full run.

An earlier repeated-run failure was traced to **macOS ephemeral-port exhaustion** (`OSError errno 49`) when creating a new TCP connection per request. The fix enabled HTTP/1.1 keep-alive and **one persistent connection per worker**, rather than increasing concurrency without bounds. Three consecutive full-data runs subsequently passed.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Requests | 32,920 | 32,920 | 32,920 |
| Failures | 0 | 0 | 0 |
| Workers | 8 | 8 | 8 |
| Wall time (s) | 6.6129 | 6.5941 | 6.5341 |
| Throughput (requests/s) | 4,978.17 | 4,992.34 | 5,038.17 |
| p50 (ms) | 1.365 | 1.378 | 1.358 |
| p95 (ms) | 2.119 | 2.112 | 2.091 |
| p99 (ms) | 2.676 | 2.584 | 2.536 |

Across the three runs: **98,760 requests**, **0 failures**; mean reported throughput **≈ 5,002.89 requests/s**. The `M2.3G_STABILITY_EVIDENCE` gate passed **10/10 checks** for consistent snapshot, complete coverage, expected counts, 8 workers, keep-alive mode, no failures and valid timings. The reported local test suite passed **28/28 tests**.

**Interpretation:** These are local loopback, warm-system, pinned-snapshot measurements. They do not establish remote network latency, serverless behavior, cloud cost or production service-level objectives.

## 3. Local E2E methodology and results

### 3.1 Offline V4 + Bloom + frozen recommender (re-executed)

The runner `src/integration/e2e_v4_bloom.py` loads the held-out `Full` test split, computes request-level history features, applies the **already frozen** V4 policy, and builds in-memory Bloom indexes over the 6,455 evaluation IDs. It verifies exact/Bloom retrieval equivalence, slices each history to selected K and runs the **existing frozen K-specific recommender checkpoint**. No policy retraining or recommender retraining occurs.

| Metric | Result |
|---|---:|
| Status | **PASS** |
| Requests / evaluation examples | 6,455 |
| AUC | **0.5968776837585248** |
| History saving | **0.23844878810882886** (23.8449%) |
| Full history interactions | 3,633,082 |
| Selected history interactions | 2,766,778 |
| Retrieval equivalence | **true** |
| Bloom false negatives | **0** |
| Shard candidates | 51,640 |
| Bloom rejections | 44,968 |
| Bloom MAYBEs / exact lookups | 6,672 |
| Exact hits | 6,455 |
| Bloom false positives | 217 |

The Bloom filter rejects **44,968 of 51,640 shard candidates** (≈ **87.08%**), reducing exact dictionary lookups from **51,640** to **6,672** (≈ **87.08% fewer**). This is a reduction in *lookup attempts in the in-memory benchmark*, **not** an additional 87% reduction in selected history length, nor proof of a network latency speedup. In this particular implementation, measured in-memory Bloom comparison time (**0.05998 s**) exceeded direct in-memory exact comparison time (**0.00781 s**); the result establishes safe filtering and fewer exact probes, not a demonstrated speedup.

Frozen V4 action distribution:

| K | Requests |
|---|---:|
| Full | 3,871 |
| 100 | 508 |
| 50 | 1,055 |
| 20 | 251 |
| 10 | 133 |
| 5 | 136 |
| 1 | 142 |
| 0 | 359 |
| **Total** | **6,455** |

The offline runner reported **23.44054 s total script time**, including setup and both retrieval comparisons, and **1.02314 s** measured frozen-recommender inference. These are different measurement scopes and must not be conflated with serving latency.

### 3.2 HTTP retrieval → frozen V4 inference parity (M2.3H)

`src/integration/m23h_http_v4_e2e.py` independently validates that the **persistent, authenticated local HTTP history service** supplies exactly the histories required by the same frozen V4 decisions:

1. Load the same 6,455 held-out test requests and frozen V4 policy.
2. Compute frozen V4 actions from the original Full histories (preloaded for feature extraction).
3. Request each action's selected `K` from the pinned M2.3 HTTP snapshot.
4. Compare each returned history, in order, against the exact offline selected history.
5. Run the frozen K-specific recommender checkpoints separately on the offline and HTTP-selected histories.
6. Compare request-level predictions and held-out AUC.

| Metric | Verified M2.3H result |
|---|---:|
| Gate | `M2.3H_HTTP_TO_FROZEN_V4_LOCAL_E2E` |
| Status | **PASS** |
| Snapshot | `0e43ec79835648358b09b304ec3596cd` |
| HTTP history requests | **6,455** |
| History mismatches | **0** |
| Maximum absolute prediction delta | **0.0** |
| Prediction tolerance | `atol=1e-7`, `rtol=0` |
| Prediction parity | **true** |
| Offline exact AUC | **0.5968776837585248** |
| HTTP AUC | **0.5968776837585248** |
| History saving | **0.23844878810882886** |
| HTTP retrieval segment (s) | **1.81206** |
| Offline inference segment (s) | **0.61578** |
| HTTP-path inference segment (s) | **0.46423** |
| Total script time (s) | **40.12625** |

**Interpretation:** The HTTP-selected histories and offline exact-selected histories are identical on all 6,455 evaluated requests. The two inference passes produce numerically identical predictions, preserving the frozen V4 AUC and history saving. The **1.81206 s** retrieval segment is a sequential local loopback batch timing for this run; the **40.12625 s** overall script includes model/data setup and repeated inference, and is **not** an online per-request end-to-end serving latency measurement. The difference between the two inference segment timings is not evidence that HTTP inference is faster; the same frozen model is evaluated in both paths under different execution order/warm-up conditions.

### 3.3 Relationship between the two E2E gates

These two experiments are **complementary but distinct**:

- **Offline E2E:** V4 → in-memory Bloom-assisted history lookup → exact verification → frozen recommender.
- **HTTP parity E2E:** V4 → persistent SQLite snapshot via authenticated local HTTP → frozen recommender, compared against offline exact-selected histories and predictions.

**Not yet demonstrated:** a *single request path* that performs Bloom-assisted filtering **inside** the HTTP retrieval service and then frozen V4 inference. The HTTP service uses deterministic shard routing plus exact SQLite lookup. The Bloom-assisted in-memory runner and the HTTP-based runner are independently validated; their results must not be presented as one combined serving pipeline.

## 4. Verification, reproducibility, and evidence

| Evidence | File / command | Acceptance |
|---|---|---|
| Offline V4 + Bloom E2E | `src/integration/e2e_v4_bloom.py` | PASS, AUC and saving reproduced, zero Bloom false negatives |
| Persistent store | `src/integration/m23_persistent_store.py` | Immutable shard snapshots and integrity validation |
| Activation and API | `src/integration/m23_activation.py`, `m23_local_api.py` | Signed activation, authenticated loopback serving |
| Full HTTP correctness | `src/integration/m23f_api_gate.py` | Independent SQLite truth, 32,920 requests per run |
| Stability evidence | `src/integration/m23g_release_gate.py` | 3/3 full runs, 98,760 requests, zero failures |
| HTTP-to-V4 E2E parity | `src/integration/m23h_http_v4_e2e.py` | 6,455 requests, zero history mismatches, zero prediction delta |
| Reliability regressions | `tests/test_m23*.py` | 28/28 tests in reported local run |

Run from the repository root, with the existing frozen artifacts and local KuaiRand data:

```bash
conda activate shortvideo-e2e
PYTHONPATH=. python -m src.integration.e2e_v4_bloom \
  --mode v4 --bloom --output reports/e2e_v4_bloom_local.json

# Start the API in another terminal with M23_API_TOKEN and
# M23_HMAC_KEY_HEX already exported (never commit these secrets).
PYTHONPATH=. python -m src.integration.m23_local_api \
  --root data/processed/m23_snapshots --port 8765

# With the API running and M23_API_TOKEN set in the client terminal:
PYTHONPATH=. python -m src.integration.m23h_http_v4_e2e \
  --snapshot-id 0e43ec79835648358b09b304ec3596cd \
  --output reports/m23h_http_v4_e2e.json

# Revalidate the three saved full-run benchmark reports:
PYTHONPATH=. python -m src.integration.m23g_release_gate \
  reports/m23g_keepalive_run1.json \
  reports/m23g_keepalive_run2.json \
  reports/m23g_keepalive_run3.json \
  --output reports/m23g_release_gate.json
```

Reports are generated locally and are not assumed to be committed to the repository. Secrets, history payloads and authentication headers should not be included in shared logs.

## 5. What has been achieved, and what remains

### Validated achievements

- Converted the history retrieval PoC into a **persistent, versioned, integrity-checked local data-serving subsystem**.
- Added authenticated, atomic snapshot activation and a pinned local HTTP/1.1 retrieval API.
- Tested full-data retrieval semantics, missing keys, concurrent access and repeated-run stability; resolved a real macOS socket-exhaustion issue through connection reuse.
- Demonstrated that the **existing frozen V4 model and recommendation quality are preserved** when selected histories come from the persistent HTTP service.
- Retained independent Bloom-assisted offline E2E evidence: exact equivalence, zero false negatives and 217 harmless false positives resolved by exact lookup.
- Produced machine-readable acceptance gates suitable for a research demonstration and reproducibility appendix.

### Scope limitations and next research steps

1. **No unified Bloom+HTTP serving path yet.** The current HTTP API uses exact deterministic-shard SQLite retrieval, not Bloom-based fan-out pruning in the request handler.
2. **Full histories are preloaded for V4 feature extraction.** The experiments do not yet demonstrate fully lazy online feature extraction and request-time selection.
3. **Local-only performance.** Loopback measurements do not establish WAN/network overhead, TLS, cloud/FaaS deployment, cold starts, autoscaling, energy consumption or billing.
4. **Benchmark comparability.** Direct SQLite, in-memory Bloom and HTTP benchmarks have different workloads/timing boundaries; their throughput/latency values are not causal comparisons.
5. **Operational scope.** Startup snapshot verification and pinned serving are demonstrated, but continuous runtime integrity, remote security hardening and production observability are outside this milestone.

## 6. Conclusion

**M2 is complete for the defined local research-demonstration objective.** Compared with the earlier in-memory PoC, the project now has a persistent snapshot-backed history store, authenticated and stable local HTTP serving, reproducible full-data correctness evidence, and a frozen-V4 HTTP inference-parity gate. On all **6,455** held-out evaluation requests, HTTP-selected histories matched the offline reference exactly, producing **zero prediction difference**, **AUC 0.5968776837585248**, and **23.8449% fewer selected history interactions** than Full. Across three concurrent full-data HTTP stability runs, the service processed **98,760 requests with zero failures**.

The evidence supports **local E2E correctness and reliability**, not a claim of production/cloud deployment or of an already unified Bloom-assisted HTTP request path. This distinction is important for accurate interpretation and future experimental design.
