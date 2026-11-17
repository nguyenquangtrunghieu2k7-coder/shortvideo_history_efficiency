# M2.3 — Persistent Retrieval and BLOOM-FaaS Integration Assessment

## Scope
M2.2 establishes correctness on frozen KuaiRand test histories and a warm-cache local SQLite comparison. It is **not** production serving validation. Do not change frozen V4 policy, recommender checkpoints, or test labels.

## External reference: BLOOM-FaaS-2025 (user-provided archive, not vendored)
- `deploy.yaml`: Knative Service with Kubernetes resource limits, NFS PVC, and MinIO environment.
- `operations/libs/S3Metadata.py`: PyArrow S3/Parquet metadata scanning.
- `operations/libs/ParquetS3Funcs.py`: DuckDB + PyArrow S3 access, Bloom extension loading.
- `operations/funcs/FaaSEngine.py`: Flask HTTP endpoints for scan operations.
- `benchmarks/run-bloom-faas-s3.py`: baseline versus Bloom-enabled cloud benchmark configuration.
- `bfextension/src/lib.rs`: Rust DuckDB extension; requires separate ABI/security review.

These are architecture references, not drop-in compatible components: TPC-H Parquet scanning differs from request-key history retrieval. Do not copy the archive, credentials, internal IPs, binaries, or deployment manifests into this repo.

## Critical security precondition
The uploaded reference deploy manifest contains literal MinIO access/secret environment values and private endpoints; other files also contain credential defaults. Treat them as potentially exposed, rotate/revoke any live credentials, remove hardcoded secrets, use Kubernetes Secrets or a workload identity, and review infrastructure manifests before deployment.

## M2.3 acceptance gates
1. Persistent immutable SQLite snapshot and atomic manifest activation, with SHA-256 checksums and schema/version contracts.
2. Bloom indexes tied to snapshot ID, shard ID, key count and hash configuration; stale/missing/corrupt Bloom must fall back to exact retrieval, never silently drop keys.
3. Exact indexed routing as primary baseline; Bloom only for unknown shard discovery, with no false negatives and exact verification for positives.
4. Restart, corruption, duplicate-key, missing-key, snapshot replacement and concurrent-reader tests.
5. Latency p50/p95/p99, throughput, CPU, RSS, index-build time, disk bytes and network bytes (when cloud backend exists), warm and cold-ish cache, repeated paired runs.
6. Separate offline feature-store generation from online retrieval; V4 feature equivalence to frozen Full histories before claiming storage savings.
7. No production-ready claim until live backend, concurrency, auth, monitoring, failure recovery, security and end-to-end AUC equivalence pass.

## Next implementation
Build an isolated persistent history store adapter and fault-injection test suite locally. Then evaluate a cloud object-storage adapter using short-lived credentials, with a dedicated environment and cost/latency instrumentation. Preserve raw test history ordering and model vocabulary indices.
