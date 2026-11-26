# M5 preference-aware release candidate

## Reproducibility contract
Original budgets: 0, 1, 5, 10, 20, 50, 100, 2048 (Full is capped at 2048).
Extended grid adds 16, 32, 64, 128, 256, 512; it is an ablation, not a
replacement. Recent and semantic actions are available at positive intermediate
budgets; zero and Full are unique actions. The M4 checkpoint was not trained
explicitly at every original budget, so these are frozen-weight generalization
experiments. Do not infer that added budgets improve quality before testing.

The selector estimates excess BCE versus Full, then minimizes normalized
quality regret plus weighted resource cost subject to hard caps. AUC is
measured over a cohort, not predicted per request. Weights are preferences,
not an AUC guarantee. All adaptive actions pay for reading summaries; costs
exclude physical I/O, selector compute, and cloud billing. An infeasible
request fails without fetching history tokens (summary reads may already
have occurred). Original grid is the managed default.

## Native reproduction (existing KuaiRand processed data)
Use the original Python environment to produce artifacts:
```bash
source "$HOME/.config/shortvideo/m23.env"
python -m src.integration.m5_selector_experiment \
  --m4-dir reports/m4_research_v3 \
  --output reports/m5_release_v1 --http-sample 256 --bootstrap 1000
```
For a fresh dataset, first prepare the existing processed-data contract
(train/validation/test parquet, evaluation events, phase7 vocab and Full
history); run m4_release with a new run directory, then point --m4-dir at
that directory. Raw-data preprocessing is not replaced by this package.
Never publish secrets or private artifacts. The joblib files must come
from a trusted run. M4 checkpoint, INT8 bundle, signed snapshot directory,
vocabulary and M5 selector are required for serving.

## Container reproduction
The container is a CPU Linux reference, not measured cloud deployment.
Its pinned dependencies are an alternative to the original macOS environment.
Do not assume joblib compatibility across sklearn versions: **generate M4
and M5 artifacts under the same container dependency versions** for portable
deployment, or explicitly rebuild with your exact original library versions.
No artifact binaries are included in Git.

From repository root, build:
```bash
docker build -f deploy/m5/Dockerfile -t shortvideo-m5:rc1 .
```
Run M4 in the container, with writable processed data (legacy runner writes
some outputs), external reports and credentials in Docker env-file format
(KEY=value, no export statements):
```bash
docker run --rm --cpus 2 --memory 6g \
  --env-file "$M5_ENV_FILE" \
  -v "$M5_DATA_DIR:/app/data/processed" \
  -v "$M5_M4_DIR:/app/m4" \
  --entrypoint python shortvideo-m5:rc1 \
  -m src.integration.m4_release --run-dir /app/m4 --epochs 10 --batch-size 32 --repeats 1
```
M4 runner uses Git HEAD for provenance; for this command mount the checkout
.git directory at /app/.git read-only (add -v "$PWD/.git:/app/.git:ro")
and ensure git safe.directory for /app when needed.

Generate M5:
```bash
docker run --rm --cpus 2 --memory 6g \
  --env-file "$M5_ENV_FILE" \
  -v "$M5_DATA_DIR:/app/data/processed:ro" \
  -v "$M5_M4_DIR:/app/m4:ro" \
  -v "$M5_OUTPUT_DIR:/app/m5" shortvideo-m5:rc1 \
  --m4-dir /app/m4 --output /app/m5 --http-sample 256 --bootstrap 1000
```

Set M5_DATA_DIR, M5_M4_DIR, M5_OUTPUT_DIR, M5_POLICY_DIR and M5_ENV_FILE to
absolute paths. The env file supplies M23_HMAC_KEY_HEX and M23_API_TOKEN.
Use the same signing key as the snapshot builder. Create the policy directory,
copy business.example.json to business.json, then:
```bash
docker compose -f deploy/m5/compose.yaml up --build
```
Authenticated POST /recommend accepts history_key (existing evaluation
history key), candidate_id (mapped model ID), user_id (integer), policy
"preference". Managed serving rejects request-level preferences. The
response exposes chosen action, score, read tokens, MACs, logical bytes and
policy_revision. This is a replay-oriented research API, not a live user
history ingestion service. Missing keys return 404.

## Business control plane
Only the operator has write access to the host policy directory. Change
business.example.json into a new JSON document with an incremented revision.
From repository root (native Python environment with numpy):
```bash
python -m deploy.m5.update_policy --file "$M5_POLICY_DIR/business.json" --from-json ./new_policy.json
```
The directory, not the individual file, is mounted so atomic replacement
is visible inside the container. Each request pins the current revision.
Invalid updates retain last-known-good state while the process is alive;
startup fails on invalid policy. Rollback is a new revision containing
previous preferences. Single operator writer assumed; multi-replica
transactional control and durable policy audit are not implemented.
Do not expose this HTTP reference server publicly without TLS, a gateway,
auth/authorization, rate limits, and operational hardening.

## Evidence and status
Run python -m unittest discover -s deploy/m5/tests -v.
The release candidate is not certified until local and container runtime
gates pass. No Docker image has been published and no cloud measurements
are claimed. Compare original/extended grid profiles, fixed and context-free
baselines, action histograms, actual HTTP parity and costs in M5 reports.
The existing KuaiRand test is exploratory because it has been inspected.
