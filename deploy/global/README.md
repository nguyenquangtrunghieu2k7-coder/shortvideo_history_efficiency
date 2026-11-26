# Global Policy v1 release candidate

Run from the repository root in the existing native Python environment:
```bash
source "$HOME/.config/shortvideo/m23.env"
python -m src.integration.global_release --sample 128 --workers 1 4 8 --repeats 2
```
This uses M4 checkpoint, M5 test predictions and M6 validation cache. No
retraining, full action scoring, or cloud-lab rerun. Missing inputs fail.
Output: reports/global_release_v1/REPORT.md, summary.json, quality_cost.csv,
compiled_plans.json, http_trials.csv, http_comparison.csv, runtime/.
The runtime includes model weights, vocab, signed snapshot, calibration,
initial business policy, smoke request and a SHA256 manifest. It contains
research data and must be shared only as permitted. No signing key or
API token is copied. Keep cache and original artifacts intact.

## Docker
Docker Desktop or Docker Engine with Compose is required. From repo root:
```bash
export GLOBAL_RUNTIME_DIR="$PWD/reports/global_release_v1/runtime"
export GLOBAL_ENV_FILE="$HOME/.config/shortvideo/global-docker.env"
umask 077
printf 'M23_HMAC_KEY_HEX=%s\nM23_API_TOKEN=%s\n' "$M23_HMAC_KEY_HEX" "$M23_API_TOKEN" > "$GLOBAL_ENV_FILE"
docker compose -f deploy/global/compose.yaml up --build -d
python -m deploy.global.smoke --runtime "$GLOBAL_RUNTIME_DIR"
docker compose -f deploy/global/compose.yaml logs --tail=30
```
Run smoke after service reports READY. The key must match the signed
snapshot. Keep credentials outside Git and share through a secure channel.
CPU Linux dependency pins differ from original macOS; PyTorch checkpoint
compatibility and numerical parity must be verified by the smoke check.
No sklearn joblib artifact is loaded in Global serving.
Stop using docker compose -f deploy/global/compose.yaml down.

## Policy
Example revision 2:
```json
{"revision":2,"preferences":{"quality_weight":1.0,"max_read_tokens":50},"average_tokens":30}
```
Save as new_policy.json and publish with:
```bash
python -m deploy.global.publish --runtime "$GLOBAL_RUNTIME_DIR" --from-json new_policy.json
```
The directory mount exposes atomic file replacements. Average budget is
an expectation under calibration distribution; repeated keys share a stable
assignment and drift may change realized averages. Hard limits are conservative
nominal K caps. Only the operator can update policy; clients cannot override.
Smoke fixture assumes initial revision-1 policy; run before editing it.
Manifest records initial package state; policy is intentionally mutable.
The HTTP server is loopback-only on host, not internet-facing production.
No cloud deployment or Docker certification is claimed until verified.
