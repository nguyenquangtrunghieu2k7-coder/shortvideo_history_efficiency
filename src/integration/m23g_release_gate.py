"""Validate three independent M2.3G local loopback full-data reports.

This is an evidence gate, not proof of production deployment readiness.
"""
import argparse
import json
from pathlib import Path


def evaluate(paths):
    if len(paths) != 3 or len({str(Path(p).resolve()) for p in paths}) != 3:
        raise ValueError("Provide three distinct report paths")
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    ids = {r.get("snapshot_id") for r in reports}
    counts = {r.get("source_records") for r in reports}
    checks = {
        "all_pass": all(r.get("status") == "PASS" and r.get("failures") == 0 for r in reports),
        "same_snapshot": len(ids) == 1 and None not in ids,
        "same_record_count": len(counts) == 1 and all(isinstance(n, int) and n > 0 for n in counts),
        "full_coverage": all(r.get("tested_records") == r.get("source_records") for r in reports),
        "same_request_count": len({r.get("requests") for r in reports}) == 1,
        "expected_request_count": all(
            r.get("requests") == 5 * r.get("tested_records", -1)
            + max(100, r.get("tested_records", 0) // 10)
            for r in reports
        ),
        "eight_workers": all(r.get("workers") == 8 for r in reports),
        "keepalive_mode": all(
            r.get("connection_mode") == "persistent_http11_per_worker"
            and r.get("benchmark_type") == "loopback_http_pinned_verified_snapshot_keepalive"
            for r in reports
        ),
        "no_classified_failures": all(
            not r.get("failure_types") and not r.get("failure_by_k")
            and not r.get("failure_by_expected_status") for r in reports
        ),
        "valid_timing": all(
            isinstance(r.get("wall_seconds"), (int, float)) and r["wall_seconds"] > 0
            and isinstance(r.get("throughput_rps"), (int, float)) and r["throughput_rps"] > 0
            and all(isinstance(r.get(k), (int, float)) and r[k] >= 0
                    for k in ("p50_ms", "p95_ms", "p99_ms"))
            and r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]
            for r in reports
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate": "M2.3G_STABILITY_EVIDENCE",
        "checks": checks,
        "snapshot_id": next(iter(ids)) if len(ids) == 1 else None,
        "total_requests": sum(r.get("requests", 0) for r in reports),
        "total_failures": sum(r.get("failures", 0) for r in reports),
        "scope": "local_loopback_only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/m23g_release_gate.json"))
    args = parser.parse_args()
    result = evaluate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
