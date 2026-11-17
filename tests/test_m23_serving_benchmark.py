import tempfile
import unittest

from src.integration.m23_persistent_store import build_snapshot
from src.integration.m23_serving_benchmark import benchmark, percentile


class ServingBenchmarkTests(unittest.TestCase):
    def test_percentile(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([1, 3], 50), 2)
        self.assertEqual(percentile([1, 2, 3], 95), 2.9)

    def test_small_concurrent_benchmark(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [(str(i), [i, i + 1]) for i in range(20)], shards=4)
            result = benchmark(root, sid, workers=3, repeats=2)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["keys"], 20)
            for mode in ("indexed_exact", "sharded_exact", "sharded_bloom"):
                metrics = result["modes"][mode]
                self.assertEqual(metrics["requests"], 40)
                self.assertGreater(metrics["throughput_rps"], 0)
                self.assertLessEqual(metrics["p50_ms"], metrics["p95_ms"])
                self.assertLessEqual(metrics["p95_ms"], metrics["p99_ms"])

    def test_rejects_invalid_worker_count(self):
        with tempfile.TemporaryDirectory() as root:
            sid = build_snapshot(root, [("x", [1])])
            with self.assertRaises(ValueError):
                benchmark(root, sid, workers=0)


if __name__ == "__main__":
    unittest.main()
