"""Lightweight M1 retrieval correctness tests; no KuaiRand artifacts required."""
import unittest

from src.integration.e2e_v4_bloom import build_shards, retrieve_histories, select_histories
from src.policy.adaptive_v4_policy import K_VALUES


class TestM1Retrieval(unittest.TestCase):
    def test_bloom_exact_equivalence(self):
        ids = [101, 102, 103, 104, 105, 106]
        histories = [[1, 2, 3], [], [5], [7, 7, 8, 9], [11, 12], [42]]
        stores, filters, _ = build_shards(ids, histories, 4, 12, 5, 2)
        exact, exact_stats, _ = retrieve_histories(ids, stores, filters, False)
        bloom, bloom_stats, _ = retrieve_histories(ids, stores, filters, True)
        self.assertEqual(histories, exact)
        self.assertEqual(exact, bloom)
        self.assertEqual(bloom_stats["exact_hits"], len(ids))
        self.assertEqual(exact_stats["exact_hits"], len(ids))
        self.assertEqual(bloom_stats["bloom_rejections"] + bloom_stats["bloom_maybes"], len(ids) * 4)

    def test_history_truncation(self):
        histories = [[1, 2, 3, 4], [8, 9], []]
        actions = [K_VALUES.index(1), K_VALUES.index("Full"), K_VALUES.index(0)]
        self.assertEqual(select_histories(histories, actions), [[4], [8, 9], []])


if __name__ == "__main__":
    unittest.main()
