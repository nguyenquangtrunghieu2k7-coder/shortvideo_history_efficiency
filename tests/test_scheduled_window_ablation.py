import unittest
import pandas as pd
from src.integration.scheduled_window_ablation import (
    DAY, HOUR, last_refresh, scheduled_window, evaluate_schedule,
)


class ScheduledWindowTests(unittest.TestCase):
    def test_refresh_boundaries(self):
        anchor = (2 - 7) * HOUR
        self.assertEqual(last_refresh(anchor + 24*HOUR, 24), anchor + 24*HOUR)
        self.assertEqual(last_refresh(anchor + 23*HOUR, 24), anchor)
        self.assertEqual(last_refresh(anchor + 23*HOUR, 6), anchor + 18*HOUR)

    def test_strict_temporal_history(self):
        times = [0, HOUR, 2*HOUR, 3*HOUR]
        self.assertEqual(scheduled_window(times, 2*HOUR, 2*HOUR), (0, 2))
        self.assertEqual(scheduled_window(times, 3*HOUR, 2*HOUR), (1, 3))

    def test_fixed_evaluation_events_all_scenarios(self):
        events = pd.DataFrame({
            "evaluation_id": [1, 2, 3], "user_id": [7, 7, 7],
            "time_ms": [DAY+HOUR, DAY+8*HOUR, DAY+15*HOUR],
        })
        result = evaluate_schedule(events, {7: [DAY-3*HOUR, DAY, DAY+2*HOUR]},
                                   strides=[24, 12, 6, 3])
        self.assertEqual(len(result), 4)
        self.assertTrue(all(x["n_evaluation_events"] == 3 for x in result))
        self.assertEqual([x["nominal_overlap_fraction"] for x in result],
                         [0, .5, .75, .875])
        self.assertTrue(all(x["new_interactions_since_refresh_sum"] >= 0 for x in result))

    def test_duplicate_ids_rejected(self):
        events = pd.DataFrame({"evaluation_id": [1, 1],
                               "user_id": [7, 7], "time_ms": [DAY, DAY+HOUR]})
        with self.assertRaises(ValueError):
            evaluate_schedule(events, {}, strides=[24])


if __name__ == "__main__":
    unittest.main()
