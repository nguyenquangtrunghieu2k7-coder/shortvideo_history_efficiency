import unittest
import pandas as pd
from src.integration.scheduled_window_v4_e2e import build_histories

class ScheduledV4Stage2Tests(unittest.TestCase):
    def test_window_exact_boundary_and_overlap(self):
        h = 3600000
        events = pd.DataFrame({"evaluation_id":[1,2],"user_id":[9,9],
                               "time_ms":[25*h,31*h]})
        records = {9:([0,1*h,2*h,6*h,24*h,25*h,30*h],
                       [1,2,3,4,5,6,7])}
        histories,metrics = build_histories(events,records,6,0,0)
        self.assertEqual(histories[0],[1,2,3,4])
        self.assertEqual(histories[1],[4,5,6])
        self.assertEqual(metrics["observed_user_refresh_jobs"],2)
        self.assertEqual(metrics["new_interactions_since_refresh"],2)

    def test_no_future_and_zero_history(self):
        h = 3600000
        events = pd.DataFrame({"evaluation_id":[1],"user_id":[9],
                               "time_ms":[h]})
        histories,_ = build_histories(events,{9:([h],[8])},24,0,0)
        self.assertEqual(histories,[[]])

if __name__=="__main__":
    unittest.main()
