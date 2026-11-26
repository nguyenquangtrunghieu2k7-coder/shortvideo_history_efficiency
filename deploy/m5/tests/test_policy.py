import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.integration import m5_selector as s
from src.integration.m5_policy import PolicyStore, atomic_write

class SelectorTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual({k for (m,k),ok in zip(s.ACTIONS,s.grid_mask("original")) if ok},{0,1,5,10,20,50,100,2048})
        self.assertEqual(len(s.ACTIONS),26)
    def test_zero_and_monotonic(self):
        r=s.resource_table([0,10,2048]);q=np.random.default_rng(42).normal(size=r.shape[:2])
        prev=None
        for w in s.PROFILES:
            p=s.preference(w,max_read_tokens=100)
            ix=s.choose(q,r,p)
            self.assertTrue(np.all(r[np.arange(3),ix,0]<=100))
            cost=s.cost_table(r,p)[np.arange(3),ix]
            if prev is not None:self.assertTrue(np.all(cost>=prev-1e-8))
            prev=cost
        ix=s.choose(q,r,s.preference(0,max_read_tokens=0))
        self.assertTrue(np.all(r[np.arange(3),ix,0]==0))
    def test_infeasible(self):
        r=s.resource_table([2048])
        with self.assertRaises(ValueError):s.choose(np.zeros(r.shape[:2]),r,s.preference(max_logical_bytes=0))
    def test_bad_weights(self):
        with self.assertRaises(ValueError):s.preference(float("nan"))
        with self.assertRaises(ValueError):s.preference(-1)
    def test_policy(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"business.json"
            atomic_write(path,{"revision":1,"preferences":{"quality_weight":.9}})
            store=PolicyStore(path)
            self.assertEqual(store.get()[0]["revision"],1)
            path.write_text("{")
            self.assertEqual(store.get()[0]["revision"],1)
            self.assertIsNotNone(store.get()[1])
            atomic_write(path,{"revision":2,"preferences":{"quality_weight":.1}})
            self.assertEqual(store.get()[0]["revision"],2)
            atomic_write(path,{"revision":1,"preferences":{}})
            self.assertEqual(store.get()[0]["revision"],2)
if __name__=="__main__":unittest.main()
