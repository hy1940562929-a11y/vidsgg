"""Synthetic-only exact readout checks; no native cache/model execution."""
import time
import unittest
import numpy as np
from decode import project, component, hull_eligible, _cdf, _costs, _interval


def brute(z, origin=0):
    if len(z) < 2:
        return None
    f = _cdf(z)
    a, b = np.triu_indices(len(z), k=1)
    costs = _costs(f, a, b-a+1)
    i = np.lexsort((b, a, costs))[0]
    return _interval(a[i], b[i], origin)


class DecodeTests(unittest.TestCase):
    def test_exhaustive_random(self):
        rng = np.random.default_rng(95)
        for n in range(2, 33):
            for _ in range(30):
                z = rng.normal(0, 4, n).astype(np.float32)
                self.assertEqual(project(z, 43), brute(z, 43))

    def test_constant_and_symmetric(self):
        for n in range(2, 30):
            for value in (-1000., -1., 0., 2., 1000.):
                z = np.full(n, value, np.float32)
                self.assertEqual(project(z), brute(z))
            z = np.arange(n, dtype=np.float32)
            z = -abs(z-(n-1)/2)*2
            self.assertEqual(project(z), brute(z))
            self.assertEqual(project(-z), brute(-z))

    def test_extreme_ties(self):
        for z in ([1000, -1000, -1000, 1000],
                  [-1000, 1000, -1000], [1000, -1000, 1000],
                  [1000, 1000, -1000, -1000, 1000, 1000]):
            z = np.asarray(z, np.float32)
            self.assertEqual(project(z), brute(z))

    def test_component_selection(self):
        self.assertEqual(component([1, 1, -1, 3, 3], 10)["first_index"], 3)
        self.assertEqual(component([2, 2, -1, 2, 2])["first_index"], 0)
        self.assertIsNone(component([1, -1, 1]))
        self.assertTrue(hull_eligible([1, -1, 1]))
        self.assertFalse(hull_eligible([-1, 1, -1]))
        self.assertIsNone(project([1]))
        self.assertIsNone(project([]))
        self.assertEqual(project([1, 1], 10)["frame_length"], 5)


if __name__ == "__main__":
    import sys
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecodeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    rng = np.random.default_rng(95)
    print("longdouble epsilon:", np.finfo(np.longdouble).eps)
    for n in (175, 350, 675):
        z = rng.normal(0, 2, n).astype(np.float32)
        start = time.perf_counter()
        for _ in range(3):
            project(z)
        print("N=%d seconds/slot=%.6f" % (n, (time.perf_counter()-start)/3))
