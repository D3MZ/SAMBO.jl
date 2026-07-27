import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("correctness_python.py")
SPEC = importlib.util.spec_from_file_location("correctness_python", MODULE_PATH)
correctness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(correctness)


class PythonCapabilityTests(unittest.TestCase):
    def test_shared_initial_design_capabilities_are_explicit(self):
        bounds = [(0.0, 1.0)] * 2
        initial = np.array([0.25, 0.75])
        initial_value = float(np.sum(initial**2))
        for algorithm, method, _ in correctness.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = correctness.minimize(
                    lambda point: float(np.sum(np.asarray(point) ** 2)),
                    x0=[initial],
                    y0=[initial_value],
                    bounds=bounds,
                    method=method,
                    max_iter=6,
                    rng=1,
                )
                retained = any(
                    np.array_equal(np.asarray(point), initial)
                    for point in result.xv
                )
                capability = correctness.shared_design_capability(algorithm)
                self.assertEqual(
                    retained,
                    capability == "injected-x0-y0",
                )


if __name__ == "__main__":
    unittest.main()
