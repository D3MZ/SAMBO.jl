import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("correctness_python.py")
SPEC = importlib.util.spec_from_file_location("correctness_python", MODULE_PATH)
correctness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(correctness)
GENERATOR_PATH = Path(__file__).with_name("generate_exact_fixture.py")
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_exact_fixture", GENERATOR_PATH
)
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC.loader is not None
GENERATOR_SPEC.loader.exec_module(generator)


class PythonCapabilityTests(unittest.TestCase):
    def test_deterministic_shgo_is_one_realization(self):
        trials = range(1, 6)
        self.assertEqual(correctness.benchmark_trials("SHGO", trials), (1,))
        self.assertEqual(
            correctness.benchmark_trials("SMBO", trials),
            (1, 2, 3, 4, 5),
        )

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

    def test_exact_fixture_is_reproducible_and_carries_shared_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "fixture.csv"
            generator.regenerate(regenerated)
            self.assertEqual(
                regenerated.read_bytes(),
                Path(correctness.FIXTURE_PATH).read_bytes(),
            )
        streams = correctness.exact_fixture()
        self.assertEqual(len(streams), 45)
        sce = streams[("hartmann6", "SCE-UA", 1)]
        self.assertIn("replacement_sample", {item["phase"] for item in sce})
        smbo = streams[("hartmann6", "SMBO", 1)]
        pools = [item for item in smbo if item["phase"] == "candidate_pool"]
        self.assertTrue(all(item["pool_id"] > 0 for item in pools))
        self.assertTrue(
            all(item["acquisition_coefficient"] is not None for item in pools)
        )
        self.assertTrue(
            all(
                left["acquisition_coefficient"]
                == right["acquisition_coefficient"]
                for left, right in zip(pools, pools[1:])
                if left["pool_id"] == right["pool_id"]
            )
        )
        shgo_trials = {
            key[2]
            for key in streams
            if key[:2] == ("hartmann6", "SHGO")
        }
        self.assertEqual(shgo_trials, {1, 2, 3, 4, 5})
        self.assertTrue(correctness.fixture_hash().startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
