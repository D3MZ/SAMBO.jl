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
    def test_python_algorithm_profiles_are_explicit(self):
        sce = correctness.matched_method_kwargs("SCE-UA", 5, 1000)
        self.assertEqual(
            sce,
            {
                "complex_size": 2,
                "n_complexes": 6,
                "n_iter_no_change": 30,
                "tol": 1e-6,
                "x_tol": 1e-6,
            },
        )
        smbo = correctness.matched_method_kwargs("SMBO", 5, 300)
        self.assertEqual(smbo["n_init"], 280)
        self.assertEqual(smbo["n_candidates"], 1)
        self.assertEqual(
            correctness.matched_method_kwargs("SMBO", 5, 100)["n_init"],
            80,
        )
        shgo = correctness.matched_method_kwargs("SHGO", 5, 1000)
        self.assertEqual(shgo["n_init"], 80)
        self.assertEqual(shgo["sampling_method"], "halton")

    def test_runtime_algorithms_share_canonical_solver_coordinates(self):
        self.assertEqual(
            {name: budget for name, _, budget in correctness.ALGORITHMS},
            {"SCE-UA": 1000, "SMBO": 100, "SHGO": 1000},
        )
        for _, _, _, bounds, _ in correctness.CASES:
            self.assertTrue(all(bound == (0.0, 1.0) for bound in bounds))
        objective = correctness.canonical_objective(
            lambda point: float(np.sum(point)),
            [(-2.0, 2.0), (10.0, 20.0)],
        )
        self.assertEqual(objective([0.25, 0.5]), 14.0)

    def test_six_distinct_orthogonal_challenge_rotations(self):
        rotations = [
            correctness.challenge_rotation(5, rotation_id)
            for rotation_id in correctness.ROTATION_IDS
        ]
        self.assertEqual(len(rotations), 6)
        for rotation in rotations:
            np.testing.assert_allclose(
                rotation @ rotation.T,
                np.eye(5),
                atol=1e-14,
            )
        for left, right in zip(rotations, rotations[1:]):
            self.assertFalse(np.allclose(left, right))

    def test_all_native_algorithms_use_the_full_trial_matrix(self):
        trials = range(1, 11)
        for algorithm, _, _ in correctness.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                self.assertEqual(
                    correctness.benchmark_trials(algorithm, trials),
                    tuple(range(1, 11)),
                )

    def test_shgo_halton_trials_are_seeded_realizations(self):
        objective = lambda point: float(np.sum(np.asarray(point) ** 2))
        first = correctness.minimize(
            objective,
            bounds=[(-1.0, 1.0)] * 2,
            method="shgo",
            sampling_method="halton",
            max_iter=20,
            rng=1,
        )
        second = correctness.minimize(
            objective,
            bounds=[(-1.0, 1.0)] * 2,
            method="shgo",
            sampling_method="halton",
            max_iter=20,
            rng=2,
        )
        self.assertFalse(np.array_equal(first.xv, second.xv))

    def test_shared_initial_design_capabilities_are_explicit(self):
        self.assertEqual(
            correctness.shared_design_capability("SCE-UA"),
            "injected-counted-lhs",
        )
        self.assertEqual(
            correctness.shared_design_capability("SMBO"),
            "injected-counted-lhs",
        )
        self.assertEqual(
            correctness.shared_design_capability("SHGO"),
            "not-supported-cross-runtime",
        )
        first = correctness.shared_initial_design(5, 12, 2, 3)
        second = correctness.shared_initial_design(5, 12, 2, 3)
        different = correctness.shared_initial_design(5, 12, 2, 4)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(np.all((0 <= first) & (first <= 1)))

if __name__ == "__main__":
    unittest.main()
