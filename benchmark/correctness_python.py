import csv
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import sambo
from sambo import minimize
from scipy.stats.qmc import Halton

DEFAULT_TRIALS = range(1, 11)
ROTATION_IDS = range(1, 7)
SOURCE_COMMIT = os.environ.get("GITHUB_SHA")
if not SOURCE_COMMIT:
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        SOURCE_COMMIT = subprocess.check_output(
            ("git", "-C", root, "rev-parse", "HEAD"),
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            (
                "git",
                "-C",
                root,
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ),
            text=True,
        ).strip()
        if dirty:
            SOURCE_COMMIT += "-dirty"
    except (OSError, subprocess.CalledProcessError):
        SOURCE_COMMIT = "unknown"

SHIFT = np.array([0.35, -0.55, 0.8, -0.25, 0.6])
HARTMANN6_MINIMIZER = np.array(
    [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
)

# These are nonseparable rotated-box robustness cases, not box-invariance tests.

def hartmann6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    p = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )
    return -np.sum(alpha * np.exp(-np.sum(a * (x - p) ** 2, axis=1)))


def challenge_rotation(dimensions, rotation_id):
    rotation = np.eye(dimensions)
    offset = (rotation_id - 1) % (dimensions - 1) + 1
    for axis in range(dimensions):
        other = (axis + offset) % dimensions
        angle = ((rotation_id * (2 * (axis + 1) + 1)) % 19 + 1) * np.pi / 37
        cosine = np.cos(angle)
        sine = np.sin(angle)
        left = rotation[axis].copy()
        right = rotation[other].copy()
        rotation[axis] = cosine * left - sine * right
        rotation[other] = sine * left + cosine * right
    return rotation


def transformed(x, rotation):
    return rotation @ (np.asarray(x) - SHIFT)


def rotated_rastrigin(x, rotation):
    y = transformed(x, rotation)
    return 10 * len(y) + np.sum(y**2 - 10 * np.cos(2 * np.pi * y))


def rotated_rosenbrock(x, rotation):
    y = transformed(x, rotation) + 1
    return np.sum(100 * (y[1:] - y[:-1] ** 2) ** 2 + (1 - y[:-1]) ** 2)


def oriented_hartmann6(x, rotation_id):
    oriented = np.empty(6)
    for axis in range(6):
        source = (axis + rotation_id - 1) % 6
        reflected = (axis + 1 + rotation_id) % 2 == 0
        oriented[axis] = 1 - x[source] if reflected else x[source]
    return hartmann6(oriented)


def hartmann_preimage(point, rotation_id):
    preimage = np.empty(6)
    for axis in range(6):
        source = (axis + rotation_id - 1) % 6
        reflected = (axis + 1 + rotation_id) % 2 == 0
        preimage[source] = 1 - point[axis] if reflected else point[axis]
    return preimage


def canonical_objective(objective, physical_bounds):
    lower = np.array([bound[0] for bound in physical_bounds], dtype=float)
    widths = np.array(
        [bound[1] - bound[0] for bound in physical_bounds],
        dtype=float,
    )
    return lambda latent: objective(lower + np.asarray(latent) * widths)


def benchmark_cases():
    cases = []
    for rotation_id in ROTATION_IDS:
        rotation = challenge_rotation(5, rotation_id)
        rastrigin_bounds = [(-5.12, 5.12)] * 5
        rosenbrock_bounds = [(-3.0, 3.0)] * 5
        cases.extend(
            (
                (
                    "hartmann6",
                    rotation_id,
                    lambda x, rid=rotation_id: oriented_hartmann6(x, rid),
                    [(0.0, 1.0)] * 6,
                    -3.322368011415515,
                ),
                (
                    "rotated_rastrigin5",
                    rotation_id,
                    canonical_objective(
                        lambda x, matrix=rotation: rotated_rastrigin(x, matrix),
                        rastrigin_bounds,
                    ),
                    [(0.0, 1.0)] * 5,
                    0.0,
                ),
                (
                    "rotated_rosenbrock5",
                    rotation_id,
                    canonical_objective(
                        lambda x, matrix=rotation: rotated_rosenbrock(x, matrix),
                        rosenbrock_bounds,
                    ),
                    [(0.0, 1.0)] * 5,
                    0.0,
                ),
            )
        )
    return tuple(cases)


CASES = benchmark_cases()
ALGORITHMS = (
    ("SCE-UA", "sceua", 1000),
    ("SMBO", "smbo", 100),
    ("SHGO", "shgo", 1000),
)


def shared_initial_design(dimensions, count, rotation_id, trial_id):
    design = np.empty((count, dimensions))
    for axis in range(dimensions):
        multiplier = 2 * (axis + 1) + trial_id + rotation_id
        while np.gcd(multiplier, count) != 1:
            multiplier += 1
        offset = (
            trial_id * (11 + 2 * axis)
            + rotation_id * (17 + axis)
        ) % count
        for column in range(count):
            stratum = (multiplier * column + offset) % count
            jitter = (
                trial_id * 101
                + rotation_id * 211
                + (axis + 1) * 307
                + (column + 1) * 401
            ) % 997 / 997
            design[column, axis] = (stratum + jitter) / count
    return design


def shared_halton_shift(dimensions, rotation_id, trial_id):
    return np.array(
        [
            (
                trial_id * 127
                + rotation_id * 283
                + (axis + 1) * 419
            ) % 997 / 997
            for axis in range(dimensions)
        ]
    )


class SharedShiftedHalton:
    def __init__(self, dimensions, rotation_id, trial_id):
        self.sampler = Halton(dimensions, scramble=False)
        self.sampler.fast_forward(1)
        self.shift = shared_halton_shift(
            dimensions,
            rotation_id,
            trial_id,
        )

    def __call__(self, count, dimensions):
        if dimensions != len(self.shift):
            raise ValueError("SHGO sampler dimension changed")
        return (self.sampler.random(count) + self.shift) % 1


def coordinate_hash(coordinates):
    values = np.asarray(coordinates, dtype=np.float64, order="C").ravel(
        order="C"
    )
    canonical = (",".join(f"{value:.12f}" for value in values) + "\n").encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def shared_design_capability(algorithm):
    return (
        "injected-counted-lhs"
        if algorithm in ("SCE-UA", "SMBO")
        else "not-applicable"
    )


def sampling_stream_capability(algorithm):
    return (
        "shared-shifted-halton"
        if algorithm == "SHGO"
        else "not-applicable"
    )


def shared_sampling_stream(dimensions, budget, rotation_id, trial_id):
    sampler = SharedShiftedHalton(dimensions, rotation_id, trial_id)
    return sampler(budget, dimensions)


def benchmark_trials(algorithm, trials):
    return tuple(trials)


def matched_method_kwargs(algorithm, dimensions, budget):
    if algorithm == "SCE-UA":
        return {
            "complex_size": 2,
            "n_complexes": min(
                max(2, budget // 2 - 1),
                max(5, int(3 * np.log2(dimensions))),
            ),
            "n_iter_no_change": 30,
            "tol": 1e-6,
            "x_tol": 1e-6,
        }
    if algorithm == "SMBO":
        return {
            "n_init": min(
                max(1, budget - 20),
                int(40 * dimensions * max(1, np.log2(dimensions))),
            ),
            "n_candidates": 1,
            "n_iter_no_change": 5,
            "tol": 1e-6,
        }
    if algorithm == "SHGO":
        return {
            "sampling_method": "halton",
            "n_init": min(budget // 2, max(80, 2**dimensions + 1)),
            "n_iter_no_change": 30,
            "tol": 1e-6,
        }
    raise ValueError(f"unknown matched algorithm: {algorithm}")


def normalized_gap(value, optimum):
    return max(abs(value - optimum) / max(1, abs(optimum)), 1e-15)


def main(trials=DEFAULT_TRIALS):
    assert abs(hartmann6(HARTMANN6_MINIMIZER) + 3.322368011415515) < 1e-6
    for rotation_id in ROTATION_IDS:
        assert (
            abs(
                oriented_hartmann6(
                    hartmann_preimage(HARTMANN6_MINIMIZER, rotation_id),
                    rotation_id,
                )
                + 3.322368011415515
            )
            < 1e-6
        )
        rotation = challenge_rotation(5, rotation_id)
        assert rotated_rastrigin(SHIFT, rotation) == 0
        assert rotated_rosenbrock(SHIFT, rotation) == 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "runtime",
            "runtime_version",
            "source_commit",
            "python_sambo_version",
            "problem",
            "algorithm",
            "rotation_id",
            "trial_id",
            "configuration_hash",
            "initial_design_hash",
            "initial_design_capability",
            "sampling_stream_hash",
            "sampling_stream_capability",
            "budget",
            "evaluation",
            "evaluations",
            "iteration",
            "best_value",
            "normalized_gap",
            "minimum",
            "optimum",
            "quality_threshold",
            "feasible",
            "duplicate",
            "retcode",
        )
    )
    for problem, rotation_id, objective, bounds, optimum in CASES:
        for algorithm, method, budget in ALGORITHMS:
            for trial_id in benchmark_trials(algorithm, trials):
                design_capability = shared_design_capability(algorithm)
                supports_shared_design = (
                    design_capability == "injected-counted-lhs"
                )
                sampling_capability = sampling_stream_capability(algorithm)
                evaluations = 0
                best = float("inf")

                def counted_objective(point):
                    nonlocal evaluations, best
                    value = float(objective(point))
                    evaluations += 1
                    best = min(best, value)
                    return value

                method_kwargs = matched_method_kwargs(
                    algorithm,
                    len(bounds),
                    budget,
                )
                if algorithm == "SHGO":
                    method_kwargs["sampling_method"] = SharedShiftedHalton(
                        len(bounds),
                        rotation_id,
                        trial_id,
                    )
                initial_count = (
                    method_kwargs["n_complexes"]
                    * method_kwargs["complex_size"]
                    if algorithm == "SCE-UA"
                    else method_kwargs.get("n_init", 0)
                )
                initial_design = (
                    shared_initial_design(
                        len(bounds),
                        initial_count,
                        rotation_id,
                        trial_id,
                    )
                    if supports_shared_design
                    else None
                )
                initial_kwargs = (
                    {"x0": initial_design}
                    if initial_design is not None
                    else {}
                )
                result = minimize(
                    counted_objective,
                    bounds=bounds,
                    method=method,
                    max_iter=budget,
                    rng=trial_id,
                    **initial_kwargs,
                    **method_kwargs,
                )
                if evaluations > budget:
                    raise RuntimeError(
                        "Python objective calls exceeded the requested budget"
                    )
                configuration_hash = (
                    f"python-sambo-1.25.2-matched-v6:{algorithm}:{budget}:"
                    "6-rotations"
                )
                design_hash = (
                    coordinate_hash(initial_design)
                    if supports_shared_design
                    else "none"
                )
                sampling_hash = (
                    coordinate_hash(
                        shared_sampling_stream(
                            len(bounds),
                            budget,
                            rotation_id,
                            trial_id,
                        )
                    )
                    if algorithm == "SHGO"
                    else "none"
                )
                writer.writerow(
                    (
                        "Python",
                        platform.python_version(),
                        SOURCE_COMMIT,
                        getattr(sambo, "__version__", "unknown"),
                        problem,
                        algorithm,
                        rotation_id,
                        trial_id,
                        configuration_hash,
                        design_hash,
                        design_capability,
                        sampling_hash,
                        sampling_capability,
                        budget,
                        evaluations,
                        evaluations,
                        evaluations,
                        f"{best:.17g}",
                        f"{normalized_gap(best, optimum):.17g}",
                        f"{best:.17g}",
                        optimum,
                        0.25,
                        "true",
                        "false",
                        getattr(result, "message", "completed"),
                    )
                )


if __name__ == "__main__":
    main()
