import csv
import os
import platform
import subprocess
import sys

import numpy as np
import sambo
from sambo import minimize

DEFAULT_TRIALS = range(1, 6)
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

ROTATION = np.array(
    [
        [-0.6593804733957869, 0.40611875581020845, 0.16266300278010432, 0.5752843417117289, 0.20705946293608232],
        [-0.39562828403747224, -0.5245384304015962, 0.2924634597652826, 0.08246149657099422, -0.6899296501722388],
        [-0.19781414201873612, -0.43686232517528023, -0.8254056738772803, 0.26904408514342243, 0.12783437659867097],
        [-0.5934424260562083, 0.17345445925725733, -0.1921454429053599, -0.7612651988361476, 0.03598698837021451],
        [-0.13187609467915742, -0.5822300667409905, 0.4120576106936707, -0.10167893122409566, 0.6807986232723334],
    ]
)
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


def transformed(x):
    return ROTATION @ (np.asarray(x) - SHIFT)


def rotated_rastrigin(x):
    y = transformed(x)
    return 10 * len(y) + np.sum(y**2 - 10 * np.cos(2 * np.pi * y))


def rotated_rosenbrock(x):
    y = transformed(x) + 1
    return np.sum(100 * (y[1:] - y[:-1] ** 2) ** 2 + (1 - y[:-1]) ** 2)


CASES = (
    (
        "hartmann6",
        hartmann6,
        [(0.0, 1.0)] * 6,
        -3.322368011415515,
        -2.8,
        -2.0,
    ),
    (
        "rotated_rastrigin5",
        rotated_rastrigin,
        [(-5.12, 5.12)] * 5,
        0.0,
        12.0,
        50.0,
    ),
    (
        "rotated_rosenbrock5",
        rotated_rosenbrock,
        [(-3.0, 3.0)] * 5,
        0.0,
        12.0,
        200.0,
    ),
)
ALGORITHMS = (
    ("SCE-UA", "sceua", 1000),
    ("SMBO", "smbo", 300),
    ("SHGO", "shgo", 1000),
)


def shared_initial_point(problem, bounds, trial_id):
    latent = np.array(
        [
            (trial_id * (2 * (axis + 1) + 1) % 17) / 17
            for axis in range(len(bounds))
        ]
    )
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    return lower + latent * (upper - lower)


def initial_design_hash(problem, trial_id):
    return f"shared-design-v1:{problem}:{trial_id}"


def shared_design_capability(algorithm):
    return (
        "injected-x0-y0"
        if algorithm == "SMBO"
        else "not-supported-cross-runtime"
    )


def benchmark_trials(algorithm, trials):
    trials = tuple(trials)
    return trials[:1] if algorithm == "SHGO" else trials


def normalized_gap(value, optimum):
    return max(abs(value - optimum) / max(1, abs(optimum)), 1e-15)


def main(trials=DEFAULT_TRIALS):
    assert abs(hartmann6(HARTMANN6_MINIMIZER) + 3.322368011415515) < 1e-6
    assert rotated_rastrigin(SHIFT) == 0
    assert rotated_rosenbrock(SHIFT) == 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "runtime",
            "runtime_version",
            "source_commit",
            "python_sambo_version",
            "problem",
            "algorithm",
            "trial_id",
            "configuration_hash",
            "initial_design_hash",
            "initial_design_capability",
            "budget",
            "evaluation",
            "evaluations",
            "iteration",
            "best_value",
            "normalized_gap",
            "minimum",
            "optimum",
            "target",
            "quality_target",
            "required_hit_rate",
            "noninferiority_margin",
            "feasible",
            "duplicate",
            "retcode",
            "success",
        )
    )
    for problem, objective, bounds, optimum, target, quality_target in CASES:
        for algorithm, method, budget in ALGORITHMS:
            for trial_id in benchmark_trials(algorithm, trials):
                initial_point = shared_initial_point(
                    problem,
                    bounds,
                    trial_id,
                )
                initial_value = float(objective(initial_point))
                design_capability = shared_design_capability(algorithm)
                supports_shared_design = design_capability == "injected-x0-y0"
                evaluation_trace = []

                def counted_objective(point):
                    value = float(objective(point))
                    evaluation_trace.append(
                        (np.asarray(point, dtype=float).copy(), value)
                    )
                    return value

                initial_kwargs = (
                    {"x0": [initial_point], "y0": [initial_value]}
                    if supports_shared_design
                    else {}
                )
                result = minimize(
                    counted_objective,
                    bounds=bounds,
                    method=method,
                    max_iter=budget,
                    rng=trial_id,
                    **initial_kwargs,
                )
                if len(evaluation_trace) > budget:
                    raise RuntimeError(
                        "Python objective calls exceeded the requested budget"
                    )
                configuration_hash = (
                    f"native-default-v1:{algorithm}:{budget}"
                )
                design_hash = (
                    initial_design_hash(problem, trial_id)
                    if supports_shared_design
                    else "none"
                )
                best = float("inf")
                seen = set()
                for evaluation, (point, value) in enumerate(
                    evaluation_trace,
                    start=1,
                ):
                    best = min(best, value)
                    point_key = tuple(point.tolist())
                    duplicate = point_key in seen
                    seen.add(point_key)
                    result_code = (
                        getattr(result, "message", "completed")
                        if evaluation == len(evaluation_trace)
                        else "running"
                    )
                    writer.writerow(
                        (
                            "Python",
                            platform.python_version(),
                            SOURCE_COMMIT,
                            getattr(sambo, "__version__", "unknown"),
                            problem,
                            algorithm,
                            trial_id,
                            configuration_hash,
                            design_hash,
                            design_capability,
                            budget,
                            evaluation,
                            evaluation,
                            evaluation,
                            f"{best:.17g}",
                            f"{normalized_gap(best, optimum):.17g}",
                            f"{best:.17g}",
                            optimum,
                            target,
                            quality_target,
                            0.8,
                            0.25,
                            "true",
                            str(duplicate).lower(),
                            result_code,
                            str(best <= target).lower(),
                        )
                    )


if __name__ == "__main__":
    main()
