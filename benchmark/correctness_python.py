import csv
import sys

import numpy as np
from sambo import minimize

DEFAULT_SEEDS = range(1, 6)

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
    ("hartmann6", hartmann6, [(0.0, 1.0)] * 6, -3.322368011415515, -2.8),
    ("rotated_rastrigin5", rotated_rastrigin, [(-5.12, 5.12)] * 5, 0.0, 12.0),
    ("rotated_rosenbrock5", rotated_rosenbrock, [(-3.0, 3.0)] * 5, 0.0, 12.0),
)
ALGORITHMS = (
    ("SCE-UA", "sceua", 1000),
    ("SMBO", "smbo", 300),
    ("SHGO", "shgo", 1000),
)


def main(seeds=DEFAULT_SEEDS):
    assert abs(hartmann6(HARTMANN6_MINIMIZER) + 3.322368011415515) < 1e-6
    assert rotated_rastrigin(SHIFT) == 0
    assert rotated_rosenbrock(SHIFT) == 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "runtime",
            "problem",
            "algorithm",
            "seed",
            "budget",
            "evaluations",
            "minimum",
            "optimum",
            "target",
            "success",
        )
    )
    for problem, objective, bounds, optimum, target in CASES:
        for algorithm, method, budget in ALGORITHMS:
            for seed in seeds:
                result = minimize(
                    objective,
                    bounds=bounds,
                    method=method,
                    max_iter=budget,
                    rng=seed,
                )
                value = float(result.fun)
                writer.writerow(
                    (
                        "Python",
                        problem,
                        algorithm,
                        seed,
                        budget,
                        len(result.funv),
                        f"{value:.17g}",
                        optimum,
                        target,
                        str(value <= target).lower(),
                    )
                )


if __name__ == "__main__":
    main()
