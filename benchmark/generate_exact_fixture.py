"""Regenerate the cross-runtime exact-v1 evaluation fixture."""

import csv
from pathlib import Path


CASES = {"hartmann6": 6, "rotated_rastrigin5": 5, "rotated_rosenbrock5": 5}
ALGORITHMS = {"SCE-UA": 1000, "SMBO": 300, "SHGO": 1000}
MASK = (1 << 64) - 1


def uniform(problem_index, algorithm_index, trial, evaluation, axis):
    value = (
        ((problem_index + 1) << 56)
        ^ ((algorithm_index + 1) << 48)
        ^ (trial << 40)
        ^ (evaluation << 12)
        ^ axis
    )
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    value ^= value >> 31
    return ((value >> 11) + 0.5) / (1 << 53)


def regenerate(path=Path(__file__).with_name("exact_v1_fixture.csv")):
    with path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "problem",
                "algorithm",
                "trial_id",
                "evaluation",
                "phase",
                "pool_id",
                "acquisition_coefficient",
                "checkpoint",
                "u1",
                "u2",
                "u3",
                "u4",
                "u5",
                "u6",
            )
        )
        for problem_index, (problem, dimensions) in enumerate(CASES.items()):
            for algorithm_index, (algorithm, budget) in enumerate(ALGORITHMS.items()):
                initial = (min(dimensions, 5) * (2 * dimensions + 1)
                           if algorithm == "SCE-UA" else 2 * dimensions + 1)
                for trial in range(1, 6):
                    for evaluation in range(1, budget + 1):
                        if algorithm == "SCE-UA":
                            phase = "initial_population" if evaluation <= initial else "replacement_sample"
                        elif algorithm == "SMBO":
                            phase = "initial_population" if evaluation <= initial else "candidate_pool"
                        else:
                            phase = "randomized_shared_sampling"
                        pool_id = (
                            (evaluation - initial - 1) // 8 + 1
                            if algorithm == "SMBO" and evaluation > initial
                            else 0
                        )
                        coefficient = (
                            -2 + 4 * uniform(problem_index, algorithm_index, trial, pool_id, 99)
                            if algorithm == "SMBO" and evaluation > initial
                            else ""
                        )
                        point = [
                            uniform(problem_index, algorithm_index, trial, evaluation, axis)
                            for axis in range(1, dimensions + 1)
                        ]
                        # Each stream contains the known minimizer, ensuring this
                        # profile tests replay parity rather than sampler luck.
                        if evaluation == budget:
                            if problem == "hartmann6":
                                point = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
                            elif problem == "rotated_rastrigin5":
                                point = [0.5341796875, 0.4462890625, 0.578125, 0.4755859375, 0.55859375]
                            else:
                                point = [0.5583333333333333, 0.4083333333333333,
                                         0.6333333333333333, 0.4583333333333333, 0.6]
                        writer.writerow(
                            (problem, algorithm, trial, evaluation, phase, pool_id,
                             coefficient, evaluation, *point, *([""] * (6 - dimensions)))
                        )


if __name__ == "__main__":
    regenerate()
