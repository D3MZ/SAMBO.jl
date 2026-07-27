import gc
import statistics
import time
import tracemalloc

import sambo
from sambo import minimize

BUDGET_SETTING = 40
SEED = 42
REPEATS = 10


def run(method):
    objective_calls = 0

    def rosenbrock(x):
        nonlocal objective_calls
        objective_calls += 1
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = minimize(
        rosenbrock,
        bounds=[(-2.0, 2.0), (-1.0, 3.0)],
        method=method,
        max_iter=BUDGET_SETTING,
        rng=SEED,
    )
    return result, objective_calls


def measure(method):
    # Warm imports and library-level caches. Python timings do not include tracemalloc.
    run(method)
    elapsed = []
    result = None
    objective_calls = 0
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter_ns()
        result, objective_calls = run(method)
        elapsed.append(time.perf_counter_ns() - start)

    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    memory_result, memory_calls = run(method)
    after = tracemalloc.take_snapshot()
    _, peak_bytes = tracemalloc.get_traced_memory()
    positive_blocks = sum(
        stat.count_diff
        for stat in after.compare_to(before, "lineno")
        if stat.count_diff > 0
    )
    tracemalloc.stop()

    assert memory_calls == objective_calls
    assert len(memory_result.funv) == len(result.funv)
    return statistics.median(elapsed) / 1e6, peak_bytes, positive_blocks, result, objective_calls


def main():
    version = getattr(sambo, "__version__", "unknown")
    print(
        "python_sambo_version,algorithm,time_ms,peak_bytes,positive_tracemalloc_blocks,"
        "minimum,objective_calls,reported_observations,max_iter_setting,timing_repeats,seed"
    )
    for label, method in (
        ("SCE-UA", "sceua"),
        ("SMBO", "smbo"),
        ("SHGO", "shgo"),
    ):
        try:
            elapsed, peak, blocks, result, calls = measure(method)
            print(
                f"{version},{label},{elapsed:.6f},{peak},{blocks},{result.fun:.12g},"
                f"{calls},{len(result.funv)},{BUDGET_SETTING},{REPEATS},{SEED}"
            )
        except Exception as error:
            print(f"{version},{label},ERROR: {error}")


if __name__ == "__main__":
    main()
