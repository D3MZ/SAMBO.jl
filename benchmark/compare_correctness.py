import csv
import math
import statistics
import sys


CONFIGURATION_PREFIX = "python-sambo-1.25.2-matched-v7:"
MINIMUM_TRIALS = 10
REQUIRED_ROTATIONS = 6
REGRET_FLOOR = 1e-6
MATCHED_METADATA = (
    "budget",
    "optimum",
    "quality_threshold",
    "source_commit",
    "configuration_hash",
    "initial_design_hash",
    "initial_design_capability",
    "sampling_stream_hash",
    "sampling_stream_capability",
)
REQUIRED_COLUMNS = (
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
    "evaluations",
    "minimum",
    "optimum",
    "quality_threshold",
    "retcode",
)


def _fail(message):
    raise SystemExit(message)


def _integer(row, field, key):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid {field} for {key}") from error


def _finite_float(row, field, key):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid {field} for {key}") from error
    if not math.isfinite(value):
        raise SystemExit(f"nonfinite {field} for {key}")
    return value


def read_results(path):
    with open(path, newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            _fail(f"missing CSV header in {path}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            _fail(f"missing required CSV columns in {path}: {missing}")
        rows = list(reader)
    if not rows:
        _fail(f"no correctness result rows in {path}")

    results = {}
    for row in rows:
        try:
            key = (
                row["problem"],
                row["algorithm"],
                _integer(row, "rotation_id", None),
                _integer(row, "trial_id", None),
            )
        except KeyError as error:
            _fail(f"missing comparison key field: {error.args[0]}")
        if key in results:
            _fail(f"duplicate comparison key: {key}")
        budget = _integer(row, "budget", key)
        evaluations = _integer(row, "evaluations", key)
        budget > 0 or _fail(f"invalid budget for {key}")
        0 < evaluations <= budget or _fail(
            f"evaluations outside budget for {key}: "
            f"evaluations={evaluations}, budget={budget}"
        )
        for field in ("minimum", "optimum"):
            _finite_float(row, field, key)
        for field in (
            "runtime_version",
            "source_commit",
            "configuration_hash",
            "initial_design_hash",
            "initial_design_capability",
            "sampling_stream_hash",
            "sampling_stream_capability",
            "retcode",
        ):
            if not row.get(field, "").strip():
                _fail(f"missing {field} for {key}")
        if not row["configuration_hash"].startswith(CONFIGURATION_PREFIX):
            _fail(
                f"unsupported configuration_hash for {key}: "
                f"{row['configuration_hash']!r}"
            )
        source_commit = row["source_commit"]
        if source_commit == "unknown" or source_commit.endswith("-dirty"):
            _fail(f"non-reproducible source_commit for {key}: {source_commit}")
        if row["runtime"] == "Python" and not row.get("python_sambo_version"):
            _fail(f"missing python_sambo_version for {key}")
        results[key] = row
    return results


def normalized_log_regret(row, key):
    minimum = _finite_float(row, "minimum", key)
    optimum = _finite_float(row, "optimum", key)
    scale = max(1.0, abs(optimum))
    regret = abs(minimum - optimum) / scale
    return math.log10(max(regret, REGRET_FLOOR))


def quality_threshold_gate(julia, python):
    rotations = {}
    rows = {}
    for case in sorted(julia):
        julia_row = julia[case]
        python_row = python[case]
        rotation = case[:3]
        row = case[:2]
        samples_for_rotation = rotations.setdefault(
            rotation,
            {"differences": [], "row": julia_row},
        )
        samples_for_rotation["differences"].append(
            normalized_log_regret(julia_row, case)
            - normalized_log_regret(python_row, case)
        )
        row_values = rows.setdefault(
            row,
            {
                "julia_minima": [],
                "python_minima": [],
                "rotations": {},
                "optimum": _finite_float(julia_row, "optimum", row),
            },
        )
        row_values["julia_minima"].append(
            _finite_float(julia_row, "minimum", case)
        )
        row_values["python_minima"].append(
            _finite_float(python_row, "minimum", case)
        )

    failures = []
    for rotation, values in sorted(rotations.items()):
        count = len(values["differences"])
        if count < MINIMUM_TRIALS:
            failures.append(
                f"quality threshold {rotation}: requires at least "
                f"{MINIMUM_TRIALS} trials, found {count}"
            )
            continue
        margin = _finite_float(
            values["row"],
            "quality_threshold",
            rotation,
        )
        observed = statistics.median(values["differences"])
        passed = observed <= margin
        rows[rotation[:2]]["rotations"][rotation[2]] = {
            "trials": count,
            "median_difference": observed,
            "margin": margin,
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"quality threshold {rotation}: median log-regret difference "
                f"{observed:.6g} exceeds margin {margin}"
            )

    summaries = {}
    for row, values in sorted(rows.items()):
        found = len(values["rotations"])
        if found != REQUIRED_ROTATIONS:
            failures.append(
                f"quality threshold {row}: requires {REQUIRED_ROTATIONS} "
                f"rotations, found {found}"
            )
        passed = sum(
            rotation["passed"]
            for rotation in values["rotations"].values()
        )
        differences = [
            rotation["median_difference"]
            for rotation in values["rotations"].values()
        ]
        summaries[row] = {
            "julia_median": statistics.median(values["julia_minima"]),
            "python_median": statistics.median(values["python_minima"]),
            "optimum": values["optimum"],
            "passed_rotations": passed,
            "rotations": found,
            "worst_difference": max(differences) if differences else math.inf,
        }
    return failures, summaries


def validate_matrices(julia, python):
    failures = []
    julia_cases = set(julia)
    python_cases = set(python)
    if julia_cases != python_cases:
        failures.append(
            "correctness matrices differ: "
            f"missing Julia={sorted(python_cases - julia_cases)}, "
            f"missing Python={sorted(julia_cases - python_cases)}"
        )
        return failures

    for runtime, rows in (("Julia", julia), ("Python", python)):
        for key, row in rows.items():
            if row.get("runtime") != runtime:
                failures.append(
                    f"{runtime} input {key}: runtime={row.get('runtime')!r}"
                )

    for case in sorted(julia):
        julia_row = julia[case]
        python_row = python[case]
        for field in MATCHED_METADATA:
            if julia_row.get(field) != python_row.get(field):
                failures.append(
                    f"{case}: {field} differs: "
                    f"Julia={julia_row.get(field)!r}, "
                    f"Python={python_row.get(field)!r}"
                )
    return failures


def main(julia_path, python_path):
    julia = read_results(julia_path)
    python = read_results(python_path)
    failures = validate_matrices(julia, python)
    summaries = {}
    if not failures:
        threshold_failures, summaries = quality_threshold_gate(julia, python)
        failures.extend(threshold_failures)
    if failures:
        raise SystemExit(
            "matched cross-runtime quality threshold failed:\n"
            + "\n".join(failures)
        )
    worst_group, worst = max(
        summaries.items(),
        key=lambda item: item[1]["worst_difference"],
    )
    for group, summary in sorted(summaries.items()):
        print(
            f"{group}: Julia median={summary['julia_median']:.6g}; "
            f"Python median={summary['python_median']:.6g}; "
            f"within threshold={summary['passed_rotations']}/"
            f"{summary['rotations']}"
        )
    print(
        f"validated {len(summaries)} native solver/problem groups; "
        f"worst median log-regret difference={worst['worst_difference']:.6g} "
        f"for {worst_group}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
