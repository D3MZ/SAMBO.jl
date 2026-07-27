import csv
import math
import statistics
import sys


CONFIGURATION_PREFIX = "python-sambo-1.25.2-matched-v5:"
MINIMUM_TRIALS = 10
REQUIRED_ROTATIONS = 6
REGRET_FLOOR = 1e-6
MATCHED_METADATA = (
    "budget",
    "optimum",
    "noninferiority_margin",
    "source_commit",
    "configuration_hash",
    "initial_design_hash",
    "initial_design_capability",
)
REQUIRED_COLUMNS = (
    "runtime",
    "runtime_version",
    "source_commit",
    "problem",
    "algorithm",
    "rotation_id",
    "trial_id",
    "configuration_hash",
    "initial_design_hash",
    "initial_design_capability",
    "budget",
    "evaluation",
    "evaluations",
    "minimum",
    "optimum",
    "noninferiority_margin",
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
                _integer(row, "evaluation", None),
            )
        except KeyError as error:
            _fail(f"missing comparison key field: {error.args[0]}")
        if key in results:
            _fail(f"duplicate comparison key: {key}")
        budget = _integer(row, "budget", key)
        evaluations = _integer(row, "evaluations", key)
        budget > 0 or _fail(f"invalid budget for {key}")
        evaluations == key[4] or _fail(
            f"checkpoint/evaluations mismatch for {key}: "
            f"checkpoint={key[4]}, evaluations={evaluations}"
        )
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


def _terminal_rows(rows):
    terminal = {}
    for key, row in rows.items():
        case = key[:4]
        if case not in terminal or key[4] > terminal[case][0]:
            terminal[case] = (key[4], key, row)
    return {case: pair[1:] for case, pair in terminal.items()}


def normalized_log_regret(row, key):
    minimum = _finite_float(row, "minimum", key)
    optimum = _finite_float(row, "optimum", key)
    scale = max(1.0, abs(optimum))
    regret = abs(minimum - optimum) / scale
    return math.log10(max(regret, REGRET_FLOOR))


def noninferiority_gate(julia, python):
    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    rotations = {}
    rows = {}
    for case in sorted(julia_terminal):
        julia_key, julia_row = julia_terminal[case]
        python_key, python_row = python_terminal[case]
        rotation = case[:3]
        row = case[:2]
        samples_for_rotation = rotations.setdefault(
            rotation,
            {"julia": [], "python": [], "row": julia_row},
        )
        samples_for_rotation["julia"].append(
            normalized_log_regret(julia_row, julia_key)
        )
        samples_for_rotation["python"].append(
            normalized_log_regret(python_row, python_key)
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
            _finite_float(julia_row, "minimum", julia_key)
        )
        row_values["python_minima"].append(
            _finite_float(python_row, "minimum", python_key)
        )

    failures = []
    for rotation, values in sorted(rotations.items()):
        count = len(values["julia"])
        if count < MINIMUM_TRIALS:
            failures.append(
                f"noninferiority {rotation}: requires at least "
                f"{MINIMUM_TRIALS} trials, found {count}"
            )
            continue
        margin = _finite_float(
            values["row"],
            "noninferiority_margin",
            rotation,
        )
        observed = statistics.median(values["julia"]) - statistics.median(
            values["python"]
        )
        passed = observed <= margin
        rows[rotation[:2]]["rotations"][rotation[2]] = {
            "trials": count,
            "median_difference": observed,
            "margin": margin,
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"noninferiority {rotation}: median log-regret difference "
                f"{observed:.6g} exceeds margin {margin}"
            )

    summaries = {}
    for row, values in sorted(rows.items()):
        found = len(values["rotations"])
        if found != REQUIRED_ROTATIONS:
            failures.append(
                f"noninferiority {row}: requires {REQUIRED_ROTATIONS} "
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
    julia_cases = {key[:4] for key in julia}
    python_cases = {key[:4] for key in python}
    if julia_cases != python_cases:
        failures.append(
            "correctness matrices differ: "
            f"missing Julia={sorted(python_cases - julia_cases)}, "
            f"missing Python={sorted(julia_cases - python_cases)}"
        )
        return failures

    for runtime, rows in (("Julia", julia), ("Python", python)):
        grouped = {}
        for key, row in rows.items():
            if row.get("runtime") != runtime:
                failures.append(
                    f"{runtime} input {key}: runtime={row.get('runtime')!r}"
                )
            grouped.setdefault(key[:4], []).append((key, row))
        for case, checkpoints in sorted(grouped.items()):
            _, baseline = checkpoints[0]
            for key, row in checkpoints[1:]:
                for field in MATCHED_METADATA:
                    if row.get(field) != baseline.get(field):
                        failures.append(
                            f"{runtime} {case}: {field} changes at checkpoint "
                            f"{key[4]}"
                        )

    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    for case in sorted(julia_terminal):
        julia_key, julia_row = julia_terminal[case]
        python_key, python_row = python_terminal[case]
        for field in MATCHED_METADATA:
            if julia_row.get(field) != python_row.get(field):
                failures.append(
                    f"{case}: {field} differs: "
                    f"Julia={julia_row.get(field)!r}, "
                    f"Python={python_row.get(field)!r}"
                )
        if julia_key[:4] != python_key[:4]:
            failures.append(f"terminal comparison keys differ for {case}")
    return failures


def main(julia_path, python_path):
    julia = read_results(julia_path)
    python = read_results(python_path)
    failures = validate_matrices(julia, python)
    summaries = {}
    if not failures:
        noninferiority_failures, summaries = noninferiority_gate(julia, python)
        failures.extend(noninferiority_failures)
    if failures:
        raise SystemExit(
            "native cross-runtime non-inferiority failed:\n"
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
