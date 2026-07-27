import csv
import math
import random
import statistics
import sys


CONFIGURATION_PREFIX = "native-noninferiority-v3:"
MINIMUM_TRIALS = 10
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
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
        evaluations == key[3] or _fail(
            f"checkpoint/evaluations mismatch for {key}: "
            f"checkpoint={key[3]}, evaluations={evaluations}"
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
        case = key[:3]
        if case not in terminal or key[3] > terminal[case][0]:
            terminal[case] = (key[3], key, row)
    return {case: pair[1:] for case, pair in terminal.items()}


def normalized_log_regret(row, key):
    minimum = _finite_float(row, "minimum", key)
    optimum = _finite_float(row, "optimum", key)
    scale = max(1.0, abs(optimum))
    regret = abs(minimum - optimum) / scale
    return math.log10(max(regret, 1e-15))


def bootstrap_median_difference_upper(
    julia_values,
    python_values,
    *,
    samples=BOOTSTRAP_SAMPLES,
    confidence=CONFIDENCE,
    seed=0,
):
    julia_values = tuple(julia_values)
    python_values = tuple(python_values)
    if not julia_values:
        raise ValueError("Julia sample must not be empty")
    if not python_values:
        raise ValueError("Python sample must not be empty")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    generator = random.Random(seed)
    estimates = sorted(
        statistics.median(
            julia_values[generator.randrange(len(julia_values))]
            for _ in julia_values
        )
        - statistics.median(
            python_values[generator.randrange(len(python_values))]
            for _ in python_values
        )
        for _ in range(samples)
    )
    index = min(samples - 1, math.ceil(confidence * samples) - 1)
    return estimates[index]


def noninferiority_gate(
    julia,
    python,
    *,
    samples=BOOTSTRAP_SAMPLES,
    confidence=CONFIDENCE,
):
    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    grouped = {}
    for case in sorted(julia_terminal):
        julia_key, julia_row = julia_terminal[case]
        python_key, python_row = python_terminal[case]
        group = case[:2]
        samples_for_group = grouped.setdefault(
            group,
            {"julia": [], "python": [], "row": julia_row},
        )
        samples_for_group["julia"].append(
            normalized_log_regret(julia_row, julia_key)
        )
        samples_for_group["python"].append(
            normalized_log_regret(python_row, python_key)
        )

    failures = []
    summaries = {}
    for group, values in sorted(grouped.items()):
        count = len(values["julia"])
        if count < MINIMUM_TRIALS:
            failures.append(
                f"noninferiority {group}: requires at least "
                f"{MINIMUM_TRIALS} trials, found {count}"
            )
            continue
        margin = _finite_float(
            values["row"],
            "noninferiority_margin",
            group,
        )
        observed = statistics.median(values["julia"]) - statistics.median(
            values["python"]
        )
        upper = bootstrap_median_difference_upper(
            values["julia"],
            values["python"],
            samples=samples,
            confidence=confidence,
        )
        summaries[group] = {
            "trials": count,
            "median_difference": observed,
            "upper_bound": upper,
            "margin": margin,
        }
        if upper > margin:
            failures.append(
                f"noninferiority {group}: upper {confidence:.0%} bootstrap "
                f"bound {upper:.6g} exceeds margin {margin}; "
                f"observed median log-regret difference={observed:.6g}"
            )
    return failures, summaries


def validate_matrices(julia, python):
    failures = []
    julia_cases = {key[:3] for key in julia}
    python_cases = {key[:3] for key in python}
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
            grouped.setdefault(key[:3], []).append((key, row))
        for case, checkpoints in sorted(grouped.items()):
            _, baseline = checkpoints[0]
            for key, row in checkpoints[1:]:
                for field in MATCHED_METADATA:
                    if row.get(field) != baseline.get(field):
                        failures.append(
                            f"{runtime} {case}: {field} changes at checkpoint "
                            f"{key[3]}"
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
        if julia_key[:3] != python_key[:3]:
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
        key=lambda item: item[1]["upper_bound"],
    )
    print(
        f"validated {len(summaries)} native solver/problem groups; "
        f"worst upper bound={worst['upper_bound']:.6g} for {worst_group}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
