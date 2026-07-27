import csv
import math
import random
import statistics
import sys


EXACT_EQUIVALENCE_LOG_GAP_TOLERANCE = 0.25
EXACT_PROFILE = "exact-v1"
NATIVE_DEFAULT_PROFILE = "native-default-v1"
SUPPORTED_PROFILES = (EXACT_PROFILE, NATIVE_DEFAULT_PROFILE)
MATCHED_METADATA = (
    "profile",
    "budget",
    "optimum",
    "target",
    "quality_target",
    "required_hit_rate",
    "noninferiority_margin",
    "source_commit",
    "configuration_hash",
    "initial_design_hash",
    "initial_design_capability",
)
REQUIRED_PROVENANCE = (
    "runtime_version",
    "source_commit",
)
REQUIRED_CONFIGURATION = (
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
    "target",
)


def _trial_id(row):
    value = row.get("trial_id", row.get("seed"))
    if value is None:
        raise SystemExit("missing trial_id")
    return int(value)


def _checkpoint(row):
    value = row.get("evaluation", row.get("evaluations"))
    if value is None:
        raise SystemExit("missing evaluation checkpoint")
    return int(value)


def _profile(row, key):
    configuration_hash = row.get("configuration_hash", "").strip()
    inferred = configuration_hash.split(":", 1)[0]
    declared = row.get("profile", "").strip()
    if declared and declared != inferred:
        raise SystemExit(
            f"profile/configuration_hash mismatch for {key}: "
            f"profile={declared!r}, configuration_hash={configuration_hash!r}"
        )
    profile = declared or inferred
    if profile not in SUPPORTED_PROFILES:
        raise SystemExit(f"unsupported comparison profile for {key}: {profile!r}")
    return profile


def _finite_float(row, field, key):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid {field} for {key}") from error
    if not math.isfinite(value):
        raise SystemExit(f"nonfinite {field} for {key}")
    return value


def _integer(row, field, key):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid {field} for {key}") from error


def read_results(path):
    with open(path, newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"missing CSV header in {path}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        if missing:
            raise SystemExit(
                f"missing required CSV columns in {path}: {missing}"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no correctness result rows in {path}")
    results = {}
    for row in rows:
        try:
            key = (
                row["problem"],
                row["algorithm"],
                _trial_id(row),
                _checkpoint(row),
            )
        except KeyError as error:
            raise SystemExit(f"missing comparison key field: {error.args[0]}") from error
        if key in results:
            raise SystemExit(f"duplicate comparison key: {key}")
        _finite_float(row, "minimum", key)
        _finite_float(row, "optimum", key)
        _finite_float(row, "target", key)
        budget = _integer(row, "budget", key)
        evaluations = _integer(row, "evaluations", key)
        checkpoint = key[3]
        budget > 0 or _fail(f"invalid budget for {key}")
        evaluations >= 0 or _fail(f"invalid evaluations for {key}")
        checkpoint >= 0 or _fail(f"invalid evaluation checkpoint for {key}")
        checkpoint == evaluations or _fail(
            f"checkpoint/evaluations mismatch for {key}: "
            f"checkpoint={checkpoint}, evaluations={evaluations}"
        )
        evaluations <= budget or _fail(
            f"evaluations exceed budget for {key}: "
            f"evaluations={evaluations}, budget={budget}"
        )
        for field in REQUIRED_PROVENANCE + REQUIRED_CONFIGURATION:
            if field not in row or not row[field].strip():
                raise SystemExit(f"missing {field} for {key}")
        source_commit = row["source_commit"]
        if source_commit == "unknown" or source_commit.endswith("-dirty"):
            raise SystemExit(
                f"non-reproducible source_commit for {key}: {source_commit}"
            )
        if row.get("runtime") == "Python" and not row.get(
            "python_sambo_version",
        ):
            raise SystemExit(f"missing python_sambo_version for {key}")
        _profile(row, key)
        results[key] = row
    return results


def _fail(message):
    raise SystemExit(message)


def _normalized_log_gap(row, key, *, clip_at_quality_target=False):
    minimum = _finite_float(row, "minimum", key)
    optimum = _finite_float(row, "optimum", key)
    scale = max(1.0, abs(optimum))
    gap = abs(minimum - optimum) / scale
    if clip_at_quality_target:
        quality_target = row.get("quality_target")
        if quality_target not in (None, ""):
            quality_floor = abs(float(quality_target) - optimum) / scale
            gap = max(gap, quality_floor)
    gap = max(gap, 1e-15)
    return math.log10(gap)


def quality_gate(runtime, rows):
    failures = []
    terminal = _terminal_rows(rows)
    for key, row in sorted(rows.items()):
        evaluations = _integer(row, "evaluations", key)
        budget = _integer(row, "budget", key)
        if evaluations > budget:
            failures.append(
                f"{runtime} {key}: evaluations={evaluations} budget={budget}"
            )
    grouped = {}
    for case, (_, row) in terminal.items():
        group = case[:2]
        target = float(row.get("quality_target") or row["target"])
        required = float(row.get("required_hit_rate") or 1.0)
        grouped.setdefault(group, []).append(
            (_finite_float(row, "minimum", case) <= target, target, required)
        )
    for group, trials in sorted(grouped.items()):
        target = trials[0][1]
        required = trials[0][2]
        hit_rate = sum(hit for hit, _, _ in trials) / len(trials)
        if hit_rate < required:
            failures.append(
                f"{runtime} {group}: quality hit rate {hit_rate:.3f} "
                f"is below {required:.3f} at target={target}"
            )
    return failures


def exact_equivalence_gate(julia, python):
    failures = []
    for key in sorted(set(julia) & set(python)):
        if _profile(julia[key], key) != EXACT_PROFILE:
            continue
        statistic = symmetric_equivalence_statistic(
            julia[key],
            python[key],
            key,
        )
        if statistic > EXACT_EQUIVALENCE_LOG_GAP_TOLERANCE:
            failures.append(
                f"exact equivalence (symmetric) {key}: absolute normalized "
                f"log-gap difference {statistic:.6g} exceeds "
                f"{EXACT_EQUIVALENCE_LOG_GAP_TOLERANCE}"
            )
    return failures


def symmetric_equivalence_statistic(left, right, key):
    """Absolute log-error ratio; invariant to runtime ordering."""
    return abs(
        _normalized_log_gap(left, key)
        - _normalized_log_gap(right, key)
    )


def paired_log_gap_summary(julia, python, *, profile=NATIVE_DEFAULT_PROFILE):
    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    differences = [
        _normalized_log_gap(
            julia_terminal[case][1], case, clip_at_quality_target=True,
        )
        - _normalized_log_gap(
            python_terminal[case][1], case, clip_at_quality_target=True,
        )
        for case in sorted(julia_terminal)
        if _profile(julia_terminal[case][1], julia_terminal[case][0]) == profile
    ]
    return {
        "count": len(differences),
        "median": statistics.median(differences) if differences else math.nan,
    }


def paired_bootstrap_interval(
    differences,
    *,
    samples=2_000,
    confidence=0.95,
    seed=0,
):
    differences = list(differences)
    differences or raise_empty_differences()
    samples > 0 or _fail("bootstrap samples must be positive")
    0 < confidence < 1 or _fail("bootstrap confidence must be in (0, 1)")
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        statistics.mean(
            differences[generator.randrange(count)]
            for _ in range(count)
        )
        for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    lower = estimates[max(0, math.floor(tail * samples))]
    upper = estimates[min(samples - 1, math.ceil((1 - tail) * samples) - 1)]
    return lower, upper


def raise_empty_differences():
    raise ValueError("at least one paired difference is required")


def _terminal_rows(rows):
    terminal = {}
    for key, row in rows.items():
        case = key[:3]
        if case not in terminal or key[3] > terminal[case][0]:
            terminal[case] = (key[3], key, row)
    return {case: pair[1:] for case, pair in terminal.items()}


def noninferiority_gate(
    julia,
    python,
    *,
    margin=None,
    samples=2_000,
    confidence=0.95,
):
    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    grouped = {}
    for case in sorted(julia_terminal):
        julia_key, julia_row = julia_terminal[case]
        if _profile(julia_row, julia_key) != NATIVE_DEFAULT_PROFILE:
            continue
        _, python_row = python_terminal[case]
        group = case[:2]
        grouped.setdefault(group, []).append(
            _normalized_log_gap(
                julia_row, case, clip_at_quality_target=True,
            )
            - _normalized_log_gap(
                python_row, case, clip_at_quality_target=True,
            )
        )
    failures = []
    for group, differences in sorted(grouped.items()):
        _, sample_row = julia_terminal[next(
            case for case in julia_terminal if case[:2] == group
        )]
        group_margin = (
            float(sample_row.get("noninferiority_margin") or 0.25)
            if margin is None
            else margin
        )
        _, upper = paired_bootstrap_interval(
            differences,
            samples=samples,
            confidence=confidence,
        )
        if upper > group_margin:
            failures.append(
                f"noninferiority {group}: upper paired log-gap bound "
                f"{upper:.6g} exceeds margin {group_margin}"
            )
    return failures


def _validate_metadata(julia, python):
    failures = []
    julia_terminal = _terminal_rows(julia)
    python_terminal = _terminal_rows(python)
    for runtime, rows in (("Julia", julia), ("Python", python)):
        for key, row in sorted(rows.items()):
            if row.get("runtime") != runtime:
                failures.append(
                    f"{runtime} input {key}: runtime={row.get('runtime')!r}"
                )
        grouped = {}
        for key, row in rows.items():
            grouped.setdefault(key[:3], []).append((key, row))
        for case, checkpoints in sorted(grouped.items()):
            _, baseline = checkpoints[0]
            for key, row in checkpoints[1:]:
                for field in MATCHED_METADATA:
                    if row.get(field) != baseline.get(field):
                        failures.append(
                            f"{runtime} {case}: {field} changes at checkpoint "
                            f"{key[3]}: {baseline.get(field)!r} -> "
                            f"{row.get(field)!r}"
                        )
    for case in sorted(julia_terminal):
        julia_key, julia_row = julia_terminal[case]
        python_key, python_row = python_terminal[case]
        for field in MATCHED_METADATA:
            if field in julia_row or field in python_row:
                left = julia_row.get(field)
                right = python_row.get(field)
                if left != right:
                    failures.append(
                        f"{case}: {field} differs: Julia={left!r}, Python={right!r}"
                    )
        left_profile = _profile(julia_row, julia_key)
        right_profile = _profile(python_row, python_key)
        if left_profile != right_profile:
            failures.append(
                f"{case}: comparison profiles differ: "
                f"Julia={left_profile!r}, Python={right_profile!r}"
            )
        for runtime, key, row in (
            ("Julia", julia_key, julia_row),
            ("Python", python_key, python_row),
        ):
            for field in REQUIRED_PROVENANCE:
                if field not in row or not row[field].strip():
                    failures.append(f"{runtime} {key}: missing {field}")
    return failures


def main(julia_path, python_path):
    julia = read_results(julia_path)
    python = read_results(python_path)
    julia_profiles = {_profile(row, key) for key, row in julia.items()}
    python_profiles = {_profile(row, key) for key, row in python.items()}
    if len(julia_profiles) != 1 or len(python_profiles) != 1:
        raise SystemExit(
            "each correctness matrix must contain exactly one profile: "
            f"Julia={sorted(julia_profiles)}, Python={sorted(python_profiles)}"
        )
    if julia_profiles != python_profiles:
        raise SystemExit(
            "correctness matrix profiles differ: "
            f"Julia={sorted(julia_profiles)}, Python={sorted(python_profiles)}"
        )
    julia_cases = {key[:3] for key in julia}
    python_cases = {key[:3] for key in python}
    if julia_cases != python_cases:
        missing_julia = sorted(python_cases - julia_cases)
        missing_python = sorted(julia_cases - python_cases)
        raise SystemExit(
            f"correctness matrices differ: missing Julia={missing_julia}, "
            f"missing Python={missing_python}"
        )

    failures = _validate_metadata(julia, python)
    julia_terminal = _terminal_rows(julia)
    exact_cases = {
        case
        for case, (key, row) in julia_terminal.items()
        if _profile(row, key) == EXACT_PROFILE
    }
    for case in sorted(exact_cases):
        julia_keys = {key for key in julia if key[:3] == case}
        python_keys = {key for key in python if key[:3] == case}
        if julia_keys != python_keys:
            failures.append(
                f"exact equivalence matrices differ for {case}: "
                f"Julia={sorted(key[3] for key in julia_keys)}, "
                f"Python={sorted(key[3] for key in python_keys)}"
            )
    failures.extend(quality_gate("Julia", julia))
    failures.extend(quality_gate("Python", python))
    failures.extend(exact_equivalence_gate(julia, python))
    failures.extend(noninferiority_gate(julia, python))
    if failures:
        raise SystemExit(
            "cross-runtime correctness matrix failed:\n"
            + "\n".join(failures)
        )

    summary = paired_log_gap_summary(julia, python)
    exact_count = sum(
        _profile(row, key) == EXACT_PROFILE
        for key, row in julia.items()
    )
    message = (
        f"validated {exact_count} exact-v1 symmetric-equivalence checkpoints "
        f"and {summary['count']} native-default non-inferiority cases"
    )
    if summary["count"]:
        message += (
            "; median Julia-Python quality-floored log-gap="
            f"{summary['median']:.6g}"
        )
    print(message)


if __name__ == "__main__":
    main(*sys.argv[1:])
