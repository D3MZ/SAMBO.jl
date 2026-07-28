import csv
import importlib.util
import math
import sys
from pathlib import Path


START = "<!-- benchmark-table:start -->"
END = "<!-- benchmark-table:end -->"
QUALITY_START = "<!-- correctness-table:start -->"
QUALITY_END = "<!-- correctness-table:end -->"
ALGORITHMS = ("SCE-UA", "SMBO", "SHGO")
PROBLEMS = (
    ("hartmann6", "Hartmann-6"),
    ("rotated_rastrigin5", "Rotated Rastrigin-5"),
    ("rotated_rosenbrock5", "Rotated Rosenbrock-5"),
)
QUALITY_HEADER = (
    "| Algorithm | Problem | Known optimum | Julia median | "
    "Python median | Within threshold |"
)
QUALITY_SEPARATOR = "|---|---|---:|---:|---:|---:|"
# README medians are display values, not reproducibility inputs. Permit modest
# platform drift while rejecting materially stale results.
README_MEDIAN_LOG_TOLERANCE = 0.05
REGRET_FLOOR = 1e-6
BENCHMARK_DIRECTORY = Path(__file__).parent
DEFAULT_README = BENCHMARK_DIRECTORY.parent / "README.md"
DEFAULT_JULIA = BENCHMARK_DIRECTORY / "results" / "julia.csv"
DEFAULT_PYTHON = BENCHMARK_DIRECTORY / "results" / "python.csv"


def _load_comparator():
    path = BENCHMARK_DIRECTORY / "compare_correctness.py"
    spec = importlib.util.spec_from_file_location("compare_correctness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    result = {}
    for row in rows:
        algorithm = row["algorithm"]
        if algorithm in result:
            raise ValueError(f"duplicate benchmark row: {algorithm}")
        result[algorithm] = row
    return result


def _time(value):
    value = float(value)
    return f"{value:.4f}" if value < 1 else f"{value:.3f}"


def render_table(julia, python):
    if set(julia) != set(ALGORITHMS) or set(python) != set(ALGORITHMS):
        raise ValueError("benchmark CSVs must contain SCE-UA, SMBO, and SHGO")
    lines = [
        "| Algorithm | Julia | Python | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        julia_time = float(julia[algorithm]["time_ms"])
        python_time = float(python[algorithm]["time_ms"])
        lines.append(
            f"| {algorithm} | {_time(julia_time)} ms | "
            f"{_time(python_time)} ms | "
            f"{python_time / julia_time:.1f}× faster |"
        )
    return "\n".join(lines)


def section_parts(source, start, end):
    before, separator, remainder = source.partition(start)
    if not separator:
        raise ValueError(f"README marker is missing: {start}")
    section, separator, after = remainder.partition(end)
    if not separator:
        raise ValueError(f"README marker is missing: {end}")
    return before, section.strip(), after


def update(readme_path, julia_path, python_path):
    readme_path = Path(readme_path)
    before, _, after = section_parts(readme_path.read_text(), START, END)
    table = render_table(read_rows(julia_path), read_rows(python_path))
    readme_path.write_text(f"{before}{START}\n{table}\n{END}{after}")


def _score(value):
    return f"{float(value):.6g}".replace("-", "−")


def render_quality_table(summaries):
    lines = [
        QUALITY_HEADER,
        QUALITY_SEPARATOR,
    ]
    for algorithm in ALGORITHMS:
        for problem, label in PROBLEMS:
            summary = summaries[(problem, algorithm)]
            lines.append(
                f"| {algorithm} | {label} | {_score(summary['optimum'])} | "
                f"{_score(summary['julia_median'])} | "
                f"{_score(summary['python_median'])} | "
                f"{summary['passed_rotations']}/{summary['rotations']} |"
            )
    return "\n".join(lines)


def quality_table(julia_path, python_path):
    return render_quality_table(quality_summaries(julia_path, python_path))


def quality_summaries(julia_path, python_path):
    comparator = _load_comparator()
    julia = comparator.read_results(julia_path)
    python = comparator.read_results(python_path)
    failures = comparator.validate_matrices(julia, python)
    summaries = {}
    if not failures:
        failures, summaries = comparator.quality_threshold_gate(julia, python)
    if failures:
        raise ValueError("\n".join(failures))
    return summaries


def parse_quality_table(section):
    lines = section.splitlines()
    if lines[:2] != [QUALITY_HEADER, QUALITY_SEPARATOR]:
        raise SystemExit("README cross-runtime quality table schema is stale")
    labels = {label: problem for problem, label in PROBLEMS}
    rows = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            raise SystemExit("README cross-runtime quality table schema is stale")
        algorithm, label, optimum, julia, python, passed = cells
        if algorithm not in ALGORITHMS or label not in labels:
            raise SystemExit("README cross-runtime quality table rows are stale")
        key = (labels[label], algorithm)
        if key in rows:
            raise SystemExit("README cross-runtime quality table has duplicate rows")
        try:
            julia_median = float(julia.replace("−", "-"))
            python_median = float(python.replace("−", "-"))
            if not math.isfinite(julia_median) or not math.isfinite(python_median):
                raise ValueError
            rows[key] = {
                "optimum_text": optimum,
                "julia_median": julia_median,
                "python_median": python_median,
                "passed": passed,
            }
        except ValueError as error:
            raise SystemExit(
                "README cross-runtime quality table has invalid numeric values"
            ) from error
    expected = {
        (problem, algorithm)
        for algorithm in ALGORITHMS
        for problem, _ in PROBLEMS
    }
    if set(rows) != expected:
        raise SystemExit("README cross-runtime quality table rows are stale")
    return rows


def _log_regret(value, optimum):
    scale = max(1.0, abs(optimum))
    return math.log10(max(abs(value - optimum) / scale, REGRET_FLOOR))


def check_quality_table(checked_in, summaries):
    rows = parse_quality_table(checked_in)
    for key, summary in summaries.items():
        row = rows[key]
        if row["optimum_text"] != _score(summary["optimum"]):
            raise SystemExit(
                f"README cross-runtime quality table optimum is stale for {key}"
            )
        expected_passed = (
            f"{summary['passed_rotations']}/{summary['rotations']}"
        )
        if row["passed"] != expected_passed:
            raise SystemExit(
                f"README cross-runtime quality table pass count is stale for {key}"
            )
        for runtime in ("julia", "python"):
            displayed = row[f"{runtime}_median"]
            fresh = summary[f"{runtime}_median"]
            difference = abs(
                _log_regret(displayed, summary["optimum"])
                - _log_regret(fresh, summary["optimum"])
            )
            if difference > README_MEDIAN_LOG_TOLERANCE:
                raise SystemExit(
                    "README cross-runtime quality table median is stale for "
                    f"{key} {runtime}: log-regret drift={difference:.6g}"
                )


def update_quality(readme_path, julia_path, python_path):
    readme_path = Path(readme_path)
    before, _, after = section_parts(
        readme_path.read_text(),
        QUALITY_START,
        QUALITY_END,
    )
    readme_path.write_text(
        f"{before}{QUALITY_START}\n"
        f"{quality_table(julia_path, python_path)}\n"
        f"{QUALITY_END}{after}"
    )


def check_quality(readme_path, julia_path, python_path):
    _, checked_in, _ = section_parts(
        Path(readme_path).read_text(),
        QUALITY_START,
        QUALITY_END,
    )
    check_quality_table(
        checked_in,
        quality_summaries(julia_path, python_path),
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        update(DEFAULT_README, DEFAULT_JULIA, DEFAULT_PYTHON)
    elif len(sys.argv) == 5 and sys.argv[1] == "--check-quality":
        check_quality(*sys.argv[2:])
    elif len(sys.argv) == 5 and sys.argv[1] == "--update-quality":
        update_quality(*sys.argv[2:])
    elif len(sys.argv) == 4:
        update(*sys.argv[1:])
    else:
        raise SystemExit(
            "usage: update_readme.py [README.md julia.csv python.csv] | "
            "--check-quality README.md julia.csv python.csv | "
            "--update-quality README.md julia.csv python.csv"
        )
