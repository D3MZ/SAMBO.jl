import csv
import sys
from pathlib import Path


START = "<!-- benchmark-table:start -->"
END = "<!-- benchmark-table:end -->"
ALGORITHMS = ("SCE-UA", "SMBO", "SHGO")
BENCHMARK_DIRECTORY = Path(__file__).parent
DEFAULT_README = BENCHMARK_DIRECTORY.parent / "README.md"
DEFAULT_JULIA = BENCHMARK_DIRECTORY / "results" / "julia.csv"
DEFAULT_PYTHON = BENCHMARK_DIRECTORY / "results" / "python.csv"


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


def update(readme_path, julia_path, python_path):
    readme_path = Path(readme_path)
    source = readme_path.read_text()
    before, separator, remainder = source.partition(START)
    if not separator:
        raise ValueError("README benchmark start marker is missing")
    _, separator, after = remainder.partition(END)
    if not separator:
        raise ValueError("README benchmark end marker is missing")
    table = render_table(read_rows(julia_path), read_rows(python_path))
    readme_path.write_text(f"{before}{START}\n{table}\n{END}{after}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        update(DEFAULT_README, DEFAULT_JULIA, DEFAULT_PYTHON)
    elif len(sys.argv) == 4:
        update(*sys.argv[1:])
    else:
        raise SystemExit(
            "usage: update_readme.py [README.md julia.csv python.csv]"
        )
