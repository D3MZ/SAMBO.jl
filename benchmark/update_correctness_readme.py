import csv
import statistics
import sys
from pathlib import Path


START = "<!-- correctness-table:start -->"
END = "<!-- correctness-table:end -->"
ALGORITHMS = ("SCE-UA", "SMBO", "SHGO")
PROBLEMS = {
    "hartmann6": "Hartmann-6",
    "rotated_rastrigin5": "Rotated Rastrigin-5",
    "rotated_rosenbrock5": "Rotated Rosenbrock-5",
}


def read_rows(path):
    with Path(path).open(newline="") as source:
        return list(csv.DictReader(source))


def terminal_rows(rows):
    terminal = {}
    for row in rows:
        key = (row["problem"], row["algorithm"], int(row["trial_id"]))
        evaluation = int(row["evaluation"])
        if key not in terminal or evaluation > int(
            terminal[key]["evaluation"],
        ):
            terminal[key] = row
    return list(terminal.values())


def _number(value, digits=3):
    value = float(value)
    if value != 0 and abs(value) < 1e-4:
        rendered = f"{value:.3g}"
    else:
        rendered = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return rendered.replace("-", "−")


def _cell(rows):
    values = [float(row["minimum"]) for row in rows]
    hits = sum(
        float(row["minimum"]) <= float(row["target"])
        for row in rows
    )
    return f"{_number(statistics.median(values))}; {hits}/{len(rows)}"


def render_table(julia_rows, python_rows):
    julia = {
        (row["problem"], row["algorithm"], int(row["trial_id"])): row
        for row in julia_rows
    }
    python = {
        (row["problem"], row["algorithm"], int(row["trial_id"])): row
        for row in python_rows
    }
    if julia.keys() != python.keys():
        raise ValueError("terminal correctness cases differ between runtimes")
    lines = [
        "| Algorithm | Problem (known optimum; target) | Julia | Python |",
        "|---|---|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        for problem in PROBLEMS:
            keys = sorted(
                key
                for key in julia
                if key[0] == problem and key[1] == algorithm
            )
            if not keys:
                continue
            sample = julia[keys[0]]
            label = (
                f"{PROBLEMS[problem]} ({_number(sample['optimum'])}; "
                f"≤ {_number(sample['target'], 1)})"
            )
            lines.append(
                f"| {algorithm} | {label} | "
                f"{_cell([julia[key] for key in keys])} | "
                f"{_cell([python[key] for key in keys])} |"
            )
    return "\n".join(lines)


def update(readme_path, julia_path, python_path):
    readme_path = Path(readme_path)
    source = readme_path.read_text()
    before, separator, remainder = source.partition(START)
    if not separator:
        raise ValueError("README correctness start marker is missing")
    _, separator, after = remainder.partition(END)
    if not separator:
        raise ValueError("README correctness end marker is missing")
    table = render_table(
        terminal_rows(read_rows(julia_path)),
        terminal_rows(read_rows(python_path)),
    )
    readme_path.write_text(f"{before}{START}\n{table}\n{END}{after}")


def write_terminal(source_path, destination_path):
    rows = terminal_rows(read_rows(source_path))
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: update_correctness_readme.py "
            "README.md julia-correctness.csv python-correctness.csv"
        )
    update(*sys.argv[1:])
