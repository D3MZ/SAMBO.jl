import csv
import sys


def read_results(path):
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        (row["problem"], row["algorithm"], int(row["seed"])): row
        for row in rows
    }


def main(julia_path, python_path):
    julia = read_results(julia_path)
    python = read_results(python_path)
    if julia.keys() != python.keys():
        missing_julia = sorted(python.keys() - julia.keys())
        missing_python = sorted(julia.keys() - python.keys())
        raise SystemExit(
            f"correctness matrices differ: missing Julia={missing_julia}, "
            f"missing Python={missing_python}"
        )

    failures = []
    for key in sorted(julia):
        for runtime, row in (("Julia", julia[key]), ("Python", python[key])):
            if row["success"].lower() != "true":
                failures.append(
                    f"{runtime} {key}: minimum={row['minimum']} target={row['target']}"
                )
            if int(row["evaluations"]) > int(row["budget"]):
                failures.append(
                    f"{runtime} {key}: evaluations={row['evaluations']} "
                    f"budget={row['budget']}"
                )
    if failures:
        raise SystemExit("correctness matrix failed:\n" + "\n".join(failures))

    print(f"validated {len(julia)} paired correctness cases")


if __name__ == "__main__":
    main(*sys.argv[1:])
