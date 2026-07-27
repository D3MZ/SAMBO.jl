import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_correctness.py")
SPEC = importlib.util.spec_from_file_location("compare_correctness", MODULE_PATH)
compare_correctness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_correctness)

FIELDS = (
    "runtime",
    "runtime_version",
    "source_commit",
    "python_sambo_version",
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
    "quality_target",
    "required_hit_rate",
    "noninferiority_margin",
    "success",
)


def result_row(runtime, **overrides):
    row = {
        "runtime": runtime,
        "runtime_version": "test-runtime",
        "source_commit": "deadbeef",
        "python_sambo_version": "test-python-sambo",
        "problem": "sphere",
        "algorithm": "SMBO",
        "trial_id": 1,
        "configuration_hash": "fixture-v1",
        "initial_design_hash": "fixture-design",
        "initial_design_capability": "injected-x0-y0",
        "budget": 100,
        "evaluation": 100,
        "evaluations": 100,
        "minimum": 0.0,
        "optimum": 0.0,
        "target": 1.0,
        "quality_target": 1.0,
        "required_hit_rate": 1.0,
        "noninferiority_margin": 0.25,
        "success": "true",
    }
    row.update(overrides)
    return row


def write_results(path, rows):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def paired_paths(self, julia_rows, python_rows):
        julia_path = self.root / "julia.csv"
        python_path = self.root / "python.csv"
        write_results(julia_path, julia_rows)
        write_results(python_path, python_rows)
        return julia_path, python_path

    def test_comparator_rejects_divergent_runtime_values(self):
        paths = self.paired_paths(
            [result_row("Julia", minimum=0.0, target=200.0)],
            [result_row("Python", minimum=100.0, target=200.0)],
        )
        with self.assertRaisesRegex(SystemExit, "parity"):
            compare_correctness.main(*paths)

    def test_duplicate_case_keys_are_rejected(self):
        duplicate = result_row("Julia")
        paths = self.paired_paths(
            [duplicate, duplicate],
            [result_row("Python")],
        )
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            compare_correctness.main(*paths)

    def test_trace_checkpoints_are_compared(self):
        paths = self.paired_paths(
            [
                result_row("Julia", evaluation=1, evaluations=1),
                result_row("Julia", evaluation=2, evaluations=2),
            ],
            [result_row("Python", evaluation=2, evaluations=2)],
        )
        with self.assertRaisesRegex(SystemExit, "matrices differ"):
            compare_correctness.main(*paths)

    def test_quality_is_checked_at_the_terminal_checkpoint(self):
        paths = self.paired_paths(
            [
                result_row(
                    "Julia",
                    evaluation=1,
                    evaluations=1,
                    minimum=10.0,
                ),
                result_row(
                    "Julia",
                    evaluation=2,
                    evaluations=2,
                    minimum=0.0,
                ),
            ],
            [
                result_row(
                    "Python",
                    evaluation=1,
                    evaluations=1,
                    minimum=10.0,
                ),
                result_row(
                    "Python",
                    evaluation=2,
                    evaluations=2,
                    minimum=0.0,
                ),
            ],
        )
        compare_correctness.main(*paths)

    def test_runtime_metadata_must_match(self):
        fields = (
            "budget",
            "optimum",
            "target",
            "quality_target",
            "required_hit_rate",
            "noninferiority_margin",
            "configuration_hash",
            "initial_design_hash",
            "initial_design_capability",
            "source_commit",
        )
        for field in fields:
            with self.subTest(field=field):
                paths = self.paired_paths(
                    [result_row("Julia", evaluations=1)],
                    [result_row("Python", evaluations=1, **{field: 2})],
                )
                with self.assertRaisesRegex(SystemExit, field):
                    compare_correctness.main(*paths)

    def test_provenance_and_configuration_are_required(self):
        for field in (
            "runtime_version",
            "source_commit",
            "configuration_hash",
            "initial_design_hash",
            "initial_design_capability",
        ):
            with self.subTest(field=field):
                paths = self.paired_paths(
                    [result_row("Julia", **{field: ""})],
                    [result_row("Python", **{field: ""})],
                )
                with self.assertRaisesRegex(SystemExit, field):
                    compare_correctness.main(*paths)

    def test_nonfinite_values_are_rejected(self):
        for minimum in ("nan", "inf", "-inf"):
            with self.subTest(minimum=minimum):
                paths = self.paired_paths(
                    [result_row("Julia", minimum=minimum)],
                    [result_row("Python")],
                )
                with self.assertRaisesRegex(SystemExit, "nonfinite"):
                    compare_correctness.main(*paths)

    def test_declared_success_does_not_override_failed_target(self):
        paths = self.paired_paths(
            [result_row("Julia", minimum=10.0, target=1.0, success="true")],
            [result_row("Python", minimum=10.0, target=1.0, success="true")],
        )
        with self.assertRaisesRegex(SystemExit, "target"):
            compare_correctness.main(*paths)

    def test_declared_failure_does_not_override_met_target(self):
        paths = self.paired_paths(
            [result_row("Julia", minimum=0.0, target=1.0, success="false")],
            [result_row("Python", minimum=0.0, target=1.0, success="false")],
        )
        compare_correctness.main(*paths)

    def test_paired_bootstrap_interval_and_noninferiority(self):
        lower, upper = compare_correctness.paired_bootstrap_interval(
            [0.1] * 20,
            samples=200,
            seed=7,
        )
        self.assertAlmostEqual(lower, 0.1)
        self.assertAlmostEqual(upper, 0.1)

        paths = self.paired_paths(
            [result_row("Julia", minimum=10.0, target=20.0)],
            [result_row("Python", minimum=0.1, target=20.0)],
        )
        julia = compare_correctness.read_results(paths[0])
        python = compare_correctness.read_results(paths[1])
        failures = compare_correctness.noninferiority_gate(
            julia,
            python,
            samples=200,
        )
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
