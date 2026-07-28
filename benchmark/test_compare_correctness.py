import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_correctness.py")
SPEC = importlib.util.spec_from_file_location("compare_correctness", MODULE_PATH)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare)

FIELDS = (
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


def result_row(runtime, rotation_id=1, trial_id=1, **overrides):
    row = {
        "runtime": runtime,
        "runtime_version": "test-runtime",
        "source_commit": "deadbeef",
        "python_sambo_version": "test-python-sambo",
        "problem": "sphere",
        "algorithm": "SMBO",
        "trial_id": trial_id,
        "rotation_id": rotation_id,
        "configuration_hash": (
            "python-sambo-1.25.2-matched-v7:SMBO:100:6-rotations"
        ),
        "initial_design_hash": f"design-{rotation_id}-{trial_id}",
        "initial_design_capability": "injected-counted-lhs",
        "sampling_stream_hash": "none",
        "sampling_stream_capability": "not-applicable",
        "budget": 100,
        "evaluations": 100,
        "minimum": 0.1,
        "optimum": 0.0,
        "quality_threshold": 0.25,
        "retcode": "success",
    }
    row.update(overrides)
    return row


def trial_rows(runtime, minima, rotations=range(1, 7), **overrides):
    rows = []
    for rotation_id in rotations:
        for index, value in enumerate(minima, start=1):
            row_overrides = dict(overrides)
            row_overrides.setdefault("minimum", value)
            rows.append(
                result_row(
                    runtime,
                    rotation_id=rotation_id,
                    trial_id=index,
                    **row_overrides,
                )
            )
    return rows


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, rows):
        path = self.root / name
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def paired_paths(self, julia_rows, python_rows):
        return (
            self.write("julia.csv", julia_rows),
            self.write("python.csv", python_rows),
        )

    def test_quality_threshold_passes_when_julia_is_better(self):
        paths = self.paired_paths(
            trial_rows("Julia", [0.01] * 10),
            trial_rows("Python", [0.1] * 10),
        )
        compare.main(*paths)

    def test_quality_threshold_fails_when_julia_is_materially_worse(self):
        paths = self.paired_paths(
            trial_rows("Julia", [1.0] * 10),
            trial_rows("Python", [0.1] * 10),
        )
        with self.assertRaisesRegex(SystemExit, "quality threshold"):
            compare.main(*paths)

    def test_row_pass_count_is_rotation_level(self):
        julia_rows = trial_rows("Julia", [0.1] * 10)
        for row in julia_rows:
            if row["rotation_id"] == 6:
                row["minimum"] = 1.0
        paths = self.paired_paths(
            julia_rows,
            trial_rows("Python", [0.1] * 10),
        )
        failures, summaries = compare.quality_threshold_gate(
            compare.read_results(paths[0]),
            compare.read_results(paths[1]),
        )
        self.assertTrue(failures)
        self.assertEqual(
            summaries[("sphere", "SMBO")]["passed_rotations"],
            5,
        )

    def test_relative_regret_is_not_clipped(self):
        julia = result_row("Julia", minimum=10.0)
        python = result_row("Python", minimum=0.1)
        key = ("sphere", "SMBO", 1, 1, 100)
        self.assertAlmostEqual(
            compare.normalized_log_regret(julia, key)
            - compare.normalized_log_regret(python, key),
            2.0,
        )

    def test_at_least_ten_trials_are_required(self):
        paths = self.paired_paths(
            trial_rows("Julia", [0.1] * 9),
            trial_rows("Python", [0.1] * 9),
        )
        with self.assertRaisesRegex(SystemExit, "at least 10 trials"):
            compare.main(*paths)

    def test_all_six_rotations_are_required(self):
        paths = self.paired_paths(
            trial_rows("Julia", [0.1] * 10, rotations=range(1, 6)),
            trial_rows("Python", [0.1] * 10, rotations=range(1, 6)),
        )
        with self.assertRaisesRegex(SystemExit, "requires 6 rotations"):
            compare.main(*paths)

    def test_numerical_regret_floor_ignores_sub_tolerance_noise(self):
        julia = result_row("Julia", minimum=1e-9)
        python = result_row("Python", minimum=1e-12)
        key = ("sphere", "SMBO", 1, 1, 100)
        self.assertEqual(
            compare.normalized_log_regret(julia, key),
            compare.normalized_log_regret(python, key),
        )

    def test_runtime_metadata_must_match(self):
        for field in compare.MATCHED_METADATA:
            with self.subTest(field=field):
                paths = self.paired_paths(
                    trial_rows("Julia", [0.1] * 10),
                    trial_rows("Python", [0.1] * 10, **{field: "different"}),
                )
                with self.assertRaisesRegex(SystemExit, field):
                    compare.main(*paths)

    def test_trial_matrices_must_match(self):
        paths = self.paired_paths(
            trial_rows("Julia", [0.1] * 10),
            trial_rows("Python", [0.1] * 10, rotations=range(1, 6)),
        )
        with self.assertRaisesRegex(SystemExit, "matrices differ"):
            compare.main(*paths)

    def test_only_one_terminal_row_per_trial_is_accepted(self):
        rows = trial_rows("Julia", [0.1] * 10)
        rows.append(dict(rows[0], evaluations=99))
        path = self.write("duplicate.csv", rows)
        with self.assertRaisesRegex(SystemExit, "duplicate comparison key"):
            compare.read_results(path)

    def test_rotation_uses_median_of_paired_trial_differences(self):
        julia = [1.0] * 6 + [1e10] * 4
        python = [1.0] * 5 + [1e10] * 5
        paths = self.paired_paths(
            trial_rows("Julia", julia),
            trial_rows("Python", python),
        )
        failures, summaries = compare.quality_threshold_gate(
            compare.read_results(paths[0]),
            compare.read_results(paths[1]),
        )
        self.assertFalse(failures)
        self.assertEqual(
            summaries[("sphere", "SMBO")]["worst_difference"],
            0.0,
        )

    def test_empty_and_malformed_files_are_rejected(self):
        empty = self.root / "empty.csv"
        empty.write_text("")
        header_only = self.write("header.csv", [])
        with self.assertRaisesRegex(SystemExit, "missing CSV header"):
            compare.read_results(empty)
        with self.assertRaisesRegex(SystemExit, "no correctness result rows"):
            compare.read_results(header_only)

    def test_nonfinite_values_and_dirty_provenance_are_rejected(self):
        for override, message in (
            ({"minimum": "nan"}, "nonfinite"),
            ({"source_commit": "deadbeef-dirty"}, "non-reproducible"),
        ):
            with self.subTest(override=override):
                path = self.write(
                    "invalid.csv",
                    trial_rows("Julia", [0.1] * 10, **override),
                )
                with self.assertRaisesRegex(SystemExit, message):
                    compare.read_results(path)


if __name__ == "__main__":
    unittest.main()
