import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("update_readme.py")
SPEC = importlib.util.spec_from_file_location("update_readme", MODULE_PATH)
update_readme = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_readme)


class BenchmarkReadmeTests(unittest.TestCase):
    @staticmethod
    def quality_summaries():
        summaries = {}
        for algorithm in update_readme.ALGORITHMS:
            for problem, _ in update_readme.PROBLEMS:
                summaries[(problem, algorithm)] = {
                    "optimum": 0.0,
                    "julia_median": 10.0,
                    "python_median": 10.0,
                    "passed_rotations": 6,
                    "rotations": 6,
                }
        return summaries

    def test_table_is_generated_from_runtime_rows(self):
        julia = {
            "SCE-UA": {"time_ms": "0.025"},
            "SMBO": {"time_ms": "10"},
            "SHGO": {"time_ms": "0.05"},
        }
        python = {
            "SCE-UA": {"time_ms": "1.5"},
            "SMBO": {"time_ms": "400"},
            "SHGO": {"time_ms": "2"},
        }
        table = update_readme.render_table(julia, python)
        self.assertIn("| SCE-UA | 0.0250 ms | 1.500 ms | 60.0× faster |", table)
        self.assertIn("| SMBO | 10.000 ms | 400.000 ms | 40.0× faster |", table)
        self.assertIn("| SHGO | 0.0500 ms | 2.000 ms | 40.0× faster |", table)

    def test_checked_in_readme_has_generated_section(self):
        readme = (MODULE_PATH.parent.parent / "README.md").read_text()
        self.assertEqual(readme.count(update_readme.START), 1)
        self.assertEqual(readme.count(update_readme.END), 1)
        self.assertEqual(readme.count(update_readme.QUALITY_START), 1)
        self.assertEqual(readme.count(update_readme.QUALITY_END), 1)

    def test_checked_in_table_matches_saved_results(self):
        readme = update_readme.DEFAULT_README.read_text()
        generated = update_readme.render_table(
            update_readme.read_rows(update_readme.DEFAULT_JULIA),
            update_readme.read_rows(update_readme.DEFAULT_PYTHON),
        )
        checked_in = (
            readme.split(update_readme.START, 1)[1]
            .split(update_readme.END, 1)[0]
            .strip()
        )
        self.assertEqual(checked_in, generated)

    def test_quality_table_uses_requested_columns_and_rotation_counts(self):
        summaries = {}
        for algorithm in update_readme.ALGORITHMS:
            for problem, _ in update_readme.PROBLEMS:
                summaries[(problem, algorithm)] = {
                    "optimum": -3.322368,
                    "julia_median": -3.2,
                    "python_median": -3.1,
                    "passed_rotations": 6,
                    "rotations": 6,
                }
        table = update_readme.render_quality_table(summaries)
        self.assertIn(
            "| Algorithm | Problem | Known optimum | Julia median | "
            "Python median | Within threshold |",
            table,
        )
        self.assertIn(
            "| SCE-UA | Hartmann-6 | −3.32237 | −3.2 | −3.1 | 6/6 |",
            table,
        )
        self.assertEqual(table.count("6/6"), 9)

    def test_quality_section_can_be_replaced(self):
        source = (
            f"before\n{update_readme.QUALITY_START}\nold\n"
            f"{update_readme.QUALITY_END}\nafter\n"
        )
        result = update_readme.replace_section(
            source,
            update_readme.QUALITY_START,
            update_readme.QUALITY_END,
            "new",
        )
        self.assertEqual(
            update_readme.checked_section(
                result,
                update_readme.QUALITY_START,
                update_readme.QUALITY_END,
            ),
            "new",
        )

    def test_quality_check_allows_small_platform_drift(self):
        summaries = self.quality_summaries()
        table = update_readme.render_quality_table(summaries).replace(
            "| SCE-UA | Hartmann-6 | 0 | 10 | 10 | 6/6 |",
            "| SCE-UA | Hartmann-6 | 0 | 10.5 | 9.6 | 6/6 |",
        )
        update_readme.check_quality_table(table, summaries)

    def test_quality_check_rejects_materially_stale_median(self):
        summaries = self.quality_summaries()
        table = update_readme.render_quality_table(summaries).replace(
            "| SCE-UA | Hartmann-6 | 0 | 10 | 10 | 6/6 |",
            "| SCE-UA | Hartmann-6 | 0 | 20 | 10 | 6/6 |",
        )
        with self.assertRaisesRegex(SystemExit, "median is stale"):
            update_readme.check_quality_table(table, summaries)

    def test_quality_check_enforces_optimum_and_pass_count(self):
        summaries = self.quality_summaries()
        table = update_readme.render_quality_table(summaries)
        for stale, message in (
            (
                table.replace(
                    "| SCE-UA | Hartmann-6 | 0 |",
                    "| SCE-UA | Hartmann-6 | 1 |",
                ),
                "optimum is stale",
            ),
            (
                table.replace(
                    "| SCE-UA | Hartmann-6 | 0 | 10 | 10 | 6/6 |",
                    "| SCE-UA | Hartmann-6 | 0 | 10 | 10 | 5/6 |",
                ),
                "pass count is stale",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SystemExit, message):
                    update_readme.check_quality_table(stale, summaries)


if __name__ == "__main__":
    unittest.main()
