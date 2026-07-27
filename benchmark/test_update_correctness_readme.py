import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("update_correctness_readme.py")
SPEC = importlib.util.spec_from_file_location(
    "update_correctness_readme",
    MODULE_PATH,
)
update_correctness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_correctness)


class CorrectnessReadmeTests(unittest.TestCase):
    def test_terminal_rows_and_table(self):
        rows = [
            {
                "problem": "hartmann6",
                "algorithm": "SMBO",
                "trial_id": "1",
                "evaluation": "1",
                "minimum": "-2.0",
                "optimum": "-3.322368011415515",
                "target": "-2.8",
            },
            {
                "problem": "hartmann6",
                "algorithm": "SMBO",
                "trial_id": "1",
                "evaluation": "2",
                "minimum": "-3.0",
                "optimum": "-3.322368011415515",
                "target": "-2.8",
            },
            {
                "problem": "hartmann6",
                "algorithm": "SMBO",
                "trial_id": "2",
                "evaluation": "1",
                "minimum": "-2.9",
                "optimum": "-3.322368011415515",
                "target": "-2.8",
            },
        ]
        terminal = update_correctness.terminal_rows(rows)
        self.assertEqual(len(terminal), 2)
        table = update_correctness.render_table(terminal, terminal)
        self.assertIn("Hartmann-6 (−3.322; ≤ −2.8)", table)
        self.assertIn("| SMBO |", table)
        self.assertEqual(table.count("−2.95; 2/2"), 2)

    def test_checked_in_readme_has_generated_section(self):
        root = MODULE_PATH.parent.parent
        readme = (root / "README.md").read_text()
        self.assertEqual(readme.count(update_correctness.START), 1)
        self.assertEqual(readme.count(update_correctness.END), 1)
        checked_in = readme.partition(update_correctness.START)[2].partition(
            update_correctness.END,
        )[0].strip()
        generated = update_correctness.render_table(
            update_correctness.terminal_rows(
                update_correctness.read_rows(
                    root / "benchmark/results/correctness_julia.csv",
                ),
            ),
            update_correctness.terminal_rows(
                update_correctness.read_rows(
                    root / "benchmark/results/correctness_python.csv",
                ),
            ),
        )
        self.assertEqual(checked_in, generated)


if __name__ == "__main__":
    unittest.main()
