import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("update_readme.py")
SPEC = importlib.util.spec_from_file_location("update_readme", MODULE_PATH)
update_readme = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_readme)


class BenchmarkReadmeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
