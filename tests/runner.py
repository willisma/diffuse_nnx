from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TESTS), pattern='*_tests.py', top_level_dir=str(ROOT)
    )

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)