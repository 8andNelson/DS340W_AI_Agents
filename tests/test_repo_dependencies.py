import json
import tempfile
import unittest
from pathlib import Path

from src.orchestration import repo_dependencies as deps


class TestRequirementLines(unittest.TestCase):
    def test_kinds(self):
        cases = {
            "pandas==2.2.3": ("pypi", "pandas", "==2.2.3"),
            "Scikit_Learn>=1.0 ; python_version>'3.8'": ("pypi", "scikit-learn", ">=1.0"),
            "torch[cuda]": ("pypi", "torch", ""),
            "git+https://github.com/a/b.git": ("url", "", ""),
            "-e .": ("url", "", ""),
            "mypkg @ https://x.example/p.whl": ("url", "", ""),
            "--extra-index-url https://x.example": ("option", "", ""),
            "-r other.txt": ("option", "", ""),
        }
        for line, (kind, name, spec) in cases.items():
            parsed = deps.parse_requirement_line(line)
            self.assertEqual((parsed["kind"], parsed["name"], parsed["spec"]), (kind, name, spec), line)

    def test_comments_and_blank_lines_are_skipped(self):
        self.assertIsNone(deps.parse_requirement_line("# comment"))
        self.assertIsNone(deps.parse_requirement_line("   "))
        self.assertEqual(deps.parse_requirement_line("numpy  # fast arrays")["name"], "numpy")

    def test_pinned_version(self):
        self.assertEqual(deps.pinned_version("==2.2.3"), "2.2.3")
        self.assertEqual(deps.pinned_version(">=2.2"), "")


class TestImportedPackages(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_maps_imports_and_skips_stdlib_local_and_unknown(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "train.py").write_text(
            "import os, json\nimport numpy as np\nfrom sklearn.metrics import f1_score\nimport cv2\n"
            "from features import build\nfrom . import sibling\nimport mystery_lib\n")
        (self.root / "src" / "features.py").write_text("def build(): pass\n")
        (self.root / "nb.ipynb").write_text(json.dumps({"cells": [
            {"cell_type": "code", "source": ["!pip install xgboost\n", "if True:\n", "    %time x = 1\n",
                                             "    import xgboost\n"]}]}))
        packages, unknown = deps.imported_packages(self.root)
        self.assertEqual(packages, ["numpy", "opencv-python", "scikit-learn", "xgboost"])
        self.assertEqual(unknown, ["mystery_lib"])

    def test_venv_folders_are_ignored(self):
        (self.root / "venv" / "lib").mkdir(parents=True)
        (self.root / "venv" / "lib" / "x.py").write_text("import torch\n")
        self.assertEqual(deps.imported_packages(self.root), ([], []))


class TestMissingModule(unittest.TestCase):
    def test_reads_the_last_module_not_found(self):
        stderr = "Traceback...\nModuleNotFoundError: No module named 'sklearn.ensemble'\n"
        self.assertEqual(deps.missing_module(stderr), "sklearn")
        self.assertEqual(deps.missing_module("ValueError: x"), "")


if __name__ == "__main__":
    unittest.main()
