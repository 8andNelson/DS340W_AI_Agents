import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents import master_agent
from src.orchestration import run_store


@pytest.fixture(autouse=True)
def _tmp(tmp_path, request):
    # conftest already points run_store.DB_FILE at a per-test temp database.
    if request.instance is not None:
        request.instance.tmp = tmp_path


class TestRunStore(unittest.TestCase):
    def _write(self, rel: str, text: str) -> Path:
        path = self.tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_run_lifecycle_and_parent_paper(self):
        run_id = run_store.start_run("fraud detection", llm_provider="openrouter")
        self.assertTrue(run_id)
        run_store.finish_run(run_id, "success", "EXPERIMENTATION",
                             parent_paper={"title": "Parent", "doi": "10.1/x"}, result={"success": True})

        run = run_store.get_run(run_id)
        self.assertEqual(run["status"], "success")
        self.assertEqual(run["final_phase"], "EXPERIMENTATION")
        self.assertEqual(run["parent_paper_title"], "Parent")
        self.assertEqual(run["parent_paper_doi"], "10.1/x")
        self.assertEqual(run["project_git_commit"], "test-commit")
        self.assertEqual(run_store.list_runs()[0]["run_id"], run_id)

    def test_stage_output_and_path_artifacts_are_stored(self):
        master = self._write("data/processed/master_dataset.csv", "a,b\n1,2\n")
        raw = self._write("data/raw/ds.csv", "x\n1\n")
        output = {
            "master_csv_path": str(master),
            "missing_path": str(self.tmp / "nope.csv"),
            "selected_datasets": [{"name": "ds", "local_csv_path": str(raw)}],
        }
        run_id = run_store.start_run("t")
        run_store.record_stage(run_id, "Cleaning Agent", 1, "Approved", output, "report text")

        self.assertEqual(run_store.get_stage_output(run_id, "Cleaning Agent"), output)
        run = run_store.get_run(run_id)
        self.assertEqual(run["stages"][0]["decision"], "Approved")
        kinds = {a["kind"]: a for a in run["artifacts"]}
        self.assertEqual(set(kinds), {"master_csv_path", "local_csv_path"})
        self.assertEqual(run_store.read_artifact(kinds["master_csv_path"]["id"]), master.read_bytes())

    def test_replication_dir_keeps_scripts_and_skips_venv_and_data(self):
        rep = self.tmp / "replication" / "paper"
        self._write("replication/paper/reconstructed/train.py", "print('hi')\n")
        self._write("replication/paper/ENVIRONMENT.md", "# env\n")
        self._write("replication/paper/venv/lib/site.py", "x = 1\n")
        self._write("replication/paper/data/train.csv", "a\n1\n")

        paths = {p.name: kind for kind, p in run_store.collect_artifact_paths({"replication_dir": str(rep)})}
        self.assertEqual(paths, {"train.py": "script", "ENVIRONMENT.md": "source_file"})

    def test_identical_files_share_one_blob(self):
        a = self._write("one/train.py", "same\n")
        b = self._write("two/train.py", "same\n")
        run_id = run_store.start_run("t")
        run_store.record_artifact(run_id, "Replication Agent", "script", a)
        run_store.record_artifact(run_id, "Experiment Agent", "script", b)

        with run_store._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0], 1)

    def test_oversized_file_is_recorded_by_hash_only(self):
        big = self._write("big.csv", "x" * 100)
        run_id = run_store.start_run("t")
        with patch.object(run_store, "MAX_BLOB_BYTES", 10):
            run_store.record_artifact(run_id, "Data Agent", "local_csv_path", big)

        artifact = run_store.get_run(run_id)["artifacts"][0]
        self.assertEqual(artifact["content_stored"], 0)
        self.assertEqual(artifact["size_bytes"], 100)
        self.assertIsNone(run_store.read_artifact(artifact["id"]))

    def test_database_errors_fail_soft(self):
        with patch.object(run_store, "_connect", side_effect=run_store.sqlite3.OperationalError("locked")):
            self.assertIsNone(run_store.start_run("t"))
            self.assertIsNone(run_store.record_stage("rid", "A", 1, "Approved", {}))
            run_store.finish_run("rid", "success")  # must not raise

    def test_no_run_id_records_nothing(self):
        self.assertIsNone(run_store.record_stage(None, "A", 1, "Approved", {}))
        self.assertEqual(run_store.list_runs(), [])


class TestMasterAgentRecordsRuns(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(master_agent.slack_client, post_message=lambda *a, **k: None,
                                 post_error_message=lambda *a, **k: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_supervise_records_every_attempt(self):
        run_id = run_store.start_run("t")
        outputs = iter([{"n": 1}, {"n": 2}])
        checks = iter([["bad"], []])
        with patch.object(master_agent, "_current_run_id", run_id), \
             patch.object(master_agent, "_print_entry"):
            master_agent._supervise("Some Agent", lambda: next(outputs),
                                    lambda out: [master_agent._log_entry("Correction", "Some Agent", "x", "Retrying", "y")]
                                    if next(checks) else [],
                                    lambda out: f"report {out['n']}")

        stages = run_store.get_run(run_id)["stages"]
        self.assertEqual([(s["attempt"], s["decision"]) for s in stages], [(1, "Retrying"), (2, "Approved")])
        self.assertEqual(run_store.get_stage_output(run_id, "Some Agent"), {"n": 2})

    def test_run_pipeline_opens_and_closes_a_run(self):
        result = {"success": False, "phase": "DATA_DISCOVERY", "parent_paper": {"title": "P"}, "blockers": ["b"]}
        with patch.object(master_agent, "_run_stages", return_value=result):
            out = master_agent.run_pipeline("topic")

        run = run_store.get_run(out["run_id"])
        self.assertEqual(run["status"], "halted")
        self.assertEqual(run["final_phase"], "DATA_DISCOVERY")
        self.assertEqual(run["parent_paper_title"], "P")
        self.assertIsNone(master_agent._current_run_id)

    def test_run_pipeline_marks_crashed_run(self):
        with patch.object(master_agent, "_run_stages", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                master_agent.run_pipeline("topic")

        self.assertEqual(run_store.list_runs()[0]["status"], "crashed")
