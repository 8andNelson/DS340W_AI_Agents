import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents import master_agent


def make_row(**overrides) -> dict:
    row = {
        "title": "Some Paper",
        "publication_year": 2023,
        "peer_reviewed": True,
        "research_topic": "Some Venue",
        "dataset_availability": "Named + sourced",
        "dataset_source": "Official",
        "methodology_clarity": "High",
        "implementation_clarity": "High",
        "code_availability": False,
        "computing_resources_estimate": "Medium",
        "reproducibility_difficulty": "High",
        "reproducibility_estimated": False,
        "visuals": "tables, figures",
        "has_evaluation_metrics": True,
        "upstream_parent_paper_score": 8,
        "expected_difficulty": "Low",
        "composite_score": 10.0,
    }
    row.update(overrides)
    return row


def make_recommendation(status="OK", pool_source="verified", pool_size=5, row=None,
                         deciding_factor=False, policy_violation=False, chosen=None) -> dict:
    row = row or make_row()
    chosen = chosen if chosen is not None else {
        "title": row["title"],
        "methodology": "Full methodology description.",
        "results_summary": "Accuracy 94%.",
    }
    return {
        "status": status,
        "candidate_pool_source": pool_source,
        "candidate_pool_size": pool_size,
        "degraded": pool_source != "verified",
        "degraded_reason": "" if pool_source == "verified" else "some reason",
        "comparison_table": [row],
        "code_availability_audit": {
            "count_with_code": 2 if policy_violation else 0,
            "titles_with_code": [],
            "policy_violation": policy_violation,
            "note": "",
        },
        "recommended_parent_paper": chosen,
        "runner_ups": [],
        "justification": "Because it's the best.",
        "expected_difficulty": row["expected_difficulty"],
        "code_availability_deciding_factor": deciding_factor,
    }


class MasterAgentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._log_patch = patch.object(master_agent, "LOG_FILE", Path(self._tmpdir.name) / "master_agent_log.json")
        self._report_patch = patch.object(master_agent, "REPORT_DIR", Path(self._tmpdir.name) / "agent_reports")
        self._log_patch.start()
        self._report_patch.start()

    def tearDown(self):
        self._log_patch.stop()
        self._report_patch.stop()
        self._tmpdir.cleanup()

    def _log_entries(self) -> list:
        if not master_agent.LOG_FILE.exists():
            return []
        return json.loads(master_agent.LOG_FILE.read_text())


class TestValidationAndSelectionChecks(MasterAgentTestCase):
    """_check_validation_and_selection is Master Agent's independent review
    of validation_agent.run()'s Parent Paper recommendation."""

    def test_approves_clean_recommendation(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation())
        self.assertEqual(blockers, [])

    def test_rejects_no_candidates(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation(status="NO_CANDIDATES"))
        self.assertTrue(blockers)
        self.assertEqual(blockers[0]["target"], "Validation Agent")

    def test_rejects_degraded_pool_source(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation(pool_source="all_non_rejected"))
        issues = [b["issue"] for b in blockers]
        self.assertTrue(any("all_non_rejected" in i for i in issues))

    def test_rejects_too_few_qualifying_papers(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation(pool_size=3))
        issues = [b["issue"] for b in blockers]
        self.assertTrue(any("Only 3 qualifying" in i for i in issues))

    def test_rejects_non_peer_reviewed_paper(self):
        blockers = master_agent._check_validation_and_selection(
            make_recommendation(row=make_row(peer_reviewed=False))
        )
        issues = [b["issue"] for b in blockers]
        self.assertTrue(any("not confirmed peer-reviewed" in i for i in issues))

    def test_rejects_paper_with_no_replication_detail(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation(
            row=make_row(dataset_availability="Not specified"),
            chosen={"title": "Vague Paper", "methodology": "", "results_summary": ""},
        ))
        issues = [b["issue"] for b in blockers]
        self.assertTrue(any("insufficient detail" in i for i in issues))

    def test_rejects_when_code_availability_was_deciding_factor_and_policy_violated(self):
        blockers = master_agent._check_validation_and_selection(make_recommendation(
            row=make_row(code_availability=True),
            deciding_factor=True,
            policy_violation=True,
        ))
        issues = [b["issue"] for b in blockers]
        self.assertTrue(any("code convenience" in i or "code availability" in i for i in issues))

    def test_does_not_reject_solely_because_paper_has_code(self):
        blockers = master_agent._check_validation_and_selection(
            make_recommendation(row=make_row(code_availability=True), deciding_factor=False)
        )
        self.assertEqual(blockers, [])


class TestIntakeAndResearchChecks(MasterAgentTestCase):
    def test_check_intake_rejects_empty_structured_request(self):
        blockers = master_agent._check_intake({"domain": "", "keywords": [], "search_queries": []})
        self.assertTrue(blockers)
        self.assertEqual(blockers[0]["target"], "Intake Agent")

    def test_check_intake_approves_populated_request(self):
        blockers = master_agent._check_intake({
            "domain": "fraud detection", "keywords": ["fraud"], "search_queries": ["credit card fraud ML"],
        })
        self.assertEqual(blockers, [])

    def test_check_research_rejects_too_few_qualifying_papers(self):
        papers = [{"title": "P1", "peer_reviewed": True, "year": 2023}]
        blockers = master_agent._check_research(papers)
        self.assertTrue(blockers)
        self.assertEqual(blockers[0]["target"], "Research Agent")

    def test_check_research_rejects_empty_list(self):
        blockers = master_agent._check_research([])
        self.assertTrue(blockers)

    def test_check_research_approves_five_qualifying_papers(self):
        papers = [{"title": f"P{i}", "peer_reviewed": True, "year": 2023} for i in range(5)]
        blockers = master_agent._check_research(papers)
        self.assertEqual(blockers, [])


class TestSupervise(MasterAgentTestCase):
    def test_retries_once_then_succeeds(self):
        calls = []

        def run_fn():
            calls.append(1)
            return "good" if len(calls) > 1 else "bad"

        def checker_fn(output):
            if output == "good":
                return []
            return [master_agent._log_entry("Correction", "Test Agent", "bad output", "Rejected", "fix it")]

        output, blockers = master_agent._supervise("Test Agent", run_fn, checker_fn, lambda o: f"output was {o}")

        self.assertEqual(output, "good")
        self.assertEqual(blockers, [])
        self.assertEqual(len(calls), 2)

        entries = self._log_entries()
        decisions = [e["decision"] for e in entries]
        self.assertIn("Retrying", decisions)
        self.assertEqual(decisions[-1], "Approved")

    def test_exhausts_retries_then_rejects(self):
        calls = []

        def run_fn():
            calls.append(1)
            return "bad"

        def checker_fn(output):
            return [master_agent._log_entry("Correction", "Test Agent", "always bad", "Rejected", "fix it")]

        output, blockers = master_agent._supervise("Test Agent", run_fn, checker_fn, lambda o: f"output was {o}")

        self.assertTrue(blockers)
        self.assertEqual(len(calls), master_agent.MAX_ATTEMPTS)

        entries = self._log_entries()
        self.assertEqual(entries[-1]["decision"], "Rejected")
        self.assertNotIn("Approved", [e["decision"] for e in entries])

    def test_exception_in_run_fn_is_treated_as_a_failed_check(self):
        def run_fn():
            raise RuntimeError("boom")

        def checker_fn(output):
            return []  # never reached

        output, blockers = master_agent._supervise("Test Agent", run_fn, checker_fn, lambda o: "n/a")

        self.assertIsNone(output)
        self.assertTrue(blockers)
        entries = self._log_entries()
        self.assertTrue(any("boom" in e["issue"] for e in entries))

    def test_writes_a_report_for_every_attempt(self):
        calls = []

        def run_fn():
            calls.append(1)
            return "good" if len(calls) > 1 else "bad"

        def checker_fn(output):
            return [] if output == "good" else [
                master_agent._log_entry("Correction", "Test Agent", "bad", "Rejected", "fix it")
            ]

        master_agent._supervise("Test Agent", run_fn, checker_fn, lambda o: f"report for {o}")

        report_files = sorted(master_agent.REPORT_DIR.glob("test_agent_attempt*.txt"))
        self.assertEqual(len(report_files), 2)
        self.assertIn("report for bad", report_files[0].read_text())
        self.assertIn("report for good", report_files[1].read_text())


class TestWriteReport(MasterAgentTestCase):
    def test_write_report_creates_a_readable_file(self):
        path = master_agent._write_report("Some Agent", 1, "hello world")
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "hello world")
        self.assertTrue(path.parent == master_agent.REPORT_DIR)


class TestLogging(MasterAgentTestCase):
    def test_log_file_persists_entries_across_supervised_runs(self):
        master_agent._supervise(
            "Agent A", lambda: "ok", lambda o: [], lambda o: "report",
        )
        master_agent._supervise(
            "Agent B",
            lambda: "bad",
            lambda o: [master_agent._log_entry("Correction", "Agent B", "bad", "Rejected", "fix it")],
            lambda o: "report",
        )

        entries = self._log_entries()
        decisions = [e["decision"] for e in entries]
        self.assertIn("Approved", decisions)
        self.assertIn("Rejected", decisions)


if __name__ == "__main__":
    unittest.main()
