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
    """
    Every test here runs with Slack posting mocked out and "unconfigured"
    by default -- master_agent._supervise/_print_entry/check_agent_requests
    all call through src.agents.master_agent.slack_client, and the real
    .env in this repo has live Slack credentials, so leaving these
    unmocked would post real messages to the real channel on every test
    run. Tests that specifically exercise Slack behavior override these
    patches locally.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._log_patch = patch.object(master_agent, "LOG_FILE", Path(self._tmpdir.name) / "master_agent_log.json")
        self._report_patch = patch.object(master_agent, "REPORT_DIR", Path(self._tmpdir.name) / "agent_reports")
        self._slack_post_patch = patch.object(master_agent.slack_client, "post_message", return_value=None)
        self._slack_error_post_patch = patch.object(master_agent.slack_client, "post_error_message", return_value=None)
        self._slack_configured_patch = patch.object(master_agent.slack_client, "is_configured", return_value=False)
        self._log_patch.start()
        self._report_patch.start()
        self.mock_slack_post = self._slack_post_patch.start()
        self.mock_slack_error_post = self._slack_error_post_patch.start()
        self._slack_configured_patch.start()

    def tearDown(self):
        self._log_patch.stop()
        self._report_patch.stop()
        self._slack_post_patch.stop()
        self._slack_error_post_patch.stop()
        self._slack_configured_patch.stop()
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


def make_dataset_candidate(**overrides) -> dict:
    candidate = {
        "name": "Some Dataset",
        "matches_parent_paper": True,
        "match_explanation": "Confirmed via official page.",
        "source": "Kaggle",
        "source_url": "https://kaggle.com/some-dataset",
        "access_method": "direct download",
        "license": "CC0",
        "format": "CSV",
        "entry_count": 5000,
        "verified": True,
        "verification_method": "brave_search",
        "notes": "",
    }
    candidate.update(overrides)
    return candidate


def make_data_result(**overrides) -> dict:
    selected = overrides.pop("selected_datasets", [make_dataset_candidate()])
    total = overrides.pop("total_entries", sum(d.get("entry_count", 0) for d in selected))
    result = {
        "status": "OK" if total >= 10000 else ("DEGRADED" if selected else "NOT_FOUND"),
        "target_entries": 10000,
        "total_entries": total,
        "target_met": total >= 10000,
        "datasets_considered": len(selected),
        "selected_datasets": selected,
        "master_dataset": selected[0] if len(selected) == 1 else None,
        "is_combined": len(selected) > 1,
        "warnings": [],
        "notes": "",
    }
    result.update(overrides)
    return result


def make_cleaning_result(**overrides) -> dict:
    merged = overrides.pop("datasets_merged", 1)
    rows_total = overrides.pop("rows_total", 12000 if merged else 0)
    result = {
        "status": "OK" if merged else "NOT_FOUND",
        "master_csv_path": "data/processed/master_dataset.csv" if merged else "",
        "datasets_attempted": merged,
        "datasets_merged": merged,
        "rows_total": rows_total,
        "columns_total": 5 if merged else 0,
        "target_rows": 10000,
        "target_met": rows_total >= 10000,
        "conflicts": [],
        "notes": "",
    }
    result.update(overrides)
    return result


class TestDataChecks(MasterAgentTestCase):
    """_check_data is Master Agent's independent review of data_agent.run()'s
    dataset pool -- it only blocks when zero usable datasets were found."""

    def test_approves_result_that_met_the_target(self):
        blockers = master_agent._check_data(make_data_result(
            selected_datasets=[make_dataset_candidate(entry_count=12000)],
        ))
        self.assertEqual(blockers, [])

    def test_approves_degraded_result_with_at_least_one_dataset(self):
        blockers = master_agent._check_data(make_data_result(
            selected_datasets=[make_dataset_candidate(entry_count=3000)],
        ))
        self.assertEqual(blockers, [])

    def test_rejects_zero_datasets(self):
        blockers = master_agent._check_data(make_data_result(
            selected_datasets=[], warnings=["Parent Paper does not state a dataset."],
        ))
        self.assertTrue(blockers)
        self.assertEqual(blockers[0]["target"], "Data Agent")


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


def make_validation_result(ranked_pool: list) -> dict:
    return {
        "status": "OK",
        "candidate_pool_source": "verified",
        "candidate_pool_size": len(ranked_pool),
        "degraded": False,
        "degraded_reason": "",
        "comparison_table": [],
        "ranked_pool": ranked_pool,
        "code_availability_audit": {"count_with_code": 0, "titles_with_code": [], "policy_violation": False, "note": ""},
        "recommended_parent_paper": ranked_pool[0],
        "all": ranked_pool,
        "runner_ups": [],
        "justification": "Because it's the best.",
        "expected_difficulty": "Low",
        "code_availability_deciding_factor": False,
    }


class TestDataAgentPipelineWiring(MasterAgentTestCase):
    """run_pipeline's Stage 4: Data Agent runs right after Parent Paper
    approval, caches the ranked pool, and only blocks on zero datasets."""

    def _run(self, data_agent_result: dict, cleaning_agent_result: dict = None):
        ranked_pool = [
            {"title": "Parent Paper", "dataset": "DS1"},
            {"title": "Runner Up", "dataset": "DS2"},
        ]
        validation_result = make_validation_result(ranked_pool)
        cleaning_agent_result = cleaning_agent_result or make_cleaning_result()

        with patch.object(master_agent, "_check_intake", return_value=[]), \
             patch.object(master_agent, "_check_research", return_value=[]), \
             patch.object(master_agent, "_check_validation_and_selection", return_value=[]), \
             patch("src.agents.intake_agent.run", return_value={
                 "domain": "fraud", "keywords": ["fraud"], "search_queries": ["fraud ML"],
             }), \
             patch("src.agents.research_agent.run", return_value=[
                 {"title": "Parent Paper", "peer_reviewed": True, "year": 2023}
             ]), \
             patch("src.agents.validation_agent.run", return_value=validation_result), \
             patch("src.agents.data_agent.run", return_value=data_agent_result) as data_run, \
             patch("src.agents.cleaning_agent.run", return_value=cleaning_agent_result), \
             patch("src.orchestration.state_manager.update_state") as update_state, \
             patch("src.orchestration.dataset_cache.cache_validated_papers") as cache_papers:
            result = master_agent.run_pipeline("credit card fraud detection")

        return result, ranked_pool, data_run, update_state, cache_papers

    def test_advances_to_data_discovery_on_success(self):
        data_result = make_data_result(selected_datasets=[make_dataset_candidate(entry_count=12000)])
        result, ranked_pool, data_run, update_state, cache_papers = self._run(data_result)

        self.assertTrue(result["success"])
        self.assertEqual(result["dataset_result"], data_result)
        cache_papers.assert_called_once_with(ranked_pool)
        data_run.assert_called_once_with(ranked_pool[0], ranked_pool)

    def test_halts_on_zero_datasets(self):
        data_result = make_data_result(selected_datasets=[], warnings=["nothing found"])
        result, ranked_pool, data_run, update_state, cache_papers = self._run(data_result)

        self.assertFalse(result["success"])
        self.assertEqual(result["phase"], "DATA_DISCOVERY")
        self.assertTrue(result["blockers"])
        self.assertEqual(result["blockers"][0]["target"], "Data Agent")


class TestCleaningAgentPipelineWiring(MasterAgentTestCase):
    """run_pipeline's Stage 5: Cleaning Agent runs right after Data Agent
    succeeds, is handed exactly dataset_result['selected_datasets'], and
    only blocks on zero merged datasets."""

    _run = TestDataAgentPipelineWiring._run

    def test_advances_to_data_cleaning_on_success(self):
        data_result = make_data_result(selected_datasets=[make_dataset_candidate(entry_count=12000)])
        cleaning_result = make_cleaning_result(datasets_merged=1)

        with patch.object(master_agent, "_check_intake", return_value=[]), \
             patch.object(master_agent, "_check_research", return_value=[]), \
             patch.object(master_agent, "_check_validation_and_selection", return_value=[]), \
             patch("src.agents.intake_agent.run", return_value={
                 "domain": "fraud", "keywords": ["fraud"], "search_queries": ["fraud ML"],
             }), \
             patch("src.agents.research_agent.run", return_value=[
                 {"title": "Parent Paper", "peer_reviewed": True, "year": 2023}
             ]), \
             patch("src.agents.validation_agent.run", return_value=make_validation_result([
                 {"title": "Parent Paper", "dataset": "DS1"},
             ])), \
             patch("src.agents.data_agent.run", return_value=data_result), \
             patch("src.agents.cleaning_agent.run", return_value=cleaning_result) as cleaning_run, \
             patch("src.orchestration.state_manager.update_state"), \
             patch("src.orchestration.dataset_cache.cache_validated_papers"):
            result = master_agent.run_pipeline("credit card fraud detection")

        self.assertTrue(result["success"])
        self.assertEqual(result["phase"], "DATA_CLEANING")
        self.assertEqual(result["cleaning_result"], cleaning_result)
        cleaning_run.assert_called_once_with(data_result["selected_datasets"])

    def test_halts_when_nothing_could_be_merged(self):
        data_result = make_data_result(selected_datasets=[make_dataset_candidate(entry_count=12000)])
        cleaning_result = make_cleaning_result(
            datasets_merged=0,
            conflicts=[{"dataset": "Some Dataset", "link": "https://kaggle.com/x", "reason": "no local CSV path"}],
        )
        result, *_ = self._run(data_result, cleaning_result)

        self.assertFalse(result["success"])
        self.assertEqual(result["phase"], "DATA_CLEANING")
        self.assertTrue(result["blockers"])
        self.assertEqual(result["blockers"][0]["target"], "Cleaning Agent")


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


class TestSlackPosting(MasterAgentTestCase):
    """_supervise posts an agent's own started/finished messages;
    _print_entry posts Master Agent's Correction/Approval judgment on that
    same attempt. Both are separate Slack posts, per CLAUDE.md's Slack
    Message Format (an agent's own status vs. Master Agent's supervision)."""

    def test_supervise_posts_started_then_finished_on_success(self):
        master_agent._supervise("Some Agent", lambda: "ok", lambda o: [], lambda o: "the report")

        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertTrue(any(p.startswith("[Some Agent]") and "STATUS: Working" in p for p in posts))
        self.assertTrue(any(p.startswith("[Some Agent]") and "STATUS: Completed" in p and "the report" in p for p in posts))

    def test_supervise_posts_a_master_agent_approval(self):
        master_agent._supervise("Some Agent", lambda: "ok", lambda o: [], lambda o: "report")

        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertTrue(any(p.startswith("[Master Agent]") and "DECISION: Approved" in p for p in posts))

    def test_supervise_posts_a_master_agent_correction_to_the_errors_channel(self):
        master_agent._supervise(
            "Some Agent", lambda: "bad",
            lambda o: [master_agent._log_entry("Correction", "Some Agent", "bad", "Rejected", "fix it")],
            lambda o: "report",
        )

        error_posts = [c.args[0] for c in self.mock_slack_error_post.call_args_list]
        self.assertTrue(any(p.startswith("[Master Agent]") and "DECISION: Rejected" in p for p in error_posts))
        # Corrections must not also land in the main channel -- one place to check.
        main_posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertFalse(any("DECISION: Rejected" in p for p in main_posts))

    def test_supervise_posts_failed_on_exception_to_the_errors_channel(self):
        def raiser():
            raise RuntimeError("boom")

        master_agent._supervise("Some Agent", raiser, lambda o: [], lambda o: "n/a")

        error_posts = [c.args[0] for c in self.mock_slack_error_post.call_args_list]
        self.assertTrue(any(p.startswith("[Some Agent]") and "STATUS: Failed" in p and "boom" in p for p in error_posts))

    def test_long_report_is_truncated_for_slack(self):
        long_report = "x" * (master_agent.SLACK_SUMMARY_MAX_CHARS + 200)
        master_agent._supervise("Some Agent", lambda: "ok", lambda o: [], lambda o: long_report)

        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        finished = next(p for p in posts if "STATUS: Completed" in p)
        self.assertLess(len(finished), len(long_report))
        self.assertIn("see saved report for full detail", finished)


def make_dataset_agent_request(sender="Modeling Agent", target="Data Agent",
                                reason="Model needs more training examples.", detail="") -> dict:
    text = master_agent.message_formatter.format_request(sender, target, reason, detail)
    return {"ts": "999.0", "text": text, "user": "U1"}


class TestCheckAgentRequests(MasterAgentTestCase):
    """check_agent_requests is Master Agent's read side of the Slack
    channel -- it re-runs Data Agent (the only agent it currently knows
    how to act on) when it sees a REQUEST addressed to it, and otherwise
    acknowledges without acting."""

    def setUp(self):
        super().setUp()
        self._configured_patch = patch.object(master_agent.slack_client, "is_configured", return_value=True)
        self._configured_patch.start()
        self._state_patch = patch("src.orchestration.state_manager.load_state", return_value={})
        self._state_patch.start()
        self._update_state_patch = patch("src.orchestration.state_manager.update_state")
        self.mock_update_state = self._update_state_patch.start()

    def tearDown(self):
        self._configured_patch.stop()
        self._state_patch.stop()
        self._update_state_patch.stop()
        super().tearDown()

    def test_returns_none_when_slack_not_configured(self):
        self._configured_patch.stop()
        with patch.object(master_agent.slack_client, "is_configured", return_value=False):
            result = master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}])
        self.assertIsNone(result)
        self._configured_patch.start()  # keep tearDown symmetric

    def test_returns_none_when_no_new_messages(self):
        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=[]):
            result = master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}])
        self.assertIsNone(result)

    def test_ignores_plain_status_updates(self):
        messages = [{"ts": "1.0", "text": "[Data Agent]\nSTATUS: Working\nTASK: stuff", "user": "B1"}]
        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=messages):
            result = master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}])
        self.assertIsNone(result)

    def test_acknowledges_request_for_unknown_agent_without_acting(self):
        messages = [make_dataset_agent_request(target="Reporting Agent")]
        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=messages):
            result = master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}])

        self.assertIsNone(result)
        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertTrue(any("does not yet know how to act on that agent" in p for p in posts))

    def test_reruns_data_agent_with_explicit_target_from_detail(self):
        parent_paper = {"title": "Parent Paper", "dataset": "DS1"}
        ranked_pool = [parent_paper]
        messages = [make_dataset_agent_request(detail="target=25000")]
        new_data_result = make_data_result(selected_datasets=[make_dataset_candidate(entry_count=25000)])
        new_cleaning_result = make_cleaning_result(datasets_merged=1)

        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=messages), \
             patch("src.agents.data_agent.run", return_value=new_data_result) as data_run, \
             patch("src.agents.cleaning_agent.run", return_value=new_cleaning_result):
            result = master_agent.check_agent_requests(parent_paper, ranked_pool)

        data_run.assert_called_once_with(parent_paper, ranked_pool, target_entries=25000)
        self.assertEqual(result["dataset_result"], new_data_result)
        self.assertEqual(result["cleaning_result"], new_cleaning_result)
        self.assertEqual(result["blockers"], [])

    def test_falls_back_to_default_increment_without_explicit_target(self):
        parent_paper = {"title": "Parent Paper", "dataset": "DS1"}
        ranked_pool = [parent_paper]
        messages = [make_dataset_agent_request(detail="")]

        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=messages), \
             patch("src.agents.data_agent.run", return_value=make_data_result()) as data_run, \
             patch("src.agents.cleaning_agent.run", return_value=make_cleaning_result()):
            master_agent.check_agent_requests(parent_paper, ranked_pool)

        from src.agents import data_agent
        expected_target = data_agent.TARGET_ENTRIES + master_agent.DATA_REQUEST_TARGET_INCREMENT
        data_run.assert_called_once_with(parent_paper, ranked_pool, target_entries=expected_target)

    def test_blocked_when_no_parent_paper_available(self):
        messages = [make_dataset_agent_request()]
        with patch.object(master_agent.slack_client, "fetch_new_messages", return_value=messages):
            result = master_agent.check_agent_requests(None, None)

        self.assertIsNone(result)
        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertTrue(any("no Parent Paper/ranked candidate pool is available" in p for p in posts))

    def test_floor_ts_is_used_when_no_cursor_persisted_yet(self):
        with patch("src.orchestration.state_manager.load_state", return_value={}), \
             patch.object(master_agent.slack_client, "fetch_new_messages", return_value=[]) as fetch:
            master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}], floor_ts="500.0")

        fetch.assert_called_once_with(oldest_ts="500.0")

    def test_persisted_cursor_wins_when_newer_than_floor_ts(self):
        with patch("src.orchestration.state_manager.load_state",
                    return_value={"slack_last_checked_ts": "900.0"}), \
             patch.object(master_agent.slack_client, "fetch_new_messages", return_value=[]) as fetch:
            master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}], floor_ts="500.0")

        fetch.assert_called_once_with(oldest_ts="900.0")

    def test_floor_ts_wins_when_newer_than_a_stale_persisted_cursor(self):
        """The scenario this whole mechanism exists for: --reset (or a
        stale cursor from an earlier topic) must never let this run read
        further back than its own run-start banner."""
        with patch("src.orchestration.state_manager.load_state",
                    return_value={"slack_last_checked_ts": "100.0"}), \
             patch.object(master_agent.slack_client, "fetch_new_messages", return_value=[]) as fetch:
            master_agent.check_agent_requests({"title": "P"}, [{"title": "P"}], floor_ts="800.0")

        fetch.assert_called_once_with(oldest_ts="800.0")


class TestMaxTs(unittest.TestCase):
    def test_returns_the_other_when_one_is_none(self):
        self.assertEqual(master_agent._max_ts(None, "5.0"), "5.0")
        self.assertEqual(master_agent._max_ts("5.0", None), "5.0")

    def test_returns_none_when_both_none(self):
        self.assertIsNone(master_agent._max_ts(None, None))

    def test_returns_the_numerically_larger_value(self):
        self.assertEqual(master_agent._max_ts("100.0", "900.0"), "900.0")
        self.assertEqual(master_agent._max_ts("900.0", "100.0"), "900.0")


class TestRunPipelineSlackBanner(MasterAgentTestCase):
    """run_pipeline posts a run-started banner (with the topic) before
    anything else, and its ts is threaded through to every
    check_agent_requests call as the cross-topic-contamination floor."""

    def _run_minimal_success(self, topic="credit card fraud detection"):
        ranked_pool = [{"title": "Parent Paper", "dataset": "DS1"}]
        validation_result = make_validation_result(ranked_pool)

        with patch.object(master_agent, "_check_intake", return_value=[]), \
             patch.object(master_agent, "_check_research", return_value=[]), \
             patch.object(master_agent, "_check_validation_and_selection", return_value=[]), \
             patch("src.agents.intake_agent.run", return_value={
                 "domain": "fraud", "keywords": ["fraud"], "search_queries": ["fraud ML"],
             }), \
             patch("src.agents.research_agent.run", return_value=[
                 {"title": "Parent Paper", "peer_reviewed": True, "year": 2023}
             ]), \
             patch("src.agents.validation_agent.run", return_value=validation_result), \
             patch("src.agents.data_agent.run", return_value=make_data_result(
                 selected_datasets=[make_dataset_candidate(entry_count=12000)],
             )), \
             patch("src.agents.cleaning_agent.run", return_value=make_cleaning_result()), \
             patch("src.orchestration.state_manager.update_state"), \
             patch("src.orchestration.dataset_cache.cache_validated_papers"):
            master_agent.run_pipeline(topic)

    def test_posts_a_run_started_banner_with_the_topic(self):
        self._run_minimal_success(topic="credit card fraud detection")

        posts = [c.args[0] for c in self.mock_slack_post.call_args_list]
        self.assertTrue(any(
            p.startswith("[Master Agent]") and "New pipeline run started" in p
            and "credit card fraud detection" in p
            for p in posts
        ))

    def test_banner_ts_is_passed_as_floor_to_every_relay_check(self):
        self.mock_slack_post.return_value = "1234.5678"
        with patch.object(master_agent.slack_client, "is_configured", return_value=True), \
             patch("src.orchestration.state_manager.load_state", return_value={}), \
             patch.object(master_agent.slack_client, "fetch_new_messages", return_value=[]) as fetch:
            self._run_minimal_success()

        # Validation, Data, and Cleaning stages all trigger a relay check.
        self.assertEqual(fetch.call_count, 3)
        for call in fetch.call_args_list:
            self.assertEqual(call.kwargs["oldest_ts"], "1234.5678")


if __name__ == "__main__":
    unittest.main()
