import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.agents import data_agent
from src.orchestration import dataset_cache

# KAGGLE_USERNAME/KAGGLE_KEY are genuinely configured in this project's real
# .env -- without this, every test that reaches _resolve_dataset or
# _search_exploration_sites would make live Kaggle API calls. Individual
# Kaggle-specific test classes override this locally in their own setUp.
_kaggle_creds_patches = [
    patch.object(data_agent, "KAGGLE_USERNAME", ""),
    patch.object(data_agent, "KAGGLE_KEY", ""),
]

# run() normally asks the LLM how many entries the Parent Paper needs
# (plan_data_needs). The run() tests below use a fixed 10,000-entry plan so
# they test only the collection logic; TestPlanDataNeeds tests the real one.
_real_plan_data_needs = data_agent.plan_data_needs
FIXED_PLAN = {"target_entries": 10000, "paper_dataset_size": None, "paper_model": "", "paper_dataset": "",
              "reasoning": "fixed test plan", "capped": False, "source": "llm"}
_kaggle_creds_patches.append(
    patch.object(data_agent, "plan_data_needs", side_effect=lambda *a, **k: dict(FIXED_PLAN)))


def setUpModule():
    for p in _kaggle_creds_patches:
        p.start()


def tearDownModule():
    for p in _kaggle_creds_patches:
        p.stop()


def make_paper(**overrides) -> dict:
    paper = {
        "title": "Untitled Paper",
        "dataset": "Some Dataset",
        "dataset_source": "Official benchmark",
    }
    paper.update(overrides)
    return paper


def make_candidate(**overrides) -> dict:
    """A usable candidate: verified, in scope, identified by its link."""
    candidate = {
        "name": "https://kaggle.com/some-dataset",
        "display_name": "Some Dataset",
        "matches_parent_paper": True,
        "match_explanation": "Confirmed via official page.",
        "in_scope": True,
        "scope_explanation": "Matches the project's domain.",
        "source": "Kaggle",
        "source_url": "https://kaggle.com/some-dataset",
        "access_method": "direct download",
        "license": "CC0",
        "format": "CSV",
        "entry_count": 5000,
        "usability_score": None,
        "verified": True,
        "verification_method": "brave_search",
        "local_csv_path": "data/raw/fake-dataset.csv",
        "download_status": "ok",
        "notes": "",
    }
    candidate.update(overrides)
    return candidate


def make_link_candidate(link: str, **overrides) -> dict:
    return make_candidate(name=link, source_url=link, **overrides)


class TestParentOnlyMode(unittest.TestCase):
    """The first Data Agent run collects ONLY the Parent Paper's own
    dataset. Other sources are searched only as a logged substitute when
    the paper's own dataset cannot be obtained at all."""

    def test_collects_only_the_parent_dataset_even_when_others_qualify(self):
        parent, other = make_paper(title="Parent"), make_paper(title="Other", dataset="Other Dataset")
        own = make_link_candidate("https://kaggle.com/own", entry_count=300)

        with patch.object(data_agent, "_find_parent_dataset", return_value=[own]), \
             patch.object(data_agent, "_find_and_verify_dataset_for_paper") as other_papers, \
             patch.object(data_agent, "_search_exploration_sites") as explore:
            result = data_agent.run(parent, [parent, other])

        other_papers.assert_not_called()
        explore.assert_not_called()
        self.assertEqual(result["selected_datasets"], [own])
        self.assertEqual(result["status"], "OK")  # no entry minimum: 300 rows is fine
        self.assertEqual(result["mode"], "parent_only")
        self.assertTrue(result["parent_dataset_found"])
        self.assertIsNone(result["target_entries"])
        self.assertIsNone(result["target_met"])
        self.assertEqual(result["search_stage"], "parent_paper")
        self.assertEqual(result["data_plan"]["reasoning"], "fixed test plan")

    def test_records_what_the_agent_could_read_of_the_paper(self):
        parent = make_paper(title="Parent")
        own = make_link_candidate("https://kaggle.com/own")
        with patch.object(data_agent, "_find_parent_dataset", return_value=[own]):
            result = data_agent.run(parent, [parent])
        # conftest disables downloads, so only the stored metadata is readable.
        self.assertEqual(result["paper_text_status"], "ABSTRACT_ONLY")
        self.assertIn("NOT the full paper", result["paper_text_notes"])

    def test_substitute_from_other_papers_when_the_parent_dataset_is_missing(self):
        parent = make_paper(title="Parent", dataset="Elusive")
        alt, alt2 = make_paper(title="Alt"), make_paper(title="Alt2")
        alt_candidate = make_link_candidate("https://kaggle.com/alt", entry_count=12000)

        with patch.object(data_agent, "_find_parent_dataset", return_value=[]), \
             patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=alt_candidate) as find, \
             patch.object(data_agent, "_search_exploration_sites") as explore:
            result = data_agent.run(parent, [parent, alt, alt2])

        find.assert_called_once()  # one substitute is enough without a goal
        self.assertEqual(find.call_args.args[0]["title"], "Alt")  # starts at the second-highest paper
        explore.assert_not_called()
        self.assertEqual(result["status"], "DEGRADED")
        self.assertFalse(result["parent_dataset_found"])
        self.assertEqual(result["selected_datasets"], [alt_candidate])
        self.assertEqual(result["search_stage"], "substitute_paper_traversal")
        self.assertTrue(any("SUBSTITUTE" in w for w in result["warnings"]))

    def test_substitute_from_dataset_directories_when_no_paper_has_one(self):
        parent = make_paper(title="Parent", dataset="Elusive Dataset", abstract="Fraud detection study.")
        hit = make_link_candidate("https://kaggle.com/explored")

        with patch.object(data_agent, "_find_parent_dataset", return_value=[]), \
             patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[hit]) as explore:
            result = data_agent.run(parent, [parent, make_paper(title="Alt")])

        scope_arg = explore.call_args.args[0]
        self.assertNotIn("Elusive Dataset", scope_arg)  # scope-based, not a retry of the failed name
        self.assertIn("Parent", scope_arg)
        self.assertEqual(result["selected_datasets"], [hit])
        self.assertIn("exploration_sites", result["search_stage"])
        self.assertEqual(result["status"], "DEGRADED")

    def test_not_found_when_nothing_is_usable(self):
        parent = make_paper(title="Parent")
        with patch.object(data_agent, "_find_parent_dataset", return_value=[]), \
             patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]):
            result = data_agent.run(parent, [parent])
        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(result["selected_datasets"], [])


class TestFindParentDataset(unittest.TestCase):
    """Stage 1 reads the dataset from the paper itself: each link written in
    the paper first, then a search for the paper's dataset name."""

    PLAN = {"paper_dataset": "ULB Credit Card Fraud", "dataset_source": "Kaggle",
            "dataset_urls": ["https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"]}

    def test_a_link_in_the_paper_is_tried_before_any_search(self):
        found = make_link_candidate(self.PLAN["dataset_urls"][0])
        with patch.object(data_agent, "_dataset_from_paper_link", return_value=found) as by_link, \
             patch.object(data_agent, "_resolve_dataset") as search:
            tried = data_agent._find_parent_dataset(make_paper(), self.PLAN, "scope", {"target_terms": []})
        by_link.assert_called_once()
        search.assert_not_called()
        self.assertEqual(tried, [found])

    def test_falls_back_to_the_papers_name_then_the_stored_name(self):
        parent = make_paper(dataset="Credit Card Dataset", dataset_source="Kaggle")
        with patch.object(data_agent, "_dataset_from_paper_link", return_value=None), \
             patch.object(data_agent, "_resolve_dataset", return_value=make_candidate(local_csv_path="")) as search:
            data_agent._find_parent_dataset(parent, self.PLAN, "scope", {"target_terms": []})
        names = [c.args[0] for c in search.call_args_list]
        self.assertEqual(names, ["ULB Credit Card Fraud", "Credit Card Dataset"])

    def test_stops_after_the_first_usable_search_result(self):
        with patch.object(data_agent, "_dataset_from_paper_link", return_value=None), \
             patch.object(data_agent, "_resolve_dataset", return_value=make_candidate()) as search:
            data_agent._find_parent_dataset(make_paper(dataset="Other Name"), self.PLAN, "scope", {"target_terms": []})
        search.assert_called_once()

    def test_paper_link_candidate_is_marked_as_the_papers_own(self):
        def fake_download(candidate, _terms=None):
            candidate.update(local_csv_path="data/raw/x.csv", entry_count=284807, download_status="ok")
            return True

        with patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(data_agent, "_download_and_convert_to_csv", side_effect=fake_download), \
             patch.object(data_agent, "_check_fit", return_value=(True, "")):
            candidate = data_agent._dataset_from_paper_link(
                self.PLAN["dataset_urls"][0], self.PLAN, make_paper(), {"target_terms": [], "rejected_links": set()})

        self.assertTrue(candidate["matches_parent_paper"])
        self.assertTrue(candidate["in_scope"])
        self.assertEqual(candidate["verification_method"], "parent_paper_text")
        self.assertTrue(data_agent._usable(candidate, 1))

    def test_unreachable_paper_link_gives_nothing(self):
        with patch.object(data_agent, "_check_url_reachable", return_value=(False, 404)), \
             patch.object(data_agent, "_download_and_convert_to_csv") as download:
            candidate = data_agent._dataset_from_paper_link(
                self.PLAN["dataset_urls"][0], self.PLAN, make_paper(), {"target_terms": [], "rejected_links": set()})
        self.assertIsNone(candidate)
        download.assert_not_called()


class TestExpandMode(unittest.TestCase):
    """mode="expand": only on an Evaluation Agent request relayed by Master
    Agent. Keeps the datasets already collected and adds datasets from
    sources away from the Parent Paper until the request's goal is met."""

    def setUp(self):
        self.parent = make_paper(title="Parent")
        self.own = make_link_candidate("https://kaggle.com/own", entry_count=5000, matches_parent_paper=True)

    def test_needs_a_goal(self):
        with self.assertRaises(ValueError):
            data_agent.run(self.parent, [self.parent], mode="expand")

    def test_keeps_existing_data_and_stops_once_the_goal_is_met(self):
        papers = [self.parent] + [make_paper(title=f"P{i}") for i in range(3)]
        extra = {"P0": make_link_candidate("https://kaggle.com/p0", entry_count=4000),
                 "P1": make_link_candidate("https://kaggle.com/p1", entry_count=4000),
                 "P2": make_link_candidate("https://kaggle.com/p2", entry_count=4000)}

        with patch.object(data_agent, "plan_data_needs") as plan, \
             patch.object(data_agent, "_find_and_verify_dataset_for_paper",
                          side_effect=lambda p, *_a, **_k: extra[p["title"]]), \
             patch.object(data_agent, "_search_exploration_sites") as explore:
            result = data_agent.run(self.parent, papers, target_entries=12000, mode="expand",
                                    existing_datasets=[self.own])

        plan.assert_not_called()
        explore.assert_not_called()
        self.assertEqual(result["selected_datasets"], [self.own, extra["P0"], extra["P1"]])
        self.assertEqual(result["total_entries"], 13000)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["parent_dataset_found"])
        self.assertEqual(result["data_plan"]["source"], "request")

    def test_searches_dataset_directories_when_other_papers_fall_short(self):
        hit = make_link_candidate("https://kaggle.com/explored", entry_count=20000)
        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[hit]):
            result = data_agent.run(self.parent, [self.parent, make_paper(title="Alt")], target_entries=10000,
                                    mode="expand", existing_datasets=[self.own])
        self.assertEqual(result["selected_datasets"], [self.own, hit])
        self.assertEqual(result["search_stage"], "expansion+exploration_sites")

    def test_degraded_when_the_goal_cannot_be_met(self):
        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]):
            result = data_agent.run(self.parent, [self.parent], target_entries=50000, mode="expand",
                                    existing_datasets=[self.own])
        self.assertEqual(result["status"], "DEGRADED")
        self.assertFalse(result["target_met"])
        self.assertEqual(result["selected_datasets"], [self.own])

    def test_a_duplicate_link_is_not_counted_twice(self):
        duplicate = make_link_candidate("https://kaggle.com/own", entry_count=5000)
        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=duplicate), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]):
            result = data_agent.run(self.parent, [self.parent, make_paper(title="A")], target_entries=8000,
                                    mode="expand", existing_datasets=[self.own])
        self.assertEqual(result["total_entries"], 5000)

    def test_stops_at_the_dataset_cap(self):
        papers = [self.parent] + [make_paper(title=f"P{i}") for i in range(10)]
        with patch.object(data_agent, "_find_and_verify_dataset_for_paper",
                          side_effect=lambda p, *_a, **_k: make_link_candidate(f"https://kaggle.com/{p['title']}",
                                                                               entry_count=10)), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]):
            result = data_agent.run(self.parent, papers, target_entries=400000, mode="expand",
                                    existing_datasets=[self.own])
        self.assertEqual(len(result["selected_datasets"]), data_agent.MAX_DATASETS)


class TestKaggleSearchIntegration(unittest.TestCase):
    """Kaggle's own dataset search (when credentials are configured) is
    merged alongside Brave's results, not a replacement for them -- both
    _resolve_dataset (stage 1/2) and _search_exploration_sites (stage 3)
    feed the combined list through the same extraction pipeline."""

    def test_kaggle_results_merged_into_exploration_extraction(self):
        kaggle_hit = {"title": "Kaggle Hit", "url": "https://www.kaggle.com/datasets/a/b", "description": ""}
        with patch("src.search.brave_search.search_dataset_sites", return_value=[]), \
             patch.object(data_agent, "_search_kaggle_datasets", return_value=[kaggle_hit]) as kaggle_mock, \
             patch.object(data_agent, "_extract_dataset_candidates_from_results", return_value=[]) as extract_mock:
            data_agent._search_exploration_sites("fraud detection scope")

        kaggle_mock.assert_called_once_with("fraud detection scope")
        results_arg = extract_mock.call_args.args[0]
        self.assertIn(kaggle_hit, results_arg)

    def test_exploration_searches_with_short_query_but_scope_checks_with_full_description(self):
        """Verified live: the long scope description returns zero Kaggle
        results, so the short search_query is what gets sent -- while the
        extraction/scope check still sees the full description."""
        with patch("src.search.brave_search.search_dataset_sites", return_value=[]) as brave_mock, \
             patch.object(data_agent, "_search_kaggle_datasets", return_value=[{"title": "t", "url": "u"}]) as kaggle_mock, \
             patch.object(data_agent, "_extract_dataset_candidates_from_results", return_value=[]) as extract_mock:
            data_agent._search_exploration_sites(
                "Long paper title -- long abstract text", search_query="credit card fraud",
            )

        self.assertEqual(brave_mock.call_args.args[0], "credit card fraud")
        kaggle_mock.assert_called_once_with("credit card fraud")
        self.assertEqual(extract_mock.call_args.args[3], "Long paper title -- long abstract text")

    def test_run_passes_search_query_to_exploration_stage(self):
        parent = {"title": "P", "dataset": "", "abstract": "a"}
        with patch.object(data_agent, "_search_exploration_sites", return_value=[]) as explore_mock:
            data_agent.run(parent, [parent], search_query="credit card fraud")
        self.assertEqual(explore_mock.call_args.args[2], "credit card fraud")

    def test_results_per_query_within_brave_limit(self):
        from src.search import brave_search
        self.assertLessEqual(data_agent.RESULTS_PER_QUERY, brave_search.BRAVE_MAX_COUNT)

    def test_kaggle_results_merged_into_resolve_dataset_extraction(self):
        kaggle_hit = {"title": "Kaggle Hit", "url": "https://www.kaggle.com/datasets/a/b", "description": ""}
        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch("src.search.brave_search.search", return_value=[]), \
             patch.object(data_agent, "_search_kaggle_datasets", return_value=[kaggle_hit]) as kaggle_mock, \
             patch.object(data_agent, "_extract_dataset_candidates_from_results", return_value=[]) as extract_mock:
            data_agent._resolve_dataset(
                "Some Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        kaggle_mock.assert_called_once()
        results_arg = extract_mock.call_args.args[0]
        self.assertIn(kaggle_hit, results_arg)


class TestUsabilityScore(unittest.TestCase):
    def test_higher_entries_permissive_license_direct_download_scores_higher(self):
        strong = make_candidate(entry_count=50000, license="CC0", format="CSV", access_method="direct download")
        weak = make_candidate(entry_count=100, license="", format="", access_method="request/application required")
        self.assertGreater(data_agent._usability_score(strong), data_agent._usability_score(weak))

    def test_score_never_guesses_missing_entry_count(self):
        candidate = make_candidate(entry_count=None, license="MIT", format="CSV", access_method="direct download")
        # Should not raise or fabricate a value for the missing entry count.
        score = data_agent._usability_score(candidate)
        self.assertIsInstance(score, float)


class TestResolveDataset(unittest.TestCase):
    """Lower-level search/verify/cache/scope behavior, network and LLM calls mocked."""

    def test_cache_hit_skips_search_entirely(self):
        cached = make_link_candidate("https://kaggle.com/cached", entry_count=9000)
        with patch.object(dataset_cache, "get_cached_dataset", return_value=cached) as get_cached, \
             patch("src.search.brave_search.search") as search_mock:
            candidate = data_agent._resolve_dataset(
                "Cached Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        get_cached.assert_called_once_with("Cached Dataset")
        search_mock.assert_not_called()
        self.assertEqual(candidate["verification_method"], "cache_hit")
        self.assertEqual(candidate["entry_count"], 9000)

    def _llm_response(self, candidates_json: str) -> MagicMock:
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": candidates_json}}]}
        response.raise_for_status = lambda: None
        return response

    def test_entry_count_not_stated_stays_none_and_is_not_cached(self):
        """When neither the search snippet nor the HF/page-text fallbacks
        (mocked here as no-ops) turn up a stated count, the candidate is
        still returned (as a fallback answer) but never cached -- so a
        later run gets a genuine retry instead of a permanent unknown."""
        llm_response = self._llm_response(
            '[{"name": "Unsized Dataset", "source": "GitHub", "in_scope": true, '
            '"source_url": "https://github.com/example/unsized", "entry_count": null}]'
        )

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Unsized Dataset", "url": "https://github.com/example/unsized", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(data_agent, "_download_and_convert_to_csv", return_value=False), \
             patch.object(data_agent, "_resolve_entry_count"):
            candidate = data_agent._resolve_dataset(
                "Unsized Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        self.assertIsNone(candidate["entry_count"])
        self.assertTrue(candidate["verified"])
        self.assertEqual(candidate["name"], "https://github.com/example/unsized")
        self.assertEqual(candidate["display_name"], "Unsized Dataset")
        cache_dataset.assert_not_called()

    def test_successful_download_populates_entry_count_and_caches(self):
        """A candidate is only cached once it's actually been downloaded --
        _download_and_convert_to_csv (mocked here) is what sets
        local_csv_path and entry_count from the real file."""
        llm_response = self._llm_response(
            '[{"name": "Big Dataset", "source": "GitHub", "in_scope": true, '
            '"source_url": "https://github.com/example/big", "entry_count": null}]'
        )

        def fake_download(candidate, _target_terms=None):
            candidate["local_csv_path"] = "data/raw/big-dataset.csv"
            candidate["entry_count"] = 25000
            candidate["entry_count_source"] = "downloaded file"
            return True

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Big Dataset", "url": "https://github.com/example/big", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(data_agent, "_download_and_convert_to_csv", side_effect=fake_download):
            candidate = data_agent._resolve_dataset(
                "Big Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        self.assertEqual(candidate["entry_count"], 25000)
        self.assertEqual(candidate["local_csv_path"], "data/raw/big-dataset.csv")
        cache_dataset.assert_called_once()

    def test_download_failure_leaves_candidate_unusable_and_uncached(self):
        """A verified, in-scope candidate that fails to download is
        rejected the same way an unreachable one is -- _usable() requires
        local_csv_path, which a failed download never sets."""
        llm_response = self._llm_response(
            '[{"name": "Undownloadable Dataset", "source": "Kaggle", "in_scope": true, '
            '"source_url": "https://kaggle.com/gated", "entry_count": 50000}]'
        )

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Undownloadable Dataset", "url": "https://kaggle.com/gated", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(data_agent, "_download_and_convert_to_csv", return_value=False):
            candidate = data_agent._resolve_dataset(
                "Undownloadable Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        self.assertFalse(candidate.get("local_csv_path"))
        self.assertFalse(data_agent._usable(candidate, 1))
        cache_dataset.assert_not_called()

    def test_fit_failure_leaves_candidate_unusable_and_uncached(self):
        """A candidate that downloads fine but fails the fit check (no label
        for the prediction target) is rejected and not cached, same as any
        other gate."""
        llm_response = self._llm_response(
            '[{"name": "Anonymized Dataset", "source": "Kaggle", "in_scope": true, '
            '"source_url": "https://kaggle.com/anon", "entry_count": null}]'
        )

        def fake_download(candidate, _target_terms=None):
            candidate["local_csv_path"] = "data/raw/anon.csv"
            candidate["entry_count"] = 50000
            candidate["entry_count_source"] = "downloaded file"
            return True

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Anonymized Dataset", "url": "https://kaggle.com/anon", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(data_agent, "_download_and_convert_to_csv", side_effect=fake_download), \
             patch.object(data_agent, "_check_fit", return_value=(False, "no column matches the target")):
            candidate = data_agent._resolve_dataset(
                "Anonymized Dataset", "", "", "", query_builder=data_agent._build_query,
                parent_paper={"title": "P", "methodology": "named features"},
            )

        self.assertEqual(candidate["reproducibility_fit"], False)
        self.assertFalse(data_agent._usable(candidate, 1))
        cache_dataset.assert_not_called()

    def test_second_candidate_tried_when_first_is_unreachable(self):
        """top_k > 1 lets a later LLM-ranked candidate rescue a dataset name
        whose top result turns out to be dead, instead of giving up on it."""
        llm_response = self._llm_response(
            '[{"name": "First", "source": "GitHub", "in_scope": true, '
            '"source_url": "https://example.com/first", "entry_count": 5000}, '
            '{"name": "Second", "source": "GitHub", "in_scope": true, '
            '"source_url": "https://example.com/second", "entry_count": 6000}]'
        )

        def fake_reachable(url):
            return (True, 200) if url == "https://example.com/second" else (False, 0)

        def fake_download(candidate, _target_terms=None):
            candidate["local_csv_path"] = "data/raw/second.csv"
            candidate["entry_count"] = candidate.get("entry_count") or 6000
            candidate["entry_count_source"] = "downloaded file"
            return True

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset"), \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "First", "url": "https://example.com/first", "description": ""},
                 {"title": "Second", "url": "https://example.com/second", "description": ""},
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", side_effect=fake_reachable), \
             patch.object(data_agent, "_download_and_convert_to_csv", side_effect=fake_download):
            candidate = data_agent._resolve_dataset(
                "Some Dataset", "", "", "", query_builder=data_agent._build_query, top_k=3,
            )

        self.assertEqual(candidate["source_url"], "https://example.com/second")
        self.assertTrue(candidate["verified"])
        self.assertEqual(candidate["entry_count"], 6000)

    def test_unreachable_url_is_not_verified_and_not_cached(self):
        llm_response = self._llm_response(
            '[{"name": "Broken Dataset", "source": "GitHub", "in_scope": true, '
            '"source_url": "https://example.com/broken", "entry_count": 50000}]'
        )

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Broken Dataset", "url": "https://example.com/broken", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(False, 0)):
            candidate = data_agent._resolve_dataset(
                "Broken Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        self.assertFalse(candidate["verified"])
        self.assertIsNone(candidate["entry_count"])
        cache_dataset.assert_not_called()  # failures are never cached

    def test_out_of_scope_candidate_is_rejected_even_if_reachable(self):
        llm_response = self._llm_response(
            '[{"name": "Unrelated Dataset", "source": "Kaggle", "in_scope": false, '
            '"scope_explanation": "Different domain entirely.", '
            '"source_url": "https://kaggle.com/unrelated", "entry_count": 50000}]'
        )

        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", return_value=[
                 {"title": "Unrelated Dataset", "url": "https://kaggle.com/unrelated", "description": ""}
             ]), \
             patch("requests.post", return_value=llm_response), \
             patch.object(data_agent, "_check_url_reachable", return_value=(True, 200)):
            candidate = data_agent._resolve_dataset(
                "Unrelated Dataset", "", "", "fraud detection", query_builder=data_agent._build_query,
            )

        self.assertFalse(data_agent._usable(candidate, 1))
        cache_dataset.assert_not_called()  # not in scope -> not cached as a success

    def test_failed_search_is_never_cached_allowing_a_later_retry(self):
        with patch.object(dataset_cache, "get_cached_dataset", return_value=None), \
             patch.object(dataset_cache, "cache_dataset") as cache_dataset, \
             patch("src.search.brave_search.search", side_effect=RuntimeError("network down")):
            candidate = data_agent._resolve_dataset(
                "Some Dataset", "", "", "", query_builder=data_agent._build_query,
            )

        self.assertFalse(candidate["verified"])
        cache_dataset.assert_not_called()


class TestEntryCountFloor(unittest.TestCase):
    """The 10,000+ target is enforced on the merged master CSV by the
    Cleaning Agent, not per individual dataset here -- so the floor is
    trivial (any real, non-empty download counts), not a meaningful gate."""

    def test_min_entries_to_consider_is_1(self):
        self.assertEqual(data_agent.MIN_ENTRIES_TO_CONSIDER, 1)

    def test_boundary_usability(self):
        at_floor = make_candidate(entry_count=1)
        below_floor = make_candidate(entry_count=0)
        self.assertTrue(data_agent._usable(at_floor, data_agent.MIN_ENTRIES_TO_CONSIDER))
        self.assertFalse(data_agent._usable(below_floor, data_agent.MIN_ENTRIES_TO_CONSIDER))

    def test_small_dataset_is_still_usable(self):
        """A dataset far below the old 100-entry threshold is fine now --
        individual size no longer gates inclusion."""
        candidate = make_candidate(entry_count=12)
        self.assertTrue(data_agent._usable(candidate, data_agent.MIN_ENTRIES_TO_CONSIDER))

    def test_undownloaded_candidate_is_never_usable(self):
        """A verified, in-scope, well-counted candidate that was never
        actually downloaded (local_csv_path empty) is still rejected --
        download success is a required gate alongside the others."""
        candidate = make_candidate(entry_count=50000, local_csv_path="")
        self.assertFalse(data_agent._usable(candidate, data_agent.MIN_ENTRIES_TO_CONSIDER))

    def test_reproducibility_unfit_candidate_is_never_usable(self):
        """A downloaded, verified, in-scope candidate that failed the fit
        check (no label for the prediction target) is still rejected."""
        candidate = make_candidate(entry_count=50000, reproducibility_fit=False)
        self.assertFalse(data_agent._usable(candidate, data_agent.MIN_ENTRIES_TO_CONSIDER))

    def test_unchecked_reproducibility_fit_defaults_to_usable(self):
        """A candidate that was never checked (reproducibility_fit still
        None, e.g. because it was rejected earlier for unreachability) does
        not itself block usability -- existing tests that don't set the
        field must keep passing."""
        candidate = make_candidate(entry_count=50000)
        self.assertNotIn("reproducibility_fit", candidate)  # make_candidate doesn't set it
        self.assertTrue(data_agent._usable(candidate, data_agent.MIN_ENTRIES_TO_CONSIDER))


class TestResolveEntryCount(unittest.TestCase):
    """_resolve_entry_count's fallback chain: Hugging Face API first (exact,
    no LLM), then page-text extraction, never fabricating a value."""

    def test_already_known_entry_count_is_left_alone_and_skips_lookups(self):
        candidate = make_candidate(entry_count=999)
        with patch.object(data_agent, "_lookup_huggingface_entry_count") as hf_mock, \
             patch.object(data_agent, "_fetch_page_text") as fetch_mock:
            data_agent._resolve_entry_count(candidate, "Some Dataset")

        hf_mock.assert_not_called()
        fetch_mock.assert_not_called()
        self.assertEqual(candidate["entry_count"], 999)
        self.assertEqual(candidate["entry_count_source"], "search result")

    def test_huggingface_lookup_short_circuits_page_fetch(self):
        candidate = make_candidate(entry_count=None, source_url="https://huggingface.co/datasets/foo/bar")
        with patch.object(data_agent, "_lookup_huggingface_entry_count", return_value=12345) as hf_mock, \
             patch.object(data_agent, "_fetch_page_text") as fetch_mock:
            data_agent._resolve_entry_count(candidate, "foo/bar")

        hf_mock.assert_called_once_with("https://huggingface.co/datasets/foo/bar")
        fetch_mock.assert_not_called()
        self.assertEqual(candidate["entry_count"], 12345)
        self.assertEqual(candidate["entry_count_source"], "Hugging Face API")

    def test_falls_back_to_page_text_when_hf_lookup_fails(self):
        candidate = make_candidate(entry_count=None, source_url="https://example.com/dataset")
        with patch.object(data_agent, "_lookup_huggingface_entry_count", return_value=None), \
             patch.object(data_agent, "_fetch_page_text", return_value="This dataset has 42000 rows.") as fetch_mock, \
             patch.object(data_agent, "_extract_entry_count_from_page", return_value=42000) as extract_mock:
            data_agent._resolve_entry_count(candidate, "Some Dataset")

        fetch_mock.assert_called_once_with("https://example.com/dataset")
        extract_mock.assert_called_once()
        self.assertEqual(candidate["entry_count"], 42000)
        self.assertEqual(candidate["entry_count_source"], "the dataset's own page")

    def test_stays_none_when_nothing_found(self):
        candidate = make_candidate(entry_count=None, source_url="https://example.com/dataset")
        with patch.object(data_agent, "_lookup_huggingface_entry_count", return_value=None), \
             patch.object(data_agent, "_fetch_page_text", return_value=""), \
             patch.object(data_agent, "_extract_entry_count_from_page", return_value=None):
            data_agent._resolve_entry_count(candidate, "Some Dataset")

        self.assertIsNone(candidate["entry_count"])
        self.assertEqual(candidate.get("entry_count_source", ""), "")


class TestDescribeProvenance(unittest.TestCase):
    """_describe_provenance turns a candidate's verification_method and
    entry_count_source into a human-readable, per-candidate explanation of
    how it was found -- this is what makes the console log dynamic instead
    of a fixed phrase for every dataset."""

    def test_describes_fresh_search_and_stated_count(self):
        candidate = make_candidate(verification_method="brave_search", entry_count_source="search result")
        desc = data_agent._describe_provenance(candidate)
        self.assertIn("Brave search", desc)
        self.assertIn("stated in the search result", desc)

    def test_describes_exploration_search_and_huggingface_count(self):
        candidate = make_candidate(verification_method="exploration_site", entry_count_source="Hugging Face API")
        desc = data_agent._describe_provenance(candidate)
        self.assertIn("curated dataset-directory search", desc)
        self.assertIn("Hugging Face's dataset API", desc)

    def test_describes_cache_hit_and_page_derived_count(self):
        candidate = make_candidate(verification_method="cache_hit", entry_count_source="the dataset's own page")
        desc = data_agent._describe_provenance(candidate)
        self.assertIn("reused from an earlier search", desc)
        self.assertIn("dataset's own page", desc)

    def test_two_different_candidates_produce_two_different_descriptions(self):
        a = make_candidate(verification_method="brave_search", entry_count_source="search result")
        b = make_candidate(verification_method="exploration_site", entry_count_source="the dataset's own page")
        self.assertNotEqual(data_agent._describe_provenance(a), data_agent._describe_provenance(b))

    def test_unknown_or_missing_provenance_fields_produce_empty_string(self):
        candidate = make_candidate(verification_method="", entry_count_source="")
        self.assertEqual(data_agent._describe_provenance(candidate), "")


class TestHuggingFaceLookup(unittest.TestCase):
    def test_non_huggingface_url_returns_none_without_request(self):
        with patch("requests.get") as get_mock:
            result = data_agent._lookup_huggingface_entry_count("https://kaggle.com/dataset")
        get_mock.assert_not_called()
        self.assertIsNone(result)

    def test_sums_splits_for_single_config_dataset(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.json.return_value = {
            "dataset_info": {
                "splits": [{"name": "train", "num_examples": 8000}, {"name": "test", "num_examples": 2000}]
            }
        }
        with patch("requests.get", return_value=response):
            result = data_agent._lookup_huggingface_entry_count("https://huggingface.co/datasets/foo/bar")
        self.assertEqual(result, 10000)

    def test_sums_splits_across_multiple_configs(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.json.return_value = {
            "dataset_info": {
                "config_a": {"splits": [{"name": "train", "num_examples": 500}]},
                "config_b": {"splits": [{"name": "train", "num_examples": 700}]},
            }
        }
        with patch("requests.get", return_value=response):
            result = data_agent._lookup_huggingface_entry_count("https://huggingface.co/datasets/foo")
        self.assertEqual(result, 1200)

    def test_missing_dataset_info_returns_none(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.json.return_value = {}
        with patch("requests.get", return_value=response):
            result = data_agent._lookup_huggingface_entry_count("https://huggingface.co/datasets/foo")
        self.assertIsNone(result)

    def test_request_failure_returns_none(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            result = data_agent._lookup_huggingface_entry_count("https://huggingface.co/datasets/foo")
        self.assertIsNone(result)


def make_response(content: bytes, content_type: str = "") -> MagicMock:
    resp = MagicMock()
    resp.headers = {"Content-Type": content_type}
    resp.content = content
    resp.raise_for_status = lambda: None
    return resp


def make_zip_bytes(files: dict) -> bytes:
    import zipfile
    import io as _io
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TestParseTabularContent(unittest.TestCase):
    def test_html_content_type_is_rejected_without_parsing(self):
        resp = make_response(b"<html><body>Not a dataset</body></html>", content_type="text/html; charset=utf-8")
        result = data_agent._parse_tabular_content(resp, "https://example.com/landing")
        self.assertIsNone(result)

    def test_parses_direct_csv(self):
        resp = make_response(b"a,b\n1,2\n3,4\n", content_type="text/csv")
        df = data_agent._parse_tabular_content(resp, "https://example.com/data.csv")
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(len(df), 2)

    def test_parses_csv_with_no_content_type_via_extension(self):
        resp = make_response(b"a,b\n1,2\n", content_type="")
        df = data_agent._parse_tabular_content(resp, "https://example.com/data.csv")
        self.assertEqual(len(df), 1)

    def test_parses_json_array(self):
        resp = make_response(b'[{"a": 1, "b": 2}, {"a": 3, "b": 4}]', content_type="application/json")
        df = data_agent._parse_tabular_content(resp, "https://example.com/data.json")
        self.assertEqual(len(df), 2)

    def test_parses_zip_with_csv_inside(self):
        zip_bytes = make_zip_bytes({"readme.txt": "hello", "data.csv": "a,b\n1,2\n3,4\n5,6\n"})
        resp = make_response(zip_bytes, content_type="application/zip")
        df = data_agent._parse_tabular_content(resp, "https://example.com/archive.zip")
        self.assertEqual(len(df), 3)

    def test_zip_with_no_csv_inside_returns_none(self):
        zip_bytes = make_zip_bytes({"readme.txt": "hello", "data.json": "{}"})
        resp = make_response(zip_bytes, content_type="application/zip")
        result = data_agent._parse_tabular_content(resp, "https://example.com/archive.zip")
        self.assertIsNone(result)

    def test_garbage_content_returns_none(self):
        resp = make_response(b"\x00\x01\x02not tabular at all", content_type="application/octet-stream")
        result = data_agent._parse_tabular_content(resp, "https://example.com/mystery")
        self.assertIsNone(result)


class TestDownloadAndConvertToCsv(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._raw_dir_patch = patch.object(data_agent, "RAW_DATA_DIR", Path(self._tmpdir))
        self._raw_dir_patch.start()

    def tearDown(self):
        self._raw_dir_patch.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_successful_download_writes_csv_and_sets_fields(self):
        candidate = make_candidate(
            name="https://example.com/data.csv", source_url="https://example.com/data.csv",
            local_csv_path="", entry_count=None, entry_count_source="",
        )
        resp = make_response(b"a,b\n1,2\n3,4\n5,6\n", content_type="text/csv")

        with patch.object(data_agent, "_download_huggingface_csv", return_value=None), \
             patch("requests.get", return_value=resp):
            ok = data_agent._download_and_convert_to_csv(candidate)

        self.assertTrue(ok)
        self.assertTrue(candidate["local_csv_path"])
        self.assertTrue(Path(candidate["local_csv_path"]).exists())
        self.assertEqual(candidate["entry_count"], 3)
        self.assertEqual(candidate["entry_count_source"], "downloaded file")
        self.assertEqual(candidate["download_status"], "ok")

    def test_html_landing_page_fails_and_is_logged(self):
        candidate = make_candidate(
            name="https://kaggle.com/datasets/x", source_url="https://kaggle.com/datasets/x",
            local_csv_path="", entry_count=None, entry_count_source="",
        )
        resp = make_response(b"<html>recaptcha challenge</html>", content_type="text/html")

        with patch.object(data_agent, "_download_huggingface_csv", return_value=None), \
             patch("requests.get", return_value=resp):
            ok = data_agent._download_and_convert_to_csv(candidate)

        self.assertFalse(ok)
        self.assertFalse(candidate["local_csv_path"])
        self.assertIn("failed", candidate["download_status"])

    def test_network_failure_is_caught_and_logged(self):
        candidate = make_candidate(
            name="https://example.com/data.csv", source_url="https://example.com/data.csv",
            local_csv_path="", entry_count=None, entry_count_source="",
        )
        with patch.object(data_agent, "_download_huggingface_csv", return_value=None), \
             patch("requests.get", side_effect=RuntimeError("connection reset")):
            ok = data_agent._download_and_convert_to_csv(candidate)

        self.assertFalse(ok)
        self.assertFalse(candidate["local_csv_path"])
        self.assertIn("failed", candidate["download_status"])

    def test_huggingface_special_case_is_tried_first(self):
        candidate = make_candidate(
            name="https://huggingface.co/datasets/foo/bar",
            source_url="https://huggingface.co/datasets/foo/bar",
            local_csv_path="", entry_count=None, entry_count_source="",
        )
        fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with patch.object(data_agent, "_download_huggingface_csv", return_value=fake_df) as hf_mock, \
             patch("requests.get") as get_mock:
            ok = data_agent._download_and_convert_to_csv(candidate)

        hf_mock.assert_called_once_with(candidate)
        get_mock.assert_not_called()
        self.assertTrue(ok)
        self.assertEqual(candidate["entry_count"], 2)

    def test_kaggle_special_case_is_tried_between_huggingface_and_generic(self):
        candidate = make_candidate(
            name="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            local_csv_path="", entry_count=None, entry_count_source="",
        )
        fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with patch.object(data_agent, "_download_huggingface_csv", return_value=None) as hf_mock, \
             patch.object(data_agent, "_download_kaggle_dataset", return_value=fake_df) as kaggle_mock, \
             patch("requests.get") as get_mock:
            ok = data_agent._download_and_convert_to_csv(candidate)

        hf_mock.assert_called_once_with(candidate)
        kaggle_mock.assert_called_once_with(candidate, None)
        get_mock.assert_not_called()  # generic path never reached
        self.assertTrue(ok)
        self.assertEqual(candidate["entry_count"], 2)


class TestColumnNamesAreNotAGate(unittest.TestCase):
    """Anonymized/PCA columns and columns with no descriptive name are
    accepted: _check_fit only checks for a label of the prediction target,
    and records a note on undescribed columns for the report."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_csv(self, columns_and_rows: dict) -> str:
        path = Path(self._tmpdir) / "sample.csv"
        pd.DataFrame(columns_and_rows).to_csv(path, index=False)
        return str(path)

    def test_pca_columns_are_accepted_with_a_note(self):
        candidate = make_candidate(display_name="Credit Card Fraud", local_csv_path=self._write_csv({
            "Time": [0, 1], "V1": [-1.35, 0.2], "V2": [-0.07, 1.1], "V3": [1.0, 2.0],
            "Amount": [149.6, 2.6], "Class": [0, 1],
        }))
        with patch("requests.post") as post_mock:
            fit, explanation = data_agent._check_fit(candidate, {"title": "P"}, {"target_terms": ["fraud"]})

        post_mock.assert_not_called()  # deterministic, no LLM call
        self.assertTrue(fit)
        self.assertEqual(explanation, "")
        self.assertIn("V1", candidate["column_notes"])
        self.assertIn("3/6 columns", candidate["column_notes"])

    def test_only_undescribed_columns_are_accepted(self):
        candidate = make_candidate(local_csv_path=self._write_csv({
            "col_1": [1, 2], "col_2": [3, 4], "col_3": [0, 1],
        }))
        fit, _explanation = data_agent._check_fit(candidate, {"title": "P"}, {"target_terms": []})
        self.assertTrue(fit)
        self.assertIn("3/3 columns", candidate["column_notes"])

    def test_named_columns_get_no_note(self):
        candidate = make_candidate(local_csv_path=self._write_csv({
            "transaction_amount": [10.5, 200.0], "is_fraud": [0, 1],
        }))
        fit, _explanation = data_agent._check_fit(candidate, {"title": "P"}, {"target_terms": ["fraud"]})
        self.assertTrue(fit)
        self.assertEqual(candidate["column_notes"], "")

    def test_wrong_target_is_still_rejected(self):
        candidate = make_candidate(
            display_name="Default of Credit Card Clients", name="https://x/default", source_url="https://x/default",
            scope_explanation="", local_csv_path=self._write_csv({"limit_bal": [1, 2], "default_payment": [0, 1]}),
        )
        fit, explanation = data_agent._check_fit(candidate, {"title": "P"}, {"target_terms": ["fraud"]})
        self.assertFalse(fit)
        self.assertIn("predicts something else", explanation)


class TestPlanDataNeeds(unittest.TestCase):
    """plan_data_needs: the agent sets its own entry target from the Parent
    Paper; the code checks what the LLM returns."""

    PAPER = {"title": "Fraud detection with XGBoost", "dataset": "ULB Credit Card Fraud",
             "methodology": "XGBoost with SMOTE",
             "abstract": "We use 284,807 transactions from European cardholders.",
             "results_summary": "AUC 0.98"}

    def _plan(self, answer: str, paper=None):
        with patch.object(data_agent.llm, "chat", return_value=answer) as chat:
            plan = _real_plan_data_needs(paper or self.PAPER, ["fraud"])
        return plan, chat

    def test_uses_the_agents_target_and_the_papers_stated_size(self):
        plan, chat = self._plan('{"paper_model": "XGBoost", "paper_dataset": "ULB Credit Card Fraud", '
                                '"paper_dataset_size": 284807, "target_entries": 284807, '
                                '"reasoning": "Match the full dataset."}')
        self.assertEqual(plan["target_entries"], 284807)
        self.assertEqual(plan["paper_dataset_size"], 284807)
        self.assertEqual(plan["paper_model"], "XGBoost")
        self.assertEqual(plan["source"], "llm")
        prompt = chat.call_args.args[0]
        self.assertIn("XGBoost with SMOTE", prompt)
        self.assertIn("AUC 0.98", prompt)

    def test_a_small_target_is_allowed(self):
        """No project minimum: a few hundred rows is a valid choice."""
        plan, _ = self._plan('{"target_entries": 300, "paper_dataset_size": null, "reasoning": "Small set."}')
        self.assertEqual(plan["target_entries"], 300)

    def test_size_not_in_the_paper_text_is_dropped(self):
        plan, _ = self._plan('{"target_entries": 50000, "paper_dataset_size": 123456, "reasoning": "r"}')
        self.assertIsNone(plan["paper_dataset_size"])
        self.assertEqual(plan["target_entries"], 50000)
        self.assertIn("not in the paper's text", plan["reasoning"])

    def test_target_above_the_row_budget_is_capped(self):
        plan, _ = self._plan('{"target_entries": 99000000, "reasoning": "r"}')
        self.assertEqual(plan["target_entries"], data_agent.MAX_TARGET_ENTRIES)
        self.assertTrue(plan["capped"])

    def test_invalid_target_means_no_target(self):
        plan, _ = self._plan('{"target_entries": "lots", "reasoning": "r"}')
        self.assertIsNone(plan["target_entries"])

    def test_llm_failure_means_no_target(self):
        with patch.object(data_agent.llm, "chat", side_effect=RuntimeError("router down")):
            plan = _real_plan_data_needs(self.PAPER)
        self.assertIsNone(plan["target_entries"])
        self.assertIn("router down", plan["reasoning"])
        self.assertEqual(plan["source"], "none")

    def test_dataset_links_must_be_written_in_the_paper(self):
        paper = dict(self.PAPER, dataset_source="Available at https://www.kaggle.com/mlg-ulb/creditcardfraud.")
        plan, _ = self._plan('{"target_entries": 1000, "dataset_urls": ["https://www.kaggle.com/mlg-ulb/creditcardfraud", '
                             '"https://made-up.example.com/data"], "reasoning": "r"}', paper)
        self.assertEqual(plan["dataset_urls"], ["https://www.kaggle.com/mlg-ulb/creditcardfraud"])
        self.assertIn("not written in the paper", plan["reasoning"])

    def test_json_inside_extra_text_is_parsed(self):
        plan, _ = self._plan('Here you go:\n{"target_entries": 20000, "reasoning": "r"}')
        self.assertEqual(plan["target_entries"], 20000)


class TestDownloadKaggleDataset(unittest.TestCase):
    """_download_kaggle_dataset: only engages when KAGGLE_USERNAME/KAGGLE_KEY
    are configured (so it's a no-op for anyone else running the project),
    tries HTTP Basic Auth first, falls back to a Bearer token on 401/403."""

    def setUp(self):
        self._user_patch = patch.object(data_agent, "KAGGLE_USERNAME", "testuser")
        self._key_patch = patch.object(data_agent, "KAGGLE_KEY", "testkey")
        self._user_patch.start()
        self._key_patch.start()

    def tearDown(self):
        self._user_patch.stop()
        self._key_patch.stop()

    def test_no_credentials_returns_none_without_request(self):
        with patch.object(data_agent, "KAGGLE_USERNAME", ""), \
             patch.object(data_agent, "KAGGLE_KEY", ""), \
             patch("requests.get") as get_mock:
            result = data_agent._download_kaggle_dataset(
                make_candidate(source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            )
        get_mock.assert_not_called()
        self.assertIsNone(result)

    def test_non_kaggle_url_returns_none_without_request(self):
        with patch("requests.get") as get_mock:
            result = data_agent._download_kaggle_dataset(
                make_candidate(source_url="https://example.com/data.csv")
            )
        get_mock.assert_not_called()
        self.assertIsNone(result)

    def test_basic_auth_success(self):
        resp = make_response(b"a,b\n1,2\n3,4\n", content_type="text/csv")
        resp.status_code = 200
        with patch("requests.get", return_value=resp) as get_mock:
            df = data_agent._download_kaggle_dataset(
                make_candidate(source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            )
        self.assertEqual(len(df), 2)
        _args, kwargs = get_mock.call_args
        self.assertEqual(kwargs.get("auth"), ("testuser", "testkey"))

    def test_falls_back_to_bearer_token_on_401(self):
        unauthorized = MagicMock(status_code=401)
        authorized = make_response(b"a,b\n1,2\n", content_type="text/csv")
        authorized.status_code = 200

        with patch("requests.get", side_effect=[unauthorized, authorized]) as get_mock:
            df = data_agent._download_kaggle_dataset(
                make_candidate(source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            )

        self.assertEqual(len(df), 1)
        self.assertEqual(get_mock.call_count, 2)
        second_call_kwargs = get_mock.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs["headers"]["Authorization"], "Bearer testkey")

    def test_request_failure_returns_none(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            result = data_agent._download_kaggle_dataset(
                make_candidate(source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            )
        self.assertIsNone(result)


class TestSearchKaggleDatasets(unittest.TestCase):
    """_search_kaggle_datasets: queries Kaggle's own dataset search API
    directly (more reliable than depending on Brave having indexed a
    matching Kaggle page), returning results shaped like Brave's
    (title/url/description) so they flow through the existing extraction
    pipeline unchanged."""

    def setUp(self):
        self._user_patch = patch.object(data_agent, "KAGGLE_USERNAME", "testuser")
        self._key_patch = patch.object(data_agent, "KAGGLE_KEY", "testkey")
        self._user_patch.start()
        self._key_patch.start()

    def tearDown(self):
        self._user_patch.stop()
        self._key_patch.stop()

    def _list_response(self, datasets: list) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json.return_value = datasets
        return resp

    def test_no_credentials_returns_empty_without_request(self):
        with patch.object(data_agent, "KAGGLE_USERNAME", ""), \
             patch.object(data_agent, "KAGGLE_KEY", ""), \
             patch("requests.get") as get_mock:
            result = data_agent._search_kaggle_datasets("credit card fraud")
        get_mock.assert_not_called()
        self.assertEqual(result, [])

    def test_empty_query_returns_empty_without_request(self):
        with patch("requests.get") as get_mock:
            result = data_agent._search_kaggle_datasets("")
        get_mock.assert_not_called()
        self.assertEqual(result, [])

    def test_maps_kaggle_fields_into_brave_shaped_results(self):
        resp = self._list_response([
            {"ref": "mlg-ulb/creditcardfraud", "title": "Credit Card Fraud Detection",
             "subtitle": "Anonymized credit card transactions"},
            {"ref": "someone/other-dataset", "title": "Other Dataset", "subtitle": ""},
        ])
        with patch("requests.get", return_value=resp) as get_mock:
            results = data_agent._search_kaggle_datasets("credit card fraud")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Credit Card Fraud Detection")
        self.assertEqual(results[0]["url"], "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        self.assertEqual(results[0]["description"], "Anonymized credit card transactions")
        _args, kwargs = get_mock.call_args
        self.assertEqual(kwargs["auth"], ("testuser", "testkey"))
        self.assertEqual(kwargs["params"]["search"], "credit card fraud")

    def test_entries_missing_ref_are_skipped(self):
        resp = self._list_response([{"title": "No ref here", "subtitle": ""}])
        with patch("requests.get", return_value=resp):
            results = data_agent._search_kaggle_datasets("query")
        self.assertEqual(results, [])

    def test_non_list_response_returns_empty(self):
        resp = self._list_response(None)
        resp.json.return_value = {"unexpected": "shape"}
        with patch("requests.get", return_value=resp):
            results = data_agent._search_kaggle_datasets("query")
        self.assertEqual(results, [])

    def test_request_failure_returns_empty(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            results = data_agent._search_kaggle_datasets("query")
        self.assertEqual(results, [])


class TestDownloadHuggingfaceCsv(unittest.TestCase):
    def test_non_huggingface_url_returns_none_without_request(self):
        candidate = make_candidate(name="https://kaggle.com/x", source_url="https://kaggle.com/x")
        with patch("requests.get") as get_mock:
            result = data_agent._download_huggingface_csv(candidate)
        get_mock.assert_not_called()
        self.assertIsNone(result)

    def test_resolves_and_reads_parquet_file(self):
        candidate = make_candidate(
            name="https://huggingface.co/datasets/foo/bar",
            source_url="https://huggingface.co/datasets/foo/bar",
        )
        parquet_bytes = io.BytesIO()
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(parquet_bytes)

        api_resp = MagicMock()
        api_resp.raise_for_status = lambda: None
        api_resp.json.return_value = {"parquet_files": [{"url": "https://huggingface.co/datasets/foo/bar/resolve/main/data.parquet"}]}

        file_resp = MagicMock()
        file_resp.raise_for_status = lambda: None
        file_resp.content = parquet_bytes.getvalue()

        with patch("requests.get", side_effect=[api_resp, file_resp]):
            df = data_agent._download_huggingface_csv(candidate)

        self.assertEqual(len(df), 3)

    def test_no_parquet_files_returns_none(self):
        candidate = make_candidate(
            name="https://huggingface.co/datasets/foo/bar",
            source_url="https://huggingface.co/datasets/foo/bar",
        )
        api_resp = MagicMock()
        api_resp.raise_for_status = lambda: None
        api_resp.json.return_value = {"parquet_files": []}

        with patch("requests.get", return_value=api_resp):
            result = data_agent._download_huggingface_csv(candidate)
        self.assertIsNone(result)

    def test_request_failure_returns_none(self):
        candidate = make_candidate(
            name="https://huggingface.co/datasets/foo/bar",
            source_url="https://huggingface.co/datasets/foo/bar",
        )
        with patch("requests.get", side_effect=RuntimeError("network down")):
            result = data_agent._download_huggingface_csv(candidate)
        self.assertIsNone(result)


class TestFetchPageText(unittest.TestCase):
    def test_empty_url_returns_empty_string(self):
        self.assertEqual(data_agent._fetch_page_text(""), "")

    def test_strips_html_and_scripts(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.text = "<html><head><script>bad()</script></head><body><p>Hello world</p></body></html>"
        with patch("requests.get", return_value=response):
            text = data_agent._fetch_page_text("https://example.com/data")
        self.assertIn("Hello world", text)
        self.assertNotIn("bad()", text)

    def test_request_failure_returns_empty_string(self):
        with patch("requests.get", side_effect=RuntimeError("timeout")):
            self.assertEqual(data_agent._fetch_page_text("https://example.com/data"), "")


class TestExtractEntryCountFromPage(unittest.TestCase):
    def test_empty_page_text_returns_none_without_request(self):
        with patch("requests.post") as post_mock:
            result = data_agent._extract_entry_count_from_page("", "Some Dataset")
        post_mock.assert_not_called()
        self.assertIsNone(result)

    def test_extracts_stated_count(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.json.return_value = {"choices": [{"message": {"content": '{"entry_count": 284807}'}}]}
        with patch("requests.post", return_value=response):
            result = data_agent._extract_entry_count_from_page(
                "This dataset has 284807 transactions.", "Credit Card Fraud"
            )
        self.assertEqual(result, 284807)

    def test_not_stated_returns_none(self):
        response = MagicMock()
        response.raise_for_status = lambda: None
        response.json.return_value = {"choices": [{"message": {"content": '{"entry_count": null}'}}]}
        with patch("requests.post", return_value=response):
            result = data_agent._extract_entry_count_from_page("No size mentioned here.", "Some Dataset")
        self.assertIsNone(result)

    def test_request_failure_returns_none(self):
        with patch("requests.post", side_effect=RuntimeError("network down")):
            result = data_agent._extract_entry_count_from_page("some text", "Dataset")
        self.assertIsNone(result)


class TestNormalizeDatasetLink(unittest.TestCase):
    def test_equivalent_links_normalize_to_the_same_key(self):
        a = dataset_cache.normalize_dataset_link("https://Kaggle.com/dataset/")
        b = dataset_cache.normalize_dataset_link("http://kaggle.com/dataset?ref=1")
        self.assertEqual(a, b)

    def test_empty_link_normalizes_to_empty_string(self):
        self.assertEqual(dataset_cache.normalize_dataset_link(""), "")


if __name__ == "__main__":
    unittest.main()


def test_target_fit_rejects_a_cached_wrong_target_dataset(tmp_path):
    """A cache hit predates the target-fit check, so try_add applies it:
    the loan-default dataset (served from cache in a live run) must be
    rejected for a fraud project, and the fraud dataset kept."""
    loan_csv, fraud_csv = tmp_path / "loan.csv", tmp_path / "fraud.csv"
    pd.DataFrame({"limit_bal": [1, 2], "default.payment.next.month": [0, 1]}).to_csv(loan_csv, index=False)
    pd.DataFrame({"amt": [1.0, 2.0], "is_fraud": [0, 1]}).to_csv(fraud_csv, index=False)
    loan = make_candidate(name="https://kaggle.com/loan", display_name="Default of Credit Card Clients",
                          scope_explanation="Credit card clients.", local_csv_path=str(loan_csv),
                          verification_method="cache_hit")
    fraud = make_candidate(name="https://kaggle.com/fraud", display_name="Transactions",
                           local_csv_path=str(fraud_csv))
    parent, other = make_paper(title="Parent", dataset="Loan"), make_paper(title="Other", dataset="Fraud")

    # A plain pytest function: setUpModule's fixed plan may not be active here.
    with patch.object(data_agent, "plan_data_needs", return_value=dict(FIXED_PLAN)),          patch.object(data_agent, "_find_parent_dataset", return_value=[loan]),          patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=fraud),          patch.object(data_agent, "_search_exploration_sites", return_value=[]):
        result = data_agent.run(parent, [parent, other], target_terms=["fraud"])

    assert result["selected_datasets"] == [fraud]
    assert any("Default of Credit Card Clients" in w for w in result["warnings"])
