import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.agents import data_agent
from src.orchestration import dataset_cache


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


class TestRunAlgorithm(unittest.TestCase):
    """data_agent.run()'s escalation through stages 1-2 (parent paper, then
    the full ranked pool). _find_and_verify_dataset_for_paper is mocked so
    these tests exercise only the accumulation/stopping logic."""

    def test_parent_paper_dataset_alone_meets_target_and_no_other_candidates_exist(self):
        parent = make_paper(dataset="Big Dataset")
        big = make_link_candidate("https://kaggle.com/big-dataset", entry_count=15000)

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=big):
            result = data_agent.run(parent, [parent])

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["target_met"])
        self.assertEqual(result["total_entries"], 15000)
        self.assertEqual(result["selected_datasets"], [big])
        self.assertEqual(result["master_dataset"], big)
        self.assertFalse(result["is_combined"])
        self.assertEqual(result["search_stage"], "paper_traversal")

    def test_keeps_collecting_past_the_target_when_more_candidates_qualify(self):
        """The agent no longer stops the instant the entry-count target is
        met -- it keeps adding further qualifying, non-duplicate datasets
        (up to MAX_DATASETS) so a pool of several datasets is preferred
        over stopping at the first one big enough alone."""
        parent = make_paper(title="Parent", dataset="Big Dataset")
        big = make_link_candidate("https://kaggle.com/big-dataset", entry_count=15000)
        second = make_paper(title="Second", dataset="Second Dataset")
        second_candidate = make_link_candidate("https://kaggle.com/second", entry_count=2000)

        def fake_find(paper, _scope):
            return big if paper["title"] == "Parent" else second_candidate

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent, second])

        self.assertEqual(result["selected_datasets"], [big, second_candidate])
        self.assertEqual(result["total_entries"], 17000)
        self.assertTrue(result["is_combined"])
        self.assertTrue(result["target_met"])

    def test_falls_through_to_traversal_when_parent_dataset_missing(self):
        parent = make_paper(title="Parent", dataset="")
        alt = make_paper(title="Alt", dataset="Alt Dataset")
        alt_candidate = make_link_candidate("https://kaggle.com/alt", entry_count=12000)

        def fake_find(paper, _scope):
            if paper["title"] == "Parent":
                return None  # no dataset name -> nothing to search
            return alt_candidate

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent, alt])

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["selected_datasets"], [alt_candidate])
        self.assertIn("does not state a dataset", result["warnings"][0])

    def test_traversal_starts_at_second_highest_ranked_paper(self):
        parent = make_paper(title="Parent", dataset="Small Dataset")
        small = make_link_candidate("https://kaggle.com/small", entry_count=50)  # < 100, discarded
        second = make_paper(title="Second", dataset="Second Dataset")
        second_candidate = make_link_candidate("https://kaggle.com/second", entry_count=11000)

        calls = []

        def fake_find(paper, _scope):
            calls.append(paper["title"])
            if paper["title"] == "Parent":
                return small
            if paper["title"] == "Second":
                return second_candidate
            raise AssertionError(f"unexpected paper checked: {paper['title']}")

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent, second])

        self.assertEqual(calls, ["Parent", "Second"])
        self.assertEqual(result["selected_datasets"], [second_candidate])
        self.assertEqual(result["total_entries"], 11000)

    def test_parent_dataset_between_thresholds_is_counted_and_traversal_continues(self):
        parent = make_paper(title="Parent", dataset="Mid Dataset")
        mid = make_link_candidate("https://kaggle.com/mid", entry_count=4000)
        second = make_paper(title="Second", dataset="Second Dataset")
        second_candidate = make_link_candidate("https://kaggle.com/second", entry_count=7000)

        def fake_find(paper, _scope):
            return mid if paper["title"] == "Parent" else second_candidate

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent, second])

        self.assertEqual(result["total_entries"], 11000)
        self.assertTrue(result["target_met"])
        self.assertEqual(len(result["selected_datasets"]), 2)
        self.assertTrue(result["is_combined"])

    def test_duplicate_dataset_link_is_not_double_counted(self):
        parent = make_paper(title="Parent", dataset="Shared Dataset")
        shared = make_link_candidate("https://kaggle.com/shared-dataset", entry_count=4000)
        dup_paper = make_paper(title="Dup", dataset="Shared Dataset (mirror)")
        third = make_paper(title="Third", dataset="Unique Dataset")
        third_candidate = make_link_candidate("https://kaggle.com/unique", entry_count=7000)

        def fake_find(paper, _scope):
            if paper["title"] == "Parent":
                return shared
            if paper["title"] == "Dup":
                # Same link (e.g. a different paper describing the same
                # dataset with different text), different casing/trailing slash.
                return make_link_candidate("HTTPS://Kaggle.com/shared-dataset/", entry_count=4000)
            return third_candidate

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent, dup_paper, third])

        links = [d["name"] for d in result["selected_datasets"]]
        self.assertEqual(len(links), 2)
        self.assertEqual(result["total_entries"], 11000)

    def test_traversal_checks_the_whole_pool_when_under_the_dataset_cap(self):
        parent = make_paper(title="P0", dataset="D0")
        alts = [make_paper(title=f"P{i}", dataset=f"D{i}") for i in range(1, 4)]
        candidates = {
            f"D{i}": make_link_candidate(f"https://kaggle.com/d{i}", entry_count=1000)
            for i in range(4)
        }

        def fake_find(paper, _scope):
            return candidates[paper["dataset"]]

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent] + alts)

        # Only 4 candidates exist and none individually meets the target,
        # so all 4 are checked and added (well under the MAX_DATASETS cap),
        # leaving the pool DEGRADED rather than stopping short.
        self.assertEqual(len(result["selected_datasets"]), 4)
        self.assertEqual(result["total_entries"], 4000)
        self.assertEqual(result["status"], "DEGRADED")
        self.assertFalse(result["target_met"])

    def test_stops_at_max_datasets_cap_even_with_more_candidates_remaining(self):
        parent = make_paper(title="P0", dataset="D0")
        alts = [make_paper(title=f"P{i}", dataset=f"D{i}") for i in range(1, 8)]
        candidates = {
            f"D{i}": make_link_candidate(f"https://kaggle.com/d{i}", entry_count=1000)
            for i in range(8)
        }
        checked = []

        def fake_find(paper, _scope):
            checked.append(paper["dataset"])
            return candidates[paper["dataset"]]

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", side_effect=fake_find):
            result = data_agent.run(parent, [parent] + alts)

        # 8 candidates are each individually usable, but the agent stops at
        # MAX_DATASETS (5) rather than checking all 8 or trying to reach
        # the 10000-entry target, which is never hit.
        self.assertEqual(len(result["selected_datasets"]), data_agent.MAX_DATASETS)
        self.assertEqual(len(checked), data_agent.MAX_DATASETS)
        self.assertEqual(result["total_entries"], 5000)
        self.assertEqual(result["status"], "DEGRADED")
        self.assertFalse(result["target_met"])


class TestEscalationStages(unittest.TestCase):
    """Stage 3 (curated exploration sites) only fires when stages 1-2 found
    nothing usable at all -- and it must NOT retry the same (already-failed)
    per-paper dataset names, only search generally for the project scope."""

    def test_exploration_only_fires_when_paper_traversal_finds_nothing(self):
        parent = make_paper(title="Parent", dataset="Elusive Dataset")
        exploration_hit = make_link_candidate("https://kaggle.com/explored", entry_count=11000)

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[exploration_hit]) as explore_mock:
            result = data_agent.run(parent, [parent])

        explore_mock.assert_called_once()
        self.assertEqual(result["search_stage"], "exploration_sites")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["selected_datasets"], [exploration_hit])

    def test_exploration_search_is_scope_based_not_name_based(self):
        """The exploration search must be driven by the project's scope
        description, never by re-searching a specific paper's dataset name."""
        parent = make_paper(title="Parent", dataset="Elusive Dataset", abstract="Fraud detection study.")

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]) as explore_mock:
            data_agent.run(parent, [parent])

        explore_mock.assert_called_once()
        (scope_arg,), _kwargs = explore_mock.call_args
        self.assertNotIn("Elusive Dataset", scope_arg)
        self.assertIn("Parent", scope_arg)

    def test_exploration_does_not_fire_when_paper_traversal_succeeds(self):
        parent = make_paper(title="Parent", dataset="Findable Dataset")
        found = make_link_candidate("https://kaggle.com/found", entry_count=3000)

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=found), \
             patch.object(data_agent, "_search_exploration_sites") as explore_mock:
            result = data_agent.run(parent, [parent])

        explore_mock.assert_not_called()
        self.assertEqual(result["search_stage"], "paper_traversal")

    def test_exploration_stops_at_max_datasets_cap(self):
        """Stage 3 collects more than one dataset when more than one
        exploration hit qualifies, but still stops at MAX_DATASETS rather
        than adding every hit returned."""
        parent = make_paper(title="Parent", dataset="Elusive Dataset")
        hits = [
            make_link_candidate(f"https://kaggle.com/explored{i}", entry_count=1000)
            for i in range(data_agent.MAX_DATASETS + 3)
        ]

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=hits):
            result = data_agent.run(parent, [parent])

        self.assertEqual(len(result["selected_datasets"]), data_agent.MAX_DATASETS)
        self.assertEqual(result["selected_datasets"], hits[:data_agent.MAX_DATASETS])

    def test_not_found_when_every_stage_fails(self):
        parent = make_paper(title="Parent", dataset="Ghost Dataset")

        with patch.object(data_agent, "_find_and_verify_dataset_for_paper", return_value=None), \
             patch.object(data_agent, "_search_exploration_sites", return_value=[]):
            result = data_agent.run(parent, [parent])

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(result["selected_datasets"], [])
        self.assertIsNone(result["master_dataset"])


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

        def fake_download(candidate):
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

        def fake_download(candidate):
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
    def test_min_entries_to_consider_is_100(self):
        self.assertEqual(data_agent.MIN_ENTRIES_TO_CONSIDER, 100)

    def test_boundary_usability(self):
        at_floor = make_candidate(entry_count=100)
        below_floor = make_candidate(entry_count=99)
        self.assertTrue(data_agent._usable(at_floor, data_agent.MIN_ENTRIES_TO_CONSIDER))
        self.assertFalse(data_agent._usable(below_floor, data_agent.MIN_ENTRIES_TO_CONSIDER))

    def test_undownloaded_candidate_is_never_usable(self):
        """A verified, in-scope, well-counted candidate that was never
        actually downloaded (local_csv_path empty) is still rejected --
        download success is a required gate alongside the others."""
        candidate = make_candidate(entry_count=50000, local_csv_path="")
        self.assertFalse(data_agent._usable(candidate, data_agent.MIN_ENTRIES_TO_CONSIDER))


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
