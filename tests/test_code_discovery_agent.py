import unittest
from unittest.mock import patch, MagicMock

from src.agents import code_discovery_agent
from src.orchestration import code_discovery_cache
from src.search import brave_search


def make_paper(**overrides) -> dict:
    paper = {
        "title": "Big Data-Driven Distributed ML for Fraud Detection",
        "abstract": "A distributed machine learning framework for credit card fraud detection.",
        "code_url": "",
    }
    paper.update(overrides)
    return paper


def llm_response(candidates_json: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": candidates_json}}]}
    response.raise_for_status = lambda: None
    return response


def github_api_response(status_code: int = 200, json_data=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data if json_data is not None else {}
    return response


class CodeDiscoveryAgentTestCase(unittest.TestCase):
    """Every test here skips the cache (no prior successful run for this
    paper title) unless a test is specifically about caching."""

    def setUp(self):
        self._cache_patch = patch.object(code_discovery_cache, "get_cached_repo", return_value=None)
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()


class TestPaperStatedCodeUrl(CodeDiscoveryAgentTestCase):
    def test_stated_code_url_used_when_verified_and_reachable(self):
        paper = make_paper(code_url="https://github.com/example/fraud-detection")
        response = llm_response(
            '[{"repo_url": "https://github.com/example/fraud-detection", "display_name": "fraud-detection", '
            '"matches_parent_paper": true, "match_explanation": "Matches paper title and method.", '
            '"source": "github_search"}]'
        )

        with patch("requests.post", return_value=response), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(code_discovery_agent, "_enrich_github_repo") as enrich_mock, \
             patch.object(brave_search, "search_code_sites") as search_mock:
            result = code_discovery_agent.run(paper)

        search_mock.assert_not_called()  # a verified, reachable stated URL short-circuits the search stages
        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["source_stage"], "paper_stated")
        self.assertEqual(result["repository"]["repo_url"], "https://github.com/example/fraud-detection")
        self.assertEqual(result["repository"]["source"], "paper_stated")

    def test_stated_code_url_rejected_falls_through_to_search(self):
        paper = make_paper(code_url="https://github.com/unrelated/other-project")
        stated_response = llm_response(
            '[{"repo_url": "https://github.com/unrelated/other-project", '
            '"matches_parent_paper": false, "match_explanation": "No connection to this paper.", '
            '"source": "other"}]'
        )
        search_response = llm_response("[]")

        with patch("requests.post", side_effect=[stated_response, search_response]), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(brave_search, "search_code_sites", return_value=[]) as search_mock:
            result = code_discovery_agent.run(paper)

        search_mock.assert_called_once()
        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertIsNone(result["repository"])

    def test_no_stated_code_url_skips_straight_to_search(self):
        paper = make_paper(code_url="")
        search_response = llm_response("[]")

        with patch("requests.post", return_value=search_response), \
             patch.object(brave_search, "search_code_sites", return_value=[]) as search_mock:
            result = code_discovery_agent.run(paper)

        search_mock.assert_called_once()
        self.assertIn("does not state a code_url", " ".join(result["warnings"]))
        self.assertEqual(result["status"], "NOT_FOUND")


class TestGithubSearchAndVerification(CodeDiscoveryAgentTestCase):
    def test_github_match_enriched_and_chosen(self):
        paper = make_paper(code_url="")
        search_results = [
            {"title": "fraud-detection", "url": "https://github.com/author/fraud-detection", "description": ""},
        ]
        response = llm_response(
            '[{"repo_url": "https://github.com/author/fraud-detection", "display_name": "fraud-detection", '
            '"matches_parent_paper": true, "match_explanation": "Official implementation.", '
            '"source": "github_search"}]'
        )
        repo_info = github_api_response(json_data={
            "license": {"spdx_id": "MIT"}, "stargazers_count": 42, "default_branch": "main",
        })
        contents = github_api_response(json_data=[
            {"name": "README.md", "type": "file"}, {"name": "requirements.txt", "type": "file"},
        ])

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch("requests.get", side_effect=[repo_info, contents]):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["source_stage"], "github_search")
        repo = result["repository"]
        self.assertEqual(repo["license"], "MIT")
        self.assertEqual(repo["stars"], 42)
        self.assertEqual(repo["dependency_files"], ["requirements.txt"])
        self.assertTrue(repo["has_setup_instructions"])

    def test_unreachable_match_is_not_chosen(self):
        paper = make_paper(code_url="")
        search_results = [
            {"title": "fraud-detection", "url": "https://github.com/author/fraud-detection", "description": ""},
        ]
        response = llm_response(
            '[{"repo_url": "https://github.com/author/fraud-detection", '
            '"matches_parent_paper": true, "match_explanation": "Matches.", "source": "github_search"}]'
        )

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(False, 0)):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertIsNone(result["repository"])

    def test_multiple_github_matches_ranked_by_stars(self):
        paper = make_paper(code_url="")
        search_results = [
            {"title": "low-star-repo", "url": "https://github.com/a/low-star-repo", "description": ""},
            {"title": "high-star-repo", "url": "https://github.com/b/high-star-repo", "description": ""},
        ]
        response = llm_response(
            '[{"repo_url": "https://github.com/a/low-star-repo", '
            '"matches_parent_paper": true, "match_explanation": "x", "source": "github_search"}, '
            '{"repo_url": "https://github.com/b/high-star-repo", '
            '"matches_parent_paper": true, "match_explanation": "y", "source": "github_search"}]'
        )

        def fake_metadata(owner, repo):
            stars = 5 if repo == "low-star-repo" else 500
            return {"license": "", "stars": stars, "default_branch": "main"}

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(code_discovery_agent, "_fetch_github_metadata", side_effect=fake_metadata), \
             patch.object(code_discovery_agent, "_fetch_github_contents_listing", return_value=[]):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["repository"]["repo_url"], "https://github.com/b/high-star-repo")
        self.assertEqual(result["repository"]["stars"], 500)

    def test_malformed_llm_json_falls_back_to_regex_extraction(self):
        paper = make_paper(code_url="")
        search_results = [{"title": "repo", "url": "https://github.com/a/repo", "description": ""}]
        response = llm_response(
            'Here is the result:\n'
            '[{"repo_url": "https://github.com/a/repo", "matches_parent_paper": true, '
            '"match_explanation": "x", "source": "github_search"}]\n'
            'Let me know if you need anything else.'
        )

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(code_discovery_agent, "_fetch_github_metadata", return_value={}), \
             patch.object(code_discovery_agent, "_fetch_github_contents_listing", return_value=[]):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["repository"]["repo_url"], "https://github.com/a/repo")

    def test_non_github_match_chosen_as_kaggle_when_no_github_match(self):
        paper = make_paper(code_url="")
        search_results = [{"title": "notebook", "url": "https://kaggle.com/code/author/notebook", "description": ""}]
        response = llm_response(
            '[{"repo_url": "https://kaggle.com/code/author/notebook", '
            '"matches_parent_paper": true, "match_explanation": "x", "source": "kaggle"}]'
        )

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["source_stage"], "kaggle")
        self.assertEqual(result["repository"]["repo_url"], "https://kaggle.com/code/author/notebook")


class TestRepoMetadataFetch(unittest.TestCase):
    def test_enrich_sets_license_stars_and_dependency_files(self):
        repo = code_discovery_agent._default_repo(repo_url="https://github.com/a/repo")
        repo_info = github_api_response(json_data={
            "license": {"spdx_id": "Apache-2.0"}, "stargazers_count": 10, "default_branch": "main",
        })
        contents = github_api_response(json_data=[
            {"name": "README.md", "type": "file"},
            {"name": "environment.yml", "type": "file"},
            {"name": "src", "type": "dir"},
        ])

        with patch("requests.get", side_effect=[repo_info, contents]):
            code_discovery_agent._enrich_github_repo(repo, "a", "repo")

        self.assertEqual(repo["license"], "Apache-2.0")
        self.assertEqual(repo["stars"], 10)
        self.assertEqual(repo["dependency_files"], ["environment.yml"])
        self.assertTrue(repo["has_setup_instructions"])

    def test_no_dependency_file_means_no_setup_instructions(self):
        repo = code_discovery_agent._default_repo(repo_url="https://github.com/a/repo")
        repo_info = github_api_response(json_data={"license": {}, "stargazers_count": 0, "default_branch": "main"})
        contents = github_api_response(json_data=[{"name": "README.md", "type": "file"}])

        with patch("requests.get", side_effect=[repo_info, contents]):
            code_discovery_agent._enrich_github_repo(repo, "a", "repo")

        self.assertEqual(repo["dependency_files"], [])
        self.assertFalse(repo["has_setup_instructions"])

    def test_contents_listing_handles_non_list_response_without_raising(self):
        response = github_api_response(json_data={"message": "Not Found"})
        with patch("requests.get", return_value=response):
            result = code_discovery_agent._fetch_github_contents_listing("a", "repo")
        self.assertEqual(result, [])

    def test_metadata_fetch_never_raises_on_network_error(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            result = code_discovery_agent._fetch_github_metadata("a", "repo")
        self.assertEqual(result, {})

    def test_metadata_fetch_returns_empty_on_non_200(self):
        with patch("requests.get", return_value=github_api_response(status_code=404)):
            result = code_discovery_agent._fetch_github_metadata("a", "repo")
        self.assertEqual(result, {})


class TestParseGithubRepo(unittest.TestCase):
    def test_parses_owner_and_repo(self):
        self.assertEqual(
            code_discovery_agent._parse_github_repo("https://github.com/author/project"),
            ("author", "project"),
        )

    def test_strips_trailing_path_and_git_suffix(self):
        self.assertEqual(
            code_discovery_agent._parse_github_repo("https://github.com/author/project.git"),
            ("author", "project"),
        )
        self.assertEqual(
            code_discovery_agent._parse_github_repo("https://github.com/author/project/tree/main"),
            ("author", "project"),
        )

    def test_non_github_url_returns_none(self):
        self.assertIsNone(code_discovery_agent._parse_github_repo("https://kaggle.com/code/author/notebook"))


class TestCaching(CodeDiscoveryAgentTestCase):
    def test_cache_hit_skips_search_entirely(self):
        cached_repo = code_discovery_agent._default_repo(
            repo_url="https://github.com/author/fraud-detection", source="github_search",
            matches_parent_paper=True, reachable=True,
        )
        with patch.object(code_discovery_cache, "get_cached_repo", return_value=cached_repo), \
             patch.object(brave_search, "search_code_sites") as search_mock, \
             patch("requests.post") as post_mock:
            result = code_discovery_agent.run(make_paper())

        search_mock.assert_not_called()
        post_mock.assert_not_called()
        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["source_stage"], "cache_hit")
        self.assertEqual(result["repository"]["verification_method"], "cache_hit")

    def test_found_result_is_cached(self):
        paper = make_paper(code_url="")
        search_results = [{"title": "repo", "url": "https://github.com/a/repo", "description": ""}]
        response = llm_response(
            '[{"repo_url": "https://github.com/a/repo", "matches_parent_paper": true, '
            '"match_explanation": "x", "source": "github_search"}]'
        )

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=search_results), \
             patch.object(code_discovery_agent, "_check_url_reachable", return_value=(True, 200)), \
             patch.object(code_discovery_agent, "_fetch_github_metadata", return_value={}), \
             patch.object(code_discovery_agent, "_fetch_github_contents_listing", return_value=[]), \
             patch.object(code_discovery_cache, "cache_repo") as cache_mock:
            code_discovery_agent.run(paper)

        cache_mock.assert_called_once()
        self.assertEqual(cache_mock.call_args.args[0], paper["title"])

    def test_not_found_result_is_never_cached(self):
        paper = make_paper(code_url="")
        response = llm_response("[]")

        with patch("requests.post", return_value=response), \
             patch.object(brave_search, "search_code_sites", return_value=[]), \
             patch.object(code_discovery_cache, "cache_repo") as cache_mock:
            code_discovery_agent.run(paper)

        cache_mock.assert_not_called()


class TestNotFoundNeverBlocks(CodeDiscoveryAgentTestCase):
    def test_not_found_returns_clean_fully_populated_dict(self):
        paper = make_paper(code_url="")

        with patch.object(brave_search, "search_code_sites", return_value=[]):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertIsNone(result["repository"])
        self.assertEqual(result["candidates_considered"], [])
        self.assertIn("source_stage", result)
        self.assertIn("warnings", result)
        self.assertTrue(len(result["warnings"]) > 0)

    def test_search_exception_does_not_raise(self):
        paper = make_paper(code_url="")

        with patch.object(brave_search, "search_code_sites", side_effect=RuntimeError("search provider down")):
            result = code_discovery_agent.run(paper)

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertIn("Code search failed", " ".join(result["warnings"]))


if __name__ == "__main__":
    unittest.main()
