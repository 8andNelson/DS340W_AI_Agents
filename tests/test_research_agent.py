import json
from unittest.mock import patch

from src.agents import research_agent


def brave_result(title, url, description=""):
    return {"title": title, "url": url, "description": description}


class TestCleanTitle:
    def test_strips_pipe_segments(self):
        assert research_agent.clean_title(
            "Credit Card Fraud Detection using Ensemble Methods | IEEE Conference Publication | IEEE Xplore"
        ) == "Credit Card Fraud Detection using Ensemble Methods"

    def test_strips_known_site_suffix(self):
        assert research_agent.clean_title(
            "A soft voting ensemble learning approach for credit card fraud detection - ScienceDirect"
        ) == "A soft voting ensemble learning approach for credit card fraud detection"

    def test_keeps_ordinary_hyphen(self):
        title = "Fraud detection - a graph neural network approach"
        assert research_agent.clean_title(title) == title

    def test_strips_html_tags(self):
        assert research_agent.clean_title("<strong>Fraud</strong> detection") == "Fraud detection"


class TestIsRetracted:
    def test_title_prefix(self):
        assert research_agent.is_retracted("Retracted: Anomaly Detection in Credit Card Transaction")

    def test_snippet_notice(self):
        assert research_agent.is_retracted("Some paper", "This article has been retracted by the publisher.")

    def test_normal_paper(self):
        assert not research_agent.is_retracted("Detecting retraction patterns in citation graphs")


class TestExtractPapers:
    def test_copies_identifiers_from_search_result_not_llm(self):
        results = [brave_result(
            "Deep fraud detection - ScienceDirect",
            "https://doi.org/10.1016/j.eswa.2023.12345",
            "We evaluate on the IEEE-CIS dataset.",
        )]
        llm_out = json.dumps([{"index": 1, "year": 2023, "venue": "Expert Systems with Applications",
                               "peer_reviewed": True, "parent_paper_score": "7",
                               "title": "HALLUCINATED TITLE", "paper_url": "https://fake.example"}])
        with patch("src.llm.chat", return_value=llm_out):
            papers = research_agent._extract_papers_from_results(results, "fraud", "q")

        assert len(papers) == 1
        p = papers[0]
        assert p["title"] == "Deep fraud detection"
        assert p["paper_url"] == "https://doi.org/10.1016/j.eswa.2023.12345"
        assert p["doi"] == "10.1016/j.eswa.2023.12345"
        assert p["abstract"] == "We evaluate on the IEEE-CIS dataset."
        assert p["parent_paper_score"] == 7

    def test_ignores_out_of_range_and_duplicate_indices(self):
        results = [brave_result("Paper A", "https://a.org")]
        llm_out = json.dumps([{"index": 1}, {"index": 1}, {"index": 5}, {"index": "x"}])
        with patch("src.llm.chat", return_value=llm_out):
            papers = research_agent._extract_papers_from_results(results, "fraud", "q")
        assert [p["title"] for p in papers] == ["Paper A"]

    def test_no_llm_call_for_empty_results(self):
        with patch("src.llm.chat") as chat:
            assert research_agent._extract_papers_from_results([], "fraud", "q") == []
        chat.assert_not_called()


class TestFillMissingYear:
    def test_accepts_exact_title_match(self):
        paper = {"title": "Fraud Detection with GNNs", "year": 0, "venue": "", "doi": "", "notes": ""}
        match = {"title": "Fraud detection with GNNs.", "year": 2024, "venue": "KDD",
                 "externalIds": {"DOI": "10.1145/1"}}
        with patch("src.search.openalex.match_title", return_value=None),              patch("src.search.semantic_scholar.match_title", return_value=match):
            research_agent._fill_missing_year(paper)
        assert paper["year"] == 2024
        assert paper["venue"] == "KDD"
        assert paper["doi"] == "10.1145/1"

    def test_rejects_different_paper(self):
        paper = {"title": "Fraud Detection with GNNs", "year": 0, "venue": "", "doi": "", "notes": ""}
        match = {"title": "A completely different paper", "year": 2019}
        with patch("src.search.openalex.match_title", return_value={"title": "Other", "year": 2020}),              patch("src.search.semantic_scholar.match_title", return_value=match):
            research_agent._fill_missing_year(paper)
        assert paper["year"] == 0

    def test_prefers_openalex_doi_lookup(self):
        paper = {"title": "Fraud Detection with GNNs", "year": 0, "venue": "", "doi": "10.1/x", "notes": ""}
        record = {"title": "Fraud detection with GNNs", "year": 2023, "venue": "J", "doi": "10.1/x"}
        with patch("src.search.openalex.work_by_doi", return_value=record),              patch("src.search.semantic_scholar.match_title") as s2:
            research_agent._fill_missing_year(paper)
        assert paper["year"] == 2023
        assert "OpenAlex" in paper["notes"]
        s2.assert_not_called()


class TestRun:
    def test_dedupes_filters_and_skips_retracted(self):
        per_query = {
            "q1": [brave_result("Paper A - ScienceDirect", "https://sd.com/a"),
                   brave_result("Retracted: Paper B", "https://ieee.org/b")],
            "q2": [brave_result("Paper A | Springer", "https://springer.com/a"),
                   brave_result("Old Paper", "https://old.org/c")],
        }
        years = {"https://sd.com/a": 2023, "https://ieee.org/b": 2023,
                 "https://springer.com/a": 2023, "https://old.org/c": 2019}

        def fake_extract(results, topic, query):
            return [{"title": research_agent.clean_title(r["title"]), "paper_url": r["url"],
                     "abstract": "", "year": years[r["url"]], "parent_paper_score": 5}
                    for r in results]

        with patch("src.search.web_search.search_academic_papers", side_effect=lambda q, count: per_query[q]), \
             patch.object(research_agent, "_extract_papers_from_results", side_effect=fake_extract):
            papers = research_agent.run({"project_topic": "fraud", "search_queries": ["q1", "q2"]})

        assert [p["title"] for p in papers] == ["Paper A"]

    def test_one_failed_query_does_not_lose_the_others(self):
        def fake_search(q, count):
            if q == "bad":
                raise RuntimeError("boom")
            return [brave_result("Paper A", "https://a.org")]

        with patch("src.search.web_search.search_academic_papers", side_effect=fake_search), \
             patch.object(research_agent, "_extract_papers_from_results",
                          return_value=[{"title": "Paper A", "paper_url": "https://a.org",
                                         "abstract": "", "year": 2024}]):
            papers = research_agent.run({"project_topic": "fraud", "search_queries": ["bad", "good"]})

        assert len(papers) == 1
