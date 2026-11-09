"""Web search: Brave first, SerpApi (Google) when Brave fails or reaches
its limit, and every call logged for the monthly budgets."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.search import brave_search, serpapi_search, web_search

SERPAPI_REPLY = {"organic_results": [
    {"link": "https://www.sciencedirect.com/science/article/pii/S1877050923002314",
     "title": "Unbalanced Credit Card Fraud Detection Data", "snippet": "Procedia Computer Science"},
    {"title": "no link -- dropped"},
]}
BRAVE_RESULTS = [{"title": "t", "url": "https://u", "description": "d"}]


@pytest.fixture
def both(monkeypatch):
    monkeypatch.setattr(brave_search, "BRAVE_API_KEY", "brave-key")
    monkeypatch.setattr(serpapi_search, "SERPENT_API_KEY", "serp-key")


def reply(payload, status=200):
    response = MagicMock(status_code=status)
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} error", response=response)
    return response


def log_lines():
    return [json.loads(line) for line in web_search.SEARCH_LOG.read_text().splitlines()]


def test_brave_answers_first(both):
    with patch.object(brave_search, "search", return_value=BRAVE_RESULTS) as brave, \
         patch.object(serpapi_search, "search") as serp:
        assert web_search.search("q", count=5) == BRAVE_RESULTS
    brave.assert_called_once_with("q", count=5)
    serp.assert_not_called()
    assert log_lines()[0]["provider"] == "brave"


def test_brave_at_its_limit_switches_to_serpapi(both, capsys):
    with patch("requests.get", side_effect=[reply({}, status=429), reply(SERPAPI_REPLY)]) as get:
        results = web_search.search("credit card fraud", count=5)

    assert results == [{"title": "Unbalanced Credit Card Fraud Detection Data",
                         "url": "https://www.sciencedirect.com/science/article/pii/S1877050923002314",
                         "description": "Procedia Computer Science"}]
    serp_call = get.call_args_list[1]
    assert serp_call.args[0] == serpapi_search.SERPAPI_URL
    assert serp_call.kwargs["params"] == {"engine": "google", "q": "credit card fraud", "num": 5,
                                          "api_key": "serp-key"}
    assert [(e["provider"], e.get("error")) for e in log_lines()] == [("brave", "HTTP 429"), ("serpapi", None)]
    assert "brave failed (HTTP 429); trying serpapi" in capsys.readouterr().out


def test_a_spent_brave_quota_is_skipped_for_the_rest_of_the_run(both):
    with patch.object(brave_search, "search", side_effect=requests.HTTPError("429")) as brave, \
         patch.object(serpapi_search, "search", return_value=BRAVE_RESULTS) as serp:
        for _ in range(5):
            web_search.search("q")
    assert brave.call_count == web_search.MAX_CONSECUTIVE_FAILURES
    assert serp.call_count == 5


def test_serpapi_errors_in_the_body_count_as_failures(monkeypatch):
    monkeypatch.setattr(serpapi_search, "SERPENT_API_KEY", "serp-key")
    with patch("requests.get", return_value=reply({"error": "Your account has run out of searches."})):
        with pytest.raises(RuntimeError, match="run out of searches"):
            serpapi_search.search("q")


def test_every_provider_failing_raises_with_each_reason(both):
    with patch.object(brave_search, "search", side_effect=requests.ConnectionError("down")), \
         patch.object(serpapi_search, "search", side_effect=RuntimeError("quota")):
        with pytest.raises(RuntimeError, match="brave: ConnectionError.*serpapi: RuntimeError"):
            web_search.search("q")


def test_without_keys_a_clear_error_is_raised(monkeypatch):
    monkeypatch.setattr(brave_search, "BRAVE_API_KEY", "")
    with pytest.raises(RuntimeError, match="none is configured"):
        web_search.search("q")


def test_counts_are_capped_at_bravess_limit(both):
    with patch.object(brave_search, "search", return_value=[]) as brave:
        web_search.search("q", count=50)
    brave.assert_called_once_with("q", count=web_search.MAX_COUNT)


def test_site_helpers_use_the_router(both):
    with patch.object(brave_search, "search", return_value=[]) as brave:
        web_search.search_academic_papers("fraud", count=4)
    assert brave.call_args.args[0].startswith("fraud site:arxiv.org OR")
