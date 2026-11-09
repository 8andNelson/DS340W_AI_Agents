"""Direct retrieval rejects blocked pages and never invokes a paid service."""
import json
from unittest.mock import patch

import pytest
import requests

from src.agents import validation_agent
from src.search import page_fetch, paper_fulltext

CHALLENGE = b"<html><title>Just a moment...</title><body>cf-chl-bypass</body></html>"
NORMAL_PAGE = b"<html><body>Dataset of transactions. <script src='recaptcha.js'></script></body></html>"


def response(status=200, body=b"", content_type="text/html"):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    result.headers["Content-Type"] = content_type
    result.url = "https://example.com/article"
    return result


def test_block_detection():
    assert "HTTP 403" in page_fetch.blocked_reason(403)
    assert "HTTP 429" in page_fetch.blocked_reason(429)
    assert page_fetch.blocked_reason(200, "text/html", CHALLENGE)
    assert page_fetch.blocked_reason(200, "text/html", NORMAL_PAGE) == ""
    assert page_fetch.blocked_reason(404) == ""
    assert page_fetch.blocked_reason(200, "text/csv", b"a,b\n1,2") == ""


def test_accessible_page_is_returned_without_external_service():
    direct = response(body=NORMAL_PAGE)
    with patch("requests.get", return_value=direct) as get, patch("requests.post") as post:
        assert page_fetch.get("https://example.com/article") is direct
    get.assert_called_once()
    post.assert_not_called()
    entry = json.loads(page_fetch.FETCH_LOG.read_text().splitlines()[0])
    assert entry["verdict"] == "ACCESSIBLE"


@pytest.mark.parametrize("status, body", [
    (200, CHALLENGE), (403, b""), (429, b""), (200, b"<html><title>Sign in</title></html>"),
])
def test_blocked_response_is_rejected_without_retries_or_proxy(status, body):
    direct = response(status, body)
    with patch("requests.get", return_value=direct) as get, patch("requests.post") as post:
        with pytest.raises(page_fetch.BlockedPageError) as error:
            page_fetch.get("https://example.com/article", stream=True)
    assert error.value.response.status_code == status
    get.assert_called_once()
    post.assert_not_called()
    assert json.loads(page_fetch.FETCH_LOG.read_text().splitlines()[0])["verdict"] != "ACCESSIBLE"


def test_dead_link_is_available_for_classification():
    direct = response(404)
    with patch("requests.get", return_value=direct), patch("requests.post") as post:
        assert page_fetch.get("https://example.com/dead") is direct
    post.assert_not_called()


def test_network_failure_is_logged_then_raised():
    with patch("requests.get", side_effect=requests.ConnectionError("offline")):
        with pytest.raises(requests.ConnectionError):
            page_fetch.get("https://example.com/article")
    assert json.loads(page_fetch.FETCH_LOG.read_text().splitlines()[0])["verdict"] == "FETCH_FAILED"


def test_blocked_pdf_is_not_mistaken_for_full_text():
    with patch("requests.get", return_value=response(200, CHALLENGE)), patch("requests.post") as post:
        assert paper_fulltext._download_pdf("https://example.com/paper.pdf") is None
    post.assert_not_called()


def test_validation_preserves_blocked_http_status():
    with patch("requests.get", return_value=response(403)), patch("requests.post") as post:
        assert validation_agent._check_url_reachable("https://publisher.example/paper") == (False, 403)
    post.assert_not_called()
