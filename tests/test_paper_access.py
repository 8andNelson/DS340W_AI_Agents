from unittest.mock import MagicMock, patch
import io
import pytest
import requests
from src.agents import validation_agent
from src.search import content_validator, page_fetch, paper_access, paper_fulltext, web_search

TITLE = "Fraud Detection With Gradient Boosting"
ARTICLE = ("<html><head><title>" + TITLE + "</title></head><body>Abstract</body></html>").encode()
CHALLENGE = b"<html><title>Just a moment...</title><body>cf-chl-bypass</body></html>"


def response(status=200, body=b"", content_type="text/html"):
    r = MagicMock(status_code=status, content=body, headers={"Content-Type": content_type})
    r.iter_content.return_value = [body]
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(str(status), response=r)
    return r


def test_head_success_cannot_hide_challenge():
    with patch("requests.head") as head, patch("requests.get", return_value=response(body=CHALLENGE)):
        assert validation_agent._check_url_reachable("https://publisher.example/paper") == (False, 200)
    head.assert_not_called()


@pytest.mark.parametrize("body", [CHALLENGE, b"<html><title>Sign in</title></html>",
    b"<html><title>Page not found</title></html>", b"", b"<html>A different paper</html>"])
def test_bad_pages_never_prove_access(body):
    with patch("requests.get", return_value=response(body=body)), patch.object(web_search, "search", return_value=[]):
        result = paper_access.check("https://publisher.example/paper", title=TITLE)
    assert not result["reachable"]
    assert result["access_level"] == ""


def test_landing_page_is_distinct_from_full_text():
    with patch("requests.get", return_value=response(body=ARTICLE)), patch.object(web_search, "search") as search:
        result = paper_access.check("https://publisher.example/paper", title=TITLE)
    assert result["reachable"]
    assert result["access_level"] == "LANDING_PAGE"
    search.assert_not_called()


def test_serpapi_alternates_are_fetched_and_verified():
    with patch("requests.get", side_effect=[response(403), response(body=b"<html>Wrong paper</html>"), response(body=ARTICLE)]), patch.object(web_search, "search", return_value=[
            {"url": "https://wrong.example", "description": TITLE}, {"url": "https://repository.example/paper"}]) as search:
        result = paper_access.check("https://publisher.example/paper", title=TITLE)
    search.assert_called_once_with('"' + TITLE + '"', count=3, provider_name="serpapi")
    assert result["reachable"]
    assert result["source_url"] == "https://repository.example/paper"
    assert len(result["attempts"]) == 3


def test_snippet_alone_never_proves_access():
    with patch("requests.get", return_value=response(404)), patch.object(web_search, "search", return_value=[
            {"url": "https://other.example", "description": TITLE}]):
        assert not paper_access.check("https://publisher.example/paper", title=TITLE)["reachable"]


@pytest.mark.parametrize("text, reachable", [("Different title", False), (TITLE + " Methods" * 400, True), ("", False)])
def test_pdf_requires_extracted_identity(text, reachable):
    with patch("requests.get", return_value=response(body=b"%PDF-fake", content_type="application/pdf")), patch.object(
            paper_fulltext, "extract_text", return_value=(text, 2)), patch.object(web_search, "search", return_value=[]):
        result = paper_access.check("https://x/p.pdf", title=TITLE)
    assert result["reachable"] == reachable
    if reachable:
        assert result["access_level"] == "FULL_TEXT"






def test_stream_inspection_preserves_every_byte():
    original = b"a" * (page_fetch.CHALLENGE_SCAN_BYTES + 9000)
    direct = requests.Response()
    direct.status_code = 200
    direct.headers["Content-Type"] = "application/octet-stream"
    direct.raw = io.BytesIO(original)
    with patch("requests.get", return_value=direct):
        fetched = page_fetch.get("https://x", stream=True)
        assert b"".join(fetched.iter_content(16384)) == original




def test_title_on_challenge_does_not_override_block():
    body = CHALLENGE.replace(b"</body>", TITLE.encode() + b"</body>")
    assert content_validator.classify(200, {"Content-Type": "text/html"}, body, expected_text=TITLE)["verdict"] != "ACCESSIBLE"


def test_redirect_and_empty_response_are_not_access():
    assert content_validator.classify(302, {}, ARTICLE)["verdict"] == "HTTP_ERROR"
    assert content_validator.classify(204, {}, b"")["verdict"] == "WRONG_CONTENT"


def test_invalid_url_is_not_fetched():
    with patch("requests.get") as get:
        assert not paper_access.check("file:///local")["reachable"]
    get.assert_not_called()




def test_prohibited_alternative_does_not_count_as_paper():
    with patch("requests.get", return_value=response(404)) as get, patch.object(web_search, "search",
            return_value=[{"url": "https://medium.com/a-paper", "description": TITLE}]):
        result = paper_access.check("https://publisher.example/paper", title=TITLE,
                                   source_check=lambda url: "medium.com" not in url)
    assert not result["reachable"]
    assert get.call_count == 1


def test_llm_cannot_upgrade_inaccessible_paper():
    paper = {"title": TITLE, "paper_url": "https://publisher.example/paper", "year": 2024}
    def assess(papers):
        papers[0]["validation_status"] = "VERIFIED"
    with patch.object(validation_agent, "_check_url_reachable", return_value=(False, 200)), patch.object(
            validation_agent, "_llm_validate_batch", side_effect=assess), patch.object(validation_agent, "_attach_official_record"):
        result = validation_agent.run([paper])
    assert result["unverified"] == [paper]
    assert not result["verified"]


def test_short_pdf_cover_is_not_full_text():
    with patch("requests.get", return_value=response(body=b"%PDF-fake", content_type="application/pdf")), patch.object(
            paper_fulltext, "extract_text", return_value=(TITLE, 1)), patch.object(web_search, "search", return_value=[]):
        assert not paper_access.check("https://x/p.pdf", title=TITLE)["reachable"]


def test_paper_mentioned_only_in_references_is_not_the_requested_paper():
    other = ("<html><title>A different paper</title><h1>A different paper</h1><body>References: " + TITLE + "</body></html>").encode()
    with patch("requests.get", return_value=response(body=other)), patch.object(web_search, "search", return_value=[]):
        assert not paper_access.check("https://x/paper", title=TITLE)["reachable"]
