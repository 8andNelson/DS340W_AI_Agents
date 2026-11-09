import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.search import camoufox_fetch, camoufox_worker, page_fetch, paper_fulltext

HTML = b"<html><title>An actual paper</title><body>Abstract</body></html>"
CHALLENGE = b"<html><title>Just a moment...</title></html>"
URL = "https://publisher.example/paper"


def response(status=200, body=HTML, mime="text/html"):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    result.headers["Content-Type"] = mime
    result.url = URL
    return result


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(camoufox_fetch, "CAMOUFOX_ENABLED", True)


@pytest.mark.parametrize("status,body", [(403, CHALLENGE), (200, CHALLENGE), (503, b"")])
def test_browser_fallback_only_on_blocked_pages(enabled, status, body):
    direct, rendered = response(status, body), response()
    with patch("requests.get", return_value=direct), patch.object(camoufox_fetch, "fetch", return_value=rendered) as fetch:
        assert page_fetch.get(URL) is rendered
    fetch.assert_called_once_with(URL, expect_pdf=False)
    entries = [json.loads(line) for line in page_fetch.FETCH_LOG.read_text().splitlines()]
    assert [entry["provider"] for entry in entries] == ["direct", "camoufox"]


@pytest.mark.parametrize("status,body", [(200, HTML), (404, b""), (429, b""),
                                        (200, b"<html><title>Sign in</title></html>")])
def test_no_browser_for_success_dead_links_logins_or_rate_limits(enabled, status, body):
    with patch("requests.get", return_value=response(status, body)), patch.object(camoufox_fetch, "fetch") as fetch:
        try:
            page_fetch.get(URL)
        except page_fetch.BlockedPageError:
            pass
    fetch.assert_not_called()


@pytest.mark.parametrize("credentials", [{"headers": {"Authorization": "private"}},
    {"headers": {"cookie": "private"}}, {"auth": ("user", "password")}, {"cookies": {"session": "private"}}])
def test_authenticated_requests_stay_direct(enabled, credentials):
    with patch("requests.get", return_value=response(403, CHALLENGE)), patch.object(camoufox_fetch, "fetch") as fetch:
        with pytest.raises(page_fetch.BlockedPageError):
            page_fetch.get(URL, **credentials)
    fetch.assert_not_called()


def test_still_blocked_browser_response_never_counts_as_success(enabled):
    with patch("requests.get", return_value=response(403, CHALLENGE)), patch.object(
            camoufox_fetch, "fetch", return_value=response(200, CHALLENGE)):
        with pytest.raises(page_fetch.BlockedPageError) as error:
            page_fetch.get(URL)
    assert error.value.response.status_code == 403


def test_browser_failure_preserves_original_verdict(enabled):
    with patch("requests.get", return_value=response(403, CHALLENGE)), patch.object(
            camoufox_fetch, "fetch", side_effect=requests.Timeout("bounded timeout")):
        with pytest.raises(page_fetch.BlockedPageError):
            page_fetch.get(URL)
    assert "bounded timeout" in page_fetch.FETCH_LOG.read_text()


def test_fulltext_requests_single_session_pdf_mode(enabled):
    pdf = b"%PDF-1.7 a paper"
    with patch("requests.get", return_value=response(403, CHALLENGE)), patch.object(
            camoufox_fetch, "fetch", return_value=response(200, pdf, "application/pdf")) as fetch:
        assert paper_fulltext._download_pdf(URL) == pdf
    fetch.assert_called_once_with(URL, expect_pdf=True)


def test_pdf_mode_rejects_browser_landing_page(enabled):
    with patch("requests.get", return_value=response(403, CHALLENGE)), patch.object(
            camoufox_fetch, "fetch", return_value=response()):
        assert paper_fulltext._download_pdf(URL) is None


def fake_worker(body=HTML, status=200, mime="text/html"):
    def spawn(args, **kwargs):
        folder = Path(args[-1]).parent
        (folder / "body.bin").write_bytes(body)
        (folder / "result.json").write_text(json.dumps({"status": status, "url": URL,
            "headers": {"Content-Type": mime}}))
        child = MagicMock()
        child.communicate.return_value = (b"", b"")
        return child
    return spawn


def test_worker_response_is_compatible_with_streamed_callers(enabled):
    with patch.object(camoufox_fetch.subprocess, "Popen", side_effect=fake_worker()):
        result = camoufox_fetch.fetch(URL)
    assert result.status_code == 200
    assert result.fetch_provider == "camoufox"
    assert b"".join(result.iter_content(8)) == HTML
    assert "actual paper" in result.text


def test_worker_challenge_is_rejected(enabled):
    with patch.object(camoufox_fetch.subprocess, "Popen", side_effect=fake_worker(CHALLENGE)):
        with pytest.raises(requests.RequestException, match="Camoufox:"):
            camoufox_fetch.fetch(URL)


def test_hard_timeout_terminates_worker_tree(enabled):
    process = MagicMock()
    process.communicate.side_effect = subprocess.TimeoutExpired("worker", 180)
    with patch.object(camoufox_fetch.subprocess, "Popen", return_value=process), patch.object(
            camoufox_fetch, "_terminate_tree") as terminate:
        with pytest.raises(requests.Timeout, match="time limit"):
            camoufox_fetch.fetch(URL)
    terminate.assert_called_once_with(process)


def test_worker_refuses_oversized_download_before_read(tmp_path, monkeypatch):
    download = MagicMock(url=URL + ".pdf", suggested_filename="article.pdf")
    path = tmp_path / "oversized.pdf"
    path.write_bytes(b"%PDF-too-large")
    download.path.return_value = path
    monkeypatch.setattr(camoufox_worker, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds"):
        camoufox_worker._read_download(download, tmp_path, True)


def test_worker_cancels_full_issue_before_waiting_for_download(tmp_path):
    download = MagicMock(url=URL + "/full-issue.pdf", suggested_filename="Full issue.pdf")
    with pytest.raises(ValueError, match="bulk"):
        camoufox_worker._read_download(download, tmp_path, True)
    download.cancel.assert_called_once()
    download.path.assert_not_called()


def test_worker_rejects_html_disguised_as_download(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(CHALLENGE)
    download = MagicMock(url=URL + ".pdf", suggested_filename="paper.pdf")
    download.path.return_value = path
    with pytest.raises(ValueError, match="not an individual PDF"):
        camoufox_worker._read_download(download, tmp_path, True)


def test_browser_disabled_never_starts_worker():
    with patch.object(camoufox_fetch.subprocess, "Popen") as popen:
        with pytest.raises(requests.RequestException, match="disabled"):
            camoufox_fetch.fetch(URL)
    popen.assert_not_called()


def test_worker_follows_individual_pdf_in_the_same_context(tmp_path):
    article = "https://www.sciencedirect.com/science/article/pii/S1877050923002314"
    pdf_url = article + "/pdf"
    body = ('<html><a href="/full-issue.zip">Download full issue</a><a href="' +
            pdf_url + '">View PDF</a></html>').encode()
    downloaded = tmp_path / "browser.pdf"
    downloaded.write_bytes(b"%PDF-1.7 individual paper")
    download = MagicMock(url=pdf_url, suggested_filename="paper.pdf")
    download.path.return_value = downloaded
    browser = MagicMock()
    context = browser.new_context.return_value
    page = context.new_page.return_value
    page.url = article
    page.expect_download.return_value.__enter__.return_value.value = download
    factory = MagicMock()
    factory.return_value.__enter__.return_value = browser
    with patch("camoufox.sync_api.Camoufox", factory), patch("camoufox.pkgman.camoufox_path"), patch.object(
            camoufox_worker, "_render", return_value=(body, {"verdict": "ACCESSIBLE"})):
        result = camoufox_worker.run({"url": article, "expect_pdf": True}, tmp_path)
    page.goto.assert_called_once_with(pdf_url, wait_until="domcontentloaded", timeout=30000)
    assert result["headers"]["Content-Type"] == "application/pdf"
    assert (tmp_path / "body.bin").read_bytes() == downloaded.read_bytes()
    context.close.assert_called_once()


@pytest.mark.parametrize("url,body,mime", [
    ("https://github.com/example/repo", HTML, "text/html"),
    ("https://datasets.example/data.csv", b"a,b\n1,2", "text/csv"),
])
def test_generic_reachability_still_accepts_repositories_and_data(url, body, mime):
    from src.agents import validation_agent
    with patch("requests.get", return_value=response(200, body, mime)):
        assert validation_agent._check_url_reachable(url) == (True, 200)


def test_paper_identity_still_rejects_prohibited_academic_source():
    from src.agents import validation_agent
    with patch("requests.get") as get, patch("src.search.web_search.search", return_value=[]):
        assert not validation_agent._check_url_reachable("https://github.com/example/repo", title="Some Paper")[0]
    get.assert_not_called()


def test_browser_waits_for_final_response_after_challenge_redirect():
    page = MagicMock()
    page.content.return_value = HTML.decode()
    status = {"current": 302}
    ticks = []
    def advance(_milliseconds):
        ticks.append(1)
        if len(ticks) == 2:
            status["current"] = 200
    page.wait_for_timeout.side_effect = advance
    body, result = camoufox_worker._render(page, URL, status, [])
    assert body == HTML and result["verdict"] == "ACCESSIBLE"
    assert len(ticks) == 2


def test_pdf_mode_waits_for_article_controls_to_render():
    article = "https://www.sciencedirect.com/science/article/pii/S1877050923002314"
    ready = '<html><a href="' + article + '/pdf">View PDF</a></html>'
    page = MagicMock(url=article)
    page.content.side_effect = ["<html><title>Loading</title></html>", ready]
    body, result = camoufox_worker._render(page, article, {"current": 200}, [], expect_pdf=True)
    assert body.decode() == ready and result["verdict"] == "ACCESSIBLE"
    assert page.wait_for_timeout.call_count == 2


def test_worker_start_failure_is_a_fetch_error(enabled):
    with patch.object(camoufox_fetch.subprocess, "Popen", side_effect=OSError("cannot start")):
        with pytest.raises(requests.RequestException, match="worker failure"):
            camoufox_fetch.fetch(URL)
