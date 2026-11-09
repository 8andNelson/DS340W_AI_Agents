from unittest.mock import patch

import pytest
import requests

from src.search import page_fetch, paper_fulltext
from src.search.pdf_links import article_pdf_links

ARTICLE = "https://www.sciencedirect.com/science/article/pii/S1877050923002314"
PDF_ENDPOINT = ARTICLE + "/pdfft?isDTMRedir=true&download=true"
PDF = b"%PDF-1.7 article bytes"


def response(body, content_type="text/html", url=ARTICLE):
    result = requests.Response()
    result.status_code = 200
    result._content = body
    result._content_consumed = True
    result.headers["Content-Type"] = content_type
    result.url = url
    return result


def test_view_pdf_is_selected_and_full_issue_is_excluded():
    html = b'''<html><a href="/science/journal/18770509/218/supp/C">Download full issue</a>
        <a href="/science/article/pii/S1877050923002314/pdfft?isDTMRedir=true&amp;download=true">View PDF</a>
        <a href="/science/article/pii/S1877050923009999/pdfft">View PDF</a></html>'''
    assert article_pdf_links(html, ARTICLE) == [PDF_ENDPOINT]


def test_signed_asset_url_is_preserved_exactly():
    url = "https://pdf.sciencedirectassets.com/280203/issue/1-s2.0-S1877050923002314/main.pdf?X-Amz-Expires=300&X-Amz-Signature=example"
    html = ('<meta name="citation_pdf_url" content="' + url.replace("&", "&amp;") + '">').encode()
    assert article_pdf_links(html, ARTICLE) == [url]


def test_full_issue_label_excludes_even_a_pdf_named_link():
    html = ('<a href="' + PDF_ENDPOINT + '">Download full issue PDF</a>').encode()
    assert article_pdf_links(html, ARTICLE) == []


def test_javascript_control_is_not_treated_as_downloadable_url():
    assert article_pdf_links(b'<a href="javascript:openPdf()">View PDF</a>', ARTICLE) == []


def test_follows_only_article_pdf_and_downloads_bytes():
    html = ('<html><a href="/issue.zip">Download full issue</a><a href="' + PDF_ENDPOINT + '">View PDF</a></html>').encode()
    with patch.object(page_fetch, "get", side_effect=[response(html), response(PDF, "application/pdf", PDF_ENDPOINT)]) as get:
        assert paper_fulltext._download_pdf(ARTICLE) == PDF
    assert [call.args[0] for call in get.call_args_list] == [ARTICLE, PDF_ENDPOINT]


def test_issue_url_is_rejected_before_network_request():
    with patch.object(page_fetch, "get") as get:
        assert paper_fulltext._download_pdf("https://www.sciencedirect.com/science/issue/123/full-issue.pdf") is None
    get.assert_not_called()


def test_link_cycle_is_bounded():
    html = ('<html><a href="' + PDF_ENDPOINT + '">View PDF</a></html>').encode()
    with patch.object(page_fetch, "get", side_effect=[response(html), response(html, url=PDF_ENDPOINT)]) as get:
        assert paper_fulltext._download_pdf(ARTICLE) is None
    assert get.call_count == 2


