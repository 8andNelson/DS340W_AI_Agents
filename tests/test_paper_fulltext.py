from unittest.mock import MagicMock, patch

from src.search import paper_fulltext

# conftest replaces candidate_pdf_urls (so no test downloads a paper); keep
# the real one for the tests of it below.
REAL_CANDIDATE_PDF_URLS = paper_fulltext.candidate_pdf_urls

PAPER = {"title": "Fraud Detection With Gradient Boosting", "doi": "10.1/abc",
         "official_abstract": "We detect fraud.", "methodology": "XGBoost", "results_summary": "AUC 0.98"}


def _json_response(payload, status=200):
    response = MagicMock(status_code=status)
    response.json.return_value = payload
    return response


def test_candidate_urls_come_from_openalex_and_skip_reference_lists():
    work = {"best_oa_location": {"pdf_url": "https://pub.example/paper_reference.pdf"},
            "primary_location": {"pdf_url": "https://pub.example/paper.pdf"},
            "locations": [{"pdf_url": "https://pub.example/paper.pdf"}, {"pdf_url": None}],
            "open_access": {"oa_url": "https://repo.example/oa.pdf"}}
    with patch("requests.get", return_value=_json_response(work)) as get:
        urls = REAL_CANDIDATE_PDF_URLS(PAPER)
    assert get.call_args.args[0].endswith("/doi:10.1/abc")
    assert urls == ["https://pub.example/paper.pdf", "https://repo.example/oa.pdf"]


def test_arxiv_page_becomes_its_pdf_link():
    paper = {"title": "T", "paper_url": "https://arxiv.org/abs/2301.00001"}
    with patch("requests.get", return_value=_json_response({}, status=404)):
        assert REAL_CANDIDATE_PDF_URLS(paper) == ["https://arxiv.org/pdf/2301.00001"]


def test_title_search_must_match_the_exact_title():
    paper = {"title": "Exact Title"}
    other = {"results": [{"title": "A Different Paper", "best_oa_location": {"pdf_url": "https://x/p.pdf"}}]}
    with patch("requests.get", return_value=_json_response(other)):
        assert REAL_CANDIDATE_PDF_URLS(paper) == []


def test_full_text_is_downloaded_once_and_cached():
    text = PAPER["title"] + "\n" + "Methods. " * 400
    with patch.object(paper_fulltext, "candidate_pdf_urls", return_value=["https://x/p.pdf"]), \
         patch.object(paper_fulltext, "_download_pdf", return_value=b"%PDF-1.7 fake") as download, \
         patch.object(paper_fulltext, "extract_text", return_value=(text, 9)):
        first = paper_fulltext.get_parent_paper_text(PAPER)
        second = paper_fulltext.get_parent_paper_text(PAPER)

    download.assert_called_once()
    assert first["status"] == second["status"] == "FULL_TEXT"
    assert first["pages"] == 9 and first["source_url"] == "https://x/p.pdf"
    assert second["text"] == text.strip() or second["text"] == text
    assert first["pdf_path"].endswith(".pdf")


def test_without_an_open_access_pdf_only_metadata_is_used_and_said_so():
    result = paper_fulltext.get_parent_paper_text(PAPER)  # conftest: no links
    assert result["status"] == "ABSTRACT_ONLY"
    assert "We detect fraud." in result["text"] and "AUC 0.98" in result["text"]
    assert "NOT the full paper" in result["notes"]


def test_a_too_short_extraction_does_not_count_as_full_text():
    with patch.object(paper_fulltext, "candidate_pdf_urls", return_value=["https://x/p.pdf"]), \
         patch.object(paper_fulltext, "_download_pdf", return_value=b"%PDF"), \
         patch.object(paper_fulltext, "extract_text", return_value=("cover page", 1)):
        assert paper_fulltext.get_parent_paper_text(PAPER)["status"] == "ABSTRACT_ONLY"


def test_download_rejects_html_served_as_a_pdf():
    response = MagicMock(status_code=200)
    response.iter_content.return_value = [b"<html>login</html>"]
    with patch("requests.get", return_value=response) as get:
        assert paper_fulltext._download_pdf("https://x/p.pdf") is None
    assert "Pipeline" in get.call_args.kwargs["headers"]["User-Agent"]  # honest client identification


def test_excerpt_keeps_the_opening_and_the_relevant_passages_in_order():
    head = "Title and abstract. " * 80
    filler = "\n\n".join(f"Unrelated paragraph {i} about history and philosophy." for i in range(200))
    relevant = "We applied SMOTE to the training split only, then trained XGBoost with learning rate 0.1."
    text = f"{head}\n\n{filler}\n\n{relevant}\n\n{filler}"
    out = paper_fulltext.excerpt(text, paper_fulltext.METHOD_KEYWORDS, max_chars=3000)
    assert len(out) <= 3000 + 20
    assert out.startswith("Title and abstract.")
    assert relevant in out
    assert out == paper_fulltext.excerpt(text, paper_fulltext.METHOD_KEYWORDS, max_chars=3000)


def test_short_text_is_returned_whole():
    assert paper_fulltext.excerpt("short paper", paper_fulltext.DATA_KEYWORDS) == "short paper"


def test_a_pdf_the_user_saved_is_read_without_downloading():
    folder = paper_fulltext.PARENT_PAPER_DIR
    folder.mkdir(parents=True)
    pdf = folder / f"{paper_fulltext._slug(PAPER['title'])}.pdf"
    pdf.write_bytes(b"%PDF-1.7 saved by the user")
    with patch.object(paper_fulltext, "extract_text", return_value=("Full text. " * 300, 12)), \
         patch.object(paper_fulltext, "_download_pdf") as download:
        result = paper_fulltext.get_parent_paper_text(PAPER)
    download.assert_not_called()
    assert result["status"] == "FULL_TEXT" and result["pages"] == 12
    assert str(pdf) in result["notes"]
