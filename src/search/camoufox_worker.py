"""One isolated Camoufox session. Invoked only by camoufox_fetch.fetch()."""
import json
import platform
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from src.search import content_validator
from src.search.pdf_links import article_pdf_links, is_bulk_download

MAX_BYTES = 50 * 1024 * 1024
MAX_CHECKBOX_CLICKS = 2
CHALLENGE_WAIT_SECONDS = 35


def _read_download(download, folder: Path, expect_pdf: bool):
    if is_bulk_download(download.url, download.suggested_filename):
        download.cancel()
        raise ValueError("Refusing full-issue or bulk download")
    # Read from Playwright's temporary file before the context closes; the
    # parent process deadline also covers completion of the download.
    path = Path(download.path())
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Browser download exceeds 50 MB")
    body = path.read_bytes()
    if expect_pdf and not body.startswith(b"%PDF"):
        raise ValueError("Browser download is not an individual PDF")
    (folder / "body.bin").write_bytes(body)
    return {"status": 200, "url": download.url,
            "headers": {"Content-Type": "application/pdf" if body.startswith(b"%PDF") else "application/octet-stream"}}


def _render(page, url: str, status: dict, downloads: list, expect_pdf: bool = False):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception:
        if downloads:
            return b"", None
        # A navigation timeout can leave a usable page, but an exception alone
        # is never success: assess the rendered response below.
    deadline = time.monotonic() + CHALLENGE_WAIT_SECONDS
    clicks, last_click = 0, 0.0
    while True:
        if downloads:
            return b"", None
        page.wait_for_timeout(2000)
        body = page.content().encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError("Rendered page exceeds 50 MB")
        result = content_validator.classify(status["current"], {"Content-Type": "text/html"}, body)
        if status["current"] == 0 or 300 <= status["current"] < 400:
            # Challenge completion can replace the main frame asynchronously.
            # Wait for its final response; a redirect is never access evidence.
            if time.monotonic() >= deadline:
                raise ValueError("Camoufox navigation did not reach a final response")
            continue
        if result["verdict"] == "ACCESSIBLE":
            if not expect_pdf or article_pdf_links(body, page.url):
                return body, result
            # Publisher apps can render the title before hydrating View PDF.
            # Keep this session alive until the individual PDF control exists.
            if time.monotonic() >= deadline:
                raise ValueError("Accessible page has no individual article PDF link")
            continue
        if result["verdict"] not in (*content_validator.BLOCKED_VERDICTS, "FORBIDDEN"):
            raise ValueError(result["reason"])
        if time.monotonic() >= deadline:
            raise ValueError("Camoufox could not clear the page challenge")
        if clicks >= MAX_CHECKBOX_CLICKS or time.monotonic() - last_click < 12:
            continue
        for frame in page.frames:
            if urlsplit(frame.url).hostname != "challenges.cloudflare.com":
                continue
            try:
                element = frame.frame_element()
                box = element.bounding_box()
                if box and box["width"] >= 200 and box["height"] >= 50:
                    clicks += 1
                    last_click = time.monotonic()
                    element.click(position={"x": 25, "y": min(32, box["height"] / 2)}, timeout=5000)
                    break
            except Exception:
                # Frames are often replaced while the challenge runs.
                continue


def run(request: dict, folder: Path) -> dict:
    # Lazy imports keep agent startup cheap and allow clear setup failures.
    from camoufox.sync_api import Camoufox
    from camoufox.pkgman import camoufox_path
    camoufox_path(download_if_missing=False)
    url, expect_pdf = request["url"], request["expect_pdf"]
    if urlsplit(url).scheme not in ("http", "https") or is_bulk_download(url):
        raise ValueError("Refusing unsupported or bulk download URL")
    native_os = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    with Camoufox(headless=True, os=native_os, humanize=True, disable_coop=True,
                  i_know_what_im_doing=True, firefox_user_prefs={"pdfjs.disabled": True}) as browser:
        context = browser.new_context(accept_downloads=True)
        try:
            def guard(route):
                if route.request.is_navigation_request() and is_bulk_download(route.request.url):
                    route.abort()
                else:
                    route.continue_()
            context.route("**/*", guard)
            page = context.new_page()
            status, downloads = {"current": 0}, []
            def on_download(download):
                downloads.append(download)
            page.on("download", on_download)

            def track(response):
                if response.request.is_navigation_request() and response.frame == page.main_frame:
                    status["current"] = response.status
            page.on("response", track)
            body, _ = _render(page, url, status, downloads, expect_pdf=expect_pdf)
            if downloads:
                return _read_download(downloads[0], folder, expect_pdf)
            if expect_pdf:
                links = article_pdf_links(body, page.url)
                if not links:
                    raise ValueError("Accessible page has no individual article PDF link")
                with page.expect_download(timeout=30000) as pending:
                    try:
                        page.goto(links[0], wait_until="domcontentloaded", timeout=30000)
                    except Exception:
                        pass  # Firefox aborts navigation when Downloads takes over.
                return _read_download(pending.value, folder, True)
            (folder / "body.bin").write_bytes(body)
            return {"status": status["current"], "url": page.url, "headers": {"Content-Type": "text/html; charset=utf-8"}}
        finally:
            context.close()


def main():
    request_path = Path(sys.argv[1])
    folder = request_path.parent
    try:
        result = run(json.loads(request_path.read_text(encoding="utf-8")), folder)
    except ImportError:
        result = {"error": "Camoufox is not installed; install requirements and run python -m camoufox fetch"}
    except Exception as error:
        # Playwright errors may contain signed URLs: keep diagnostics free of
        # URL query credentials while retaining specific validation failures.
        detail = re.sub(r"https?://\S+", "[URL]", str(error).split("Call log:")[0])[:240]
        result = {"error": "Camoufox " + type(error).__name__ + ": " + detail}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
