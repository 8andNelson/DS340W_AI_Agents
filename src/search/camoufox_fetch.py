"""Bounded local-browser fallback, isolated from agent threads in a subprocess."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src.config import CAMOUFOX_ENABLED, CAMOUFOX_TIMEOUT_SECONDS
from src.search import content_validator

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 50 * 1024 * 1024
_slots = threading.BoundedSemaphore(2)


def _terminate_tree(process):
    """Stop only this worker and its browser children when its deadline expires."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def fetch(url: str, expect_pdf: bool = False) -> requests.Response:
    """Return validated browser content in the same shape as requests.get().

    No caller cookies/headers or API credentials are sent to this browser.
    PDF mode follows the individual article link in the SAME browser session.
    """
    if not CAMOUFOX_ENABLED:
        raise requests.RequestException("Camoufox fallback is disabled")
    if urlsplit(url).scheme not in ("http", "https"):
        raise requests.RequestException("Camoufox requires an HTTP(S) URL")
    deadline = time.monotonic() + CAMOUFOX_TIMEOUT_SECONDS
    if not _slots.acquire(timeout=CAMOUFOX_TIMEOUT_SECONDS):
        raise requests.Timeout("Camoufox browser queue timed out")
    try:
        with tempfile.TemporaryDirectory(prefix="agent-camoufox-") as temp:
            folder = Path(temp)
            request_path = folder / "request.json"
            request_path.write_text(json.dumps({"url": url, "expect_pdf": expect_pdf}), encoding="utf-8")
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            process = subprocess.Popen(
                [sys.executable, "-m", "src.search.camoufox_worker", str(request_path)],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
            try:
                process.communicate(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                _terminate_tree(process)
                raise requests.Timeout("Camoufox browser exceeded its time limit") from error
            result_path = folder / "result.json"
            if not result_path.is_file():
                raise requests.RequestException("Camoufox worker failed; install requirements and run python -m camoufox fetch")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("error"):
                raise requests.RequestException(result["error"])
            body_path = folder / "body.bin"
            if not body_path.is_file() or body_path.stat().st_size > MAX_BYTES:
                raise requests.RequestException("Camoufox returned no content or exceeded the download limit")
            body = body_path.read_bytes()
            assessment = content_validator.classify(result["status"], result["headers"], body,
                                                     expected="pdf" if expect_pdf else "any")
            if assessment["verdict"] != "ACCESSIBLE":
                raise requests.RequestException("Camoufox: " + assessment["reason"])
            response = requests.Response()
            response.status_code = result["status"]
            response.headers.update(result["headers"])
            response.url = result["url"]
            response._content = body
            response._content_consumed = True
            response.encoding = "utf-8" if "html" in response.headers.get("Content-Type", "") else None
            response.fetch_provider = "camoufox"
            return response
    except requests.RequestException:
        raise
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise requests.RequestException("Camoufox worker failure (" + type(error).__name__ + ")") from error
    finally:
        _slots.release()
