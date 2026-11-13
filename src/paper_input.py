"""Local, immutable PDF inputs shared by the dashboard and pipeline.

This module has no agent, dashboard, network, or persistence dependencies.
"""
import io
import json
import unicodedata
from hashlib import sha256
from pathlib import Path

MAX_PDF_BYTES = 50 * 1024 * 1024
MIN_TEXT_CHARS = 2000


class PaperInputError(ValueError):
    pass


def read_pdf(content: bytes) -> tuple[str, int]:
    if not content.startswith(b"%PDF") or len(content) > MAX_PDF_BYTES:
        raise PaperInputError("Choose a PDF no larger than 50 MB.")
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise PaperInputError("Use an unencrypted PDF.")
        if len(reader.pages) > 1000:
            raise PaperInputError("The PDF must have no more than 1,000 pages.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except PaperInputError:
        raise
    except Exception as error:
        raise PaperInputError("The PDF could not be read. Choose a valid, text-based PDF.") from error
    if len(text) < MIN_TEXT_CHARS:
        raise PaperInputError("The PDF needs readable paper text (at least 2,000 characters). Scanned PDFs need OCR first.")
    return text, len(reader.pages)


def _fingerprint(manifest: dict) -> str:
    fields = {k: manifest[k] for k in ("schema_version", "topic", "title", "target_terms", "pdf_sha256")}
    return sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_input(folder: Path, content: bytes, topic: str, title: str, targets: str) -> tuple[Path, dict]:
    title = " ".join((title or "").split())
    target_terms = list(dict.fromkeys(t.strip().lower() for t in (targets or "").split(",") if t.strip()))
    if not title or len(title) > 500:
        raise PaperInputError("Enter the title printed on the PDF (up to 500 characters).")
    if not target_terms or len(target_terms) > 5 or any(len(t) > 80 for t in target_terms):
        raise PaperInputError("Enter 1–5 prediction-target keywords, separated by commas (for example fraud).")
    text, pages = read_pdf(content)
    def normalize(value):
        return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())
    if normalize(title) not in normalize(text[:10000]):
        raise PaperInputError("The title was not found at the start of the PDF. Enter the title printed on the paper.")
    manifest = {"schema_version": 1, "topic": topic, "title": title, "target_terms": target_terms,
                "pdf_sha256": sha256(content).hexdigest(), "pages": pages}
    manifest["input_fingerprint"] = _fingerprint(manifest)
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / f"{manifest['pdf_sha256']}.pdf"
    path = folder / f"{manifest['input_fingerprint']}.json"
    # Exclusive creation avoids replacing an input a detached run is reading.
    for dest, data in ((pdf, content), (path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))):
        try:
            with dest.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            if dest.read_bytes() != data:
                raise PaperInputError("A stored paper input has changed; restore it before starting a run.")
    return path, manifest


def load_input(path: Path, topic: str | None = None) -> tuple[dict, Path]:
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        digest = manifest["pdf_sha256"]
        if manifest["schema_version"] != 1 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid manifest")
        if manifest["input_fingerprint"] != _fingerprint(manifest):
            raise ValueError("input settings changed")
        if topic is not None and manifest["topic"] != topic:
            raise ValueError("topic differs from the stored paper input")
        pdf = path.parent / f"{digest}.pdf"
        if pdf.stat().st_size > MAX_PDF_BYTES or sha256(pdf.read_bytes()).hexdigest() != digest:
            raise ValueError("PDF checksum mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PaperInputError(f"The saved Parent Paper input is unavailable or has changed: {error}") from error
    return manifest, pdf
