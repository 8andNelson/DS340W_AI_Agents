"""
Where a replication folder's venv lives.

Windows cannot load a DLL from a path longer than 259 characters, and the
files inside a venv go far below its folder: in a live run (fresh clone in
a deep folder) scikit-learn's vcomp140.dll was at 270 characters, and every
`import sklearn` failed with "[WinError 206] The filename or extension is
too long". The deepest file of that venv was 123 characters below the venv
folder, and larger packages (PyTorch, TensorFlow) go deeper.

So on Windows, a venv path longer than MAX_VENV_PATH_CHARS is not put in
the replication folder. It goes in a short folder under %LOCALAPPDATA%
instead, named from a hash of the replication folder, so the same folder
always gets the same venv. Nothing is needed from the user. On other
systems the venv is always <replication folder>/venv.
"""
import hashlib
import os
import tempfile
from pathlib import Path

MAX_VENV_PATH_CHARS = 100   # 259 (Windows) minus room for the deepest package files
SHORT_VENV_ROOT_NAME = "ds340w_venvs"


def short_venv_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / SHORT_VENV_ROOT_NAME


def venv_dir_for(replication_dir: Path, windows: bool = None) -> Path:
    """The venv folder for a replication folder (see the module docstring)."""
    replication_dir = Path(replication_dir)
    default = replication_dir / "venv"
    windows = os.name == "nt" if windows is None else windows
    if not windows or len(str(default.resolve())) <= MAX_VENV_PATH_CHARS:
        return default
    key = hashlib.sha256(str(replication_dir.resolve()).lower().encode("utf-8")).hexdigest()[:12]
    return short_venv_root() / key


def path_too_long(text: str) -> bool:
    """True when an error message is Windows' path-length error."""
    text = (text or "").lower()
    return "winerror 206" in text or "filename or extension is too long" in text
