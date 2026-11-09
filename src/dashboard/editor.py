"""
Opens a file in VS Code on the user's own machine.

On Windows the `code` command is a .cmd batch file. Passing a file path to a
batch file goes through cmd.exe, which reads characters such as & and % in
the path as commands, so the dashboard starts Code.exe directly instead.
"""
import os
import shutil
import subprocess
from pathlib import Path


class EditorError(Exception):
    """VS Code could not be opened; the message is shown to the user."""


def find_vscode() -> Path | None:
    if os.name != "nt":
        found = shutil.which("code")
        return Path(found) if found else None
    candidates = []
    cli = shutil.which("code")
    if cli:
        # <install>\bin\code.cmd -> <install>\Code.exe
        candidates.append(Path(cli).resolve().parent.parent / "Code.exe")
    # Default per-user and system-wide install folders.
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Microsoft VS Code" / "Code.exe")
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var]) / "Microsoft VS Code" / "Code.exe")
    return next((c for c in candidates if c.is_file()), None)


def open_in_vscode(path: Path) -> None:
    exe = find_vscode()
    if exe is None:
        raise EditorError("VS Code was not found. Install it, or download the script instead.")
    env = dict(os.environ)
    env.pop("ELECTRON_RUN_AS_NODE", None)  # would start Code.exe as plain Node, not the editor
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    try:
        subprocess.Popen([str(exe), str(path)], env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    except OSError as e:
        raise EditorError(f"VS Code could not be started: {e}") from e
