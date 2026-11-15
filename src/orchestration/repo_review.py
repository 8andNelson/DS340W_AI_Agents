"""
Master Agent's review of a discovered repository, before the Replication
Agent may install or run any of it.

review_repository(repository) clones the repository into REVIEW_DIR (git
clone --depth 1, https only, no submodules, symlinks checked out as plain
files, Git LFS files not downloaded) and executes nothing in it. The
Replication Agent later copies this exact clone, so the files that run are
the files that were reviewed (it checks the commit).

Checks, all deterministic:

* Valid for the pipeline: an https URL that clones, Python code (.py or
  .ipynb), at most MAX_REPO_FILES files and MAX_REPO_BYTES, and the Code
  Discovery Agent recorded its provenance (the paper cites it, or an author
  owns it).
* Malware patterns: the Python AST of every .py file and notebook code cell
  (exec/eval of decoded data, reverse shells, download-and-run commands,
  ctypes/winreg, credential and browser-data paths, persistence, webhook
  exfiltration, large encoded payloads), the text of shell scripts and
  notebook !commands, code in setup.py (pip runs it at install time),
  auto-run files (.pth, sitecustomize.py), executable binaries (also ones
  disguised by extension, found by their magic bytes) and symlinks.
* Dependencies: installs from outside PyPI (VCS/URL/local, extra index
  URLs, trusted hosts, find-links), names close to a popular package
  (typosquats), and known-malicious (OSV "MAL-") or vulnerable versions
  from OSV.dev (no key). OSV not reachable is reported, not a rejection.

Verdict: REJECTED on any critical or high finding; APPROVED_WITH_WARNINGS
on medium ones; APPROVED otherwise. A static review finds known patterns;
it cannot prove code is safe. Approved code still runs with this user's
permissions in a venv, which isolates packages, not the system.
"""
import ast
import os
import re
import subprocess
import warnings
from pathlib import Path

import requests

from src.orchestration import repo_dependencies

REPO_ROOT = Path(__file__).parent.parent.parent
REVIEW_DIR = REPO_ROOT / "replication" / "_reviewed"

CLONE_TIMEOUT_SECONDS = 120
OSV_URL = "https://api.osv.dev/v1/querybatch"
OSV_TIMEOUT_SECONDS = 20
MAX_REPO_FILES = 5000
MAX_REPO_BYTES = 500 * 1024 * 1024
MAX_SCANNED_FILE_BYTES = 2 * 1024 * 1024   # a larger .py file cannot be reviewed
MAX_FINDINGS = 200
ENCODED_PAYLOAD_CHARS = 2000
TYPOSQUAT_MIN_CHARS = 5   # shorter names are one letter from too many real packages

APPROVED_VERDICTS = ("APPROVED", "APPROVED_WITH_WARNINGS")
BLOCKING_SEVERITIES = ("critical", "high")
PROVENANCE_KINDS = ("paper_stated", "cited_in_paper", "author_owned")

# Executable formats that are never in research code: a rejection.
EXECUTABLE_EXTENSIONS = {".exe", ".scr", ".pif", ".com", ".hta", ".lnk", ".vbs", ".vbe", ".wsf", ".cpl", ".msi",
                         ".jar", ".ps1xml"}
# Native libraries cannot be reviewed either.
NATIVE_EXTENSIONS = {".dll", ".so", ".dylib", ".sys", ".pyd"}
SCRIPT_EXTENSIONS = {".sh", ".bash", ".bat", ".cmd", ".ps1"}
MAGIC_BYTES = {b"MZ": "a Windows executable", b"\x7fELF": "a Linux executable",
               b"\xcf\xfa\xed\xfe": "a macOS executable", b"\xfe\xed\xfa\xcf": "a macOS executable"}
AUTO_RUN_FILES = {"sitecustomize.py", "usercustomize.py"}

DECODERS = {"b64decode", "b32decode", "b85decode", "a85decode", "decodebytes", "decompress", "fromhex",
            "unhexlify", "loads", "decode"}
SHELL_CALLS = {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.Popen",
               "subprocess.check_output", "subprocess.check_call", "subprocess.getoutput",
               "subprocess.getstatusoutput", "os.execv", "os.execl", "os.execvp", "os.spawnl", "os.startfile"}
NETWORK_SEND_CALLS = {"requests.post", "requests.put", "requests.patch", "httpx.post", "httpx.put",
                      "urllib.request.urlopen", "socket.socket", "socket.create_connection", "smtplib.SMTP",
                      "smtplib.SMTP_SSL", "ftplib.FTP", "telnetlib.Telnet", "paramiko.SSHClient"}
NATIVE_MODULES = {"ctypes", "winreg", "_winreg", "_winapi", "win32api", "win32con", "pywintypes"}
UNSAFE_LOADS = {"pickle.load", "pickle.loads", "dill.load", "dill.loads", "cloudpickle.load", "shelve.open"}

DOWNLOAD_AND_RUN = re.compile(
    r"(curl|wget)[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b|invoke-webrequest|\biwr\b|downloadstring|downloadfile|"
    r"invoke-expression|\biex\b|certutil[^\n]*-urlcache|bitsadmin|powershell[^\n]*-e(nc|ncodedcommand)?\s|"
    r"\bnc\b[^\n]*-e\s|/dev/tcp/|bash\s+-i|mshta|rundll32|regsvr32|base64\s+(-d|--decode)[^\n]*\|\s*(ba)?sh",
    re.IGNORECASE)
# Paths, not words: "login data" alone can be a fraud dataset's column.
CREDENTIAL_MARKERS = ("id_rsa", "id_ed25519", "id_ecdsa", ".ssh/", ".ssh\\", ".aws/credentials",
                      ".aws\\credentials", ".git-credentials", "/.netrc", "\\.netrc", "default\\login data",
                      "default/login data", "user data\\local state", "user data/local state",
                      "network\\cookies", "network/cookies", "wallet.dat", "login.keychain", "/etc/shadow",
                      ".bash_history", "lsass.exe")
PERSISTENCE_MARKERS = ("currentversion\\run", "currentversion/run", "start menu\\programs\\startup",
                       "start menu/programs/startup", "crontab", "schtasks", "launchagents", "launchdaemons",
                       "/etc/systemd/system", "authorized_keys")
EXFIL_MARKERS = ("discord.com/api/webhooks", "discordapp.com/api/webhooks", "api.telegram.org/bot",
                 "hooks.slack.com/services", "pastebin.com/api", "ngrok.io", "transfer.sh")
RAW_IP_URL = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}")
ENCODED_PAYLOAD = re.compile(r"[A-Za-z0-9+/=\s]{%d,}" % ENCODED_PAYLOAD_CHARS)

POPULAR_PACKAGES = repo_dependencies.KNOWN_PACKAGES | {
    "requests", "urllib3", "setuptools", "pip", "wheel", "six", "python-dateutil", "pytz", "certifi", "idna",
    "charset-normalizer", "jupyter", "notebook", "ipywidgets", "tensorboard", "protobuf", "packaging", "attrs",
    "jinja2", "flask", "fastapi", "pyyaml", "pillow", "opencv-python", "beautifulsoup4", "jupyterlab",
}
# Real packages one letter from a popular one: not typosquats.
LEGITIMATE_NEAR_NAMES = {"pyaml", "pandas-ta", "numpy-financial", "tensorflow-cpu", "tensorflow-gpu", "keras-nlp",
                         "torchdata", "lightgbm", "shapely", "polars-lts-cpu"}


def _one_edit_apart(a: str, b: str) -> bool:
    """True when a and b differ by exactly one insert, delete, change, or
    swap of two neighbouring letters ("reqeusts" / "requests")."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                   and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def typosquat_of(name: str) -> str:
    """The popular package `name` is one edit away from, or ""."""
    if name in POPULAR_PACKAGES or name in LEGITIMATE_NEAR_NAMES or len(name) < TYPOSQUAT_MIN_CHARS:
        return ""
    return next((p for p in sorted(POPULAR_PACKAGES) if _one_edit_apart(name, p)), "")


def _finding(severity: str, rule: str, path: str, line: int, evidence: str) -> dict:
    return {"severity": severity, "rule": rule, "path": path, "line": line,
            "evidence": " ".join(str(evidence).split())[:200]}


def _call_name(node: ast.AST) -> str:
    """Dotted name of a call's function: os.system, subprocess.run, exec."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _strings_in(node: ast.AST) -> list:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _string_findings(text: str, rel: str, line: int) -> list:
    """Credential, persistence, exfiltration and payload markers in one string."""
    lowered = text.lower()
    found = []
    for marker in CREDENTIAL_MARKERS:
        if marker in lowered:
            found.append(_finding("critical", "credential_access", rel, line,
                                  f"refers to '{marker}' (credentials or browser data): {text[:120]}"))
            break
    for marker in PERSISTENCE_MARKERS:
        if marker in lowered:
            found.append(_finding("critical", "persistence", rel, line,
                                  f"refers to '{marker}' (runs itself again later): {text[:120]}"))
            break
    for marker in EXFIL_MARKERS:
        if marker in lowered:
            found.append(_finding("critical", "exfiltration_channel", rel, line,
                                  f"refers to '{marker}' (a channel malware uses to send data out)"))
            break
    if DOWNLOAD_AND_RUN.search(text):
        found.append(_finding("high", "download_and_run", rel, line, text[:160]))
    if RAW_IP_URL.search(text):
        found.append(_finding("medium", "raw_ip_url", rel, line, RAW_IP_URL.search(text).group(0)))
    if len(text) >= ENCODED_PAYLOAD_CHARS and ENCODED_PAYLOAD.fullmatch(text):
        found.append(_finding("high", "encoded_payload", rel, line,
                              f"a {len(text)}-character encoded string: {text[:60]}..."))
    return found


def scan_python(source: str, rel: str) -> list:
    """Malware patterns in one Python source (a .py file or notebook code)."""
    try:
        with warnings.catch_warnings():  # the repository's own invalid escapes etc. are not findings
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Python 2 code, or not Python: the text is still checked.
        return [f for i, line in enumerate(source.splitlines(), 1) for f in _string_findings(line, rel, i)]

    findings, calls = [], set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] in NATIVE_MODULES:
                    findings.append(_finding("high", "native_access", rel, line,
                                             f"imports {name} (direct access to the operating system)"))
                if name.split(".")[0] == "marshal":
                    findings.append(_finding("high", "marshal", rel, line, "imports marshal (runs hidden bytecode)"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            text = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
            findings += _string_findings(text, rel, line)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            calls.add(name)
            inner = {_call_name(n.func).rsplit(".", 1)[-1] for arg in node.args for n in ast.walk(arg)
                     if isinstance(n, ast.Call)}
            if name in ("exec", "eval", "compile"):
                if inner & DECODERS:
                    findings.append(_finding("critical", "obfuscated_exec", rel, line,
                                             f"{name}() of decoded data ({', '.join(sorted(inner & DECODERS))})"))
                elif not (node.args and isinstance(node.args[0], ast.Constant)):
                    findings.append(_finding("medium", "dynamic_exec", rel, line, f"{name}() of computed code"))
            elif name in ("__import__", "importlib.import_module") and not (
                    node.args and isinstance(node.args[0], ast.Constant)):
                findings.append(_finding("medium", "dynamic_exec", rel, line, f"{name}() of a computed module name"))
            elif name in SHELL_CALLS:
                command = " ".join(_strings_in(node))
                if DOWNLOAD_AND_RUN.search(command):
                    findings.append(_finding("high", "download_and_run", rel, line, f"{name}: {command}"))
                else:
                    findings.append(_finding("medium", "shell_command", rel, line, f"{name}: {command or '(computed)'}"))
            elif name in NETWORK_SEND_CALLS:
                sends_environment = any(isinstance(n, ast.Attribute) and n.attr == "environ" or
                                        isinstance(n, ast.Call) and _call_name(n.func).endswith("getenv")
                                        for arg in list(node.args) + [k.value for k in node.keywords]
                                        for n in ast.walk(arg))
                if sends_environment:
                    findings.append(_finding("critical", "exfiltration", rel, line,
                                             f"{name}() sends environment variables (where secrets are kept)"))
                else:
                    findings.append(_finding("medium", "network_send", rel, line, f"{name}()"))
            elif name in UNSAFE_LOADS:
                findings.append(_finding("medium", "unsafe_deserialization", rel, line,
                                         f"{name}() can run code hidden in the file it loads"))
            elif name in ("marshal.loads", "pty.spawn", "os.dup2"):
                findings.append(_finding("high", "reverse_shell" if name != "marshal.loads" else "marshal",
                                         rel, line, f"{name}()"))
    if {"socket.socket", "socket.create_connection"} & calls and {"os.dup2", "pty.spawn"} & calls:
        findings.append(_finding("critical", "reverse_shell", rel, 0,
                                 "opens a network socket and connects a shell to it"))
    return findings


def scan_tree(root: Path) -> tuple:
    """(findings, stats) for every file under root (a clone; .git excluded)."""
    root = Path(root)
    findings, files, total_bytes, python_files = [], 0, 0, 0
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if ".git" in rel_parts or not path.is_file():
            continue
        rel = "/".join(rel_parts)
        files += 1
        size = path.stat().st_size
        total_bytes += size
        suffix = path.suffix.lower()
        with path.open("rb") as handle:
            head = handle.read(4)
        magic = next((what for magic, what in MAGIC_BYTES.items() if head.startswith(magic)), "")
        if suffix in EXECUTABLE_EXTENSIONS:
            findings.append(_finding("critical", "executable_binary", rel, 0, f"an executable file ({suffix})"))
            continue
        if magic:
            severity = "high" if suffix in NATIVE_EXTENSIONS else "critical"
            findings.append(_finding(severity, "executable_binary", rel, 0,
                                     f"{magic}" + ("" if suffix in NATIVE_EXTENSIONS else f" named '{path.name}'")))
            continue
        if suffix in NATIVE_EXTENSIONS:
            findings.append(_finding("high", "executable_binary", rel, 0, f"a native library ({suffix})"))
            continue
        if suffix == ".pth" or path.name in AUTO_RUN_FILES:
            findings.append(_finding("high", "auto_run_file", rel, 0,
                                     f"{path.name} is run by Python itself when it starts"))
        if suffix in (".py", ".ipynb"):
            python_files += 1
            # A notebook's size is mostly its saved outputs (images); only its code counts.
            code = path.read_text(encoding="utf-8", errors="ignore") if suffix == ".py" \
                else repo_dependencies.notebook_code(path)
            if len(code) > MAX_SCANNED_FILE_BYTES:
                findings.append(_finding("high", "unreviewable_file", rel, 0,
                                         f"{len(code)} characters of code is too large to review"))
                continue
            file_findings = scan_python(code, rel)
            if suffix == ".ipynb":
                for line in repo_dependencies.notebook_shell_lines(path):
                    if DOWNLOAD_AND_RUN.search(line):
                        file_findings.append(_finding("high", "download_and_run", rel, 0, line))
                    elif not re.match(r"[!%]\s*(pip|conda)\s+install|%(matplotlib|load_ext|autoreload|time)",
                                      line):
                        file_findings.append(_finding("medium", "shell_command", rel, 0, line))
            if path.name == "setup.py":
                # pip runs setup.py while it installs, before anything is reviewed at run time.
                file_findings = [dict(f, severity="high", rule="setup_code:" + f["rule"])
                                 if f["severity"] == "medium" else f for f in file_findings]
            findings += file_findings
        elif suffix in SCRIPT_EXTENSIONS and size <= MAX_SCANNED_FILE_BYTES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            findings += [f for i, line in enumerate(text.splitlines(), 1) for f in _string_findings(line, rel, i)]
    if (root / ".gitmodules").is_file():
        findings.append(_finding("medium", "submodule", ".gitmodules", 0,
                                 "declares git submodules, which are not downloaded or reviewed"))
    return findings, {"files": files, "bytes": total_bytes, "python_files": python_files}


def _osv_query(packages: list) -> list:
    """OSV.dev batch query: one result per package ({"vulns": [{"id"}...]})."""
    queries = [{"package": {"name": p["name"], "ecosystem": "PyPI"}, **({"version": p["version"]}
                                                                         if p["version"] else {})}
               for p in packages]
    resp = requests.post(OSV_URL, json={"queries": queries}, timeout=OSV_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json().get("results") or []


def check_dependencies(root: Path) -> tuple:
    """(findings, notes) for the dependencies the repository declares."""
    root = Path(root)
    findings, notes, packages = [], [], []
    for req in repo_dependencies.declared_requirements(root):
        if req["kind"] == "option":
            option = req["line"].split()[0].lower()
            if option.startswith(("-i", "--index-url", "--extra-index-url", "--trusted-host", "-f", "--find-links")):
                findings.append(_finding("high", "dependency_source", "requirements.txt", 0,
                                         f"installs from outside PyPI: {req['line']}"))
            elif option.startswith(("-r", "-c", "--requirement", "--constraint")):
                findings.append(_finding("medium", "dependency_include", "requirements.txt", 0,
                                         f"includes another file that is not reviewed: {req['line']}"))
            continue
        if req["kind"] == "url":
            findings.append(_finding("high", "dependency_source", "requirements.txt", 0,
                                     f"installs from a URL, repository or local path: {req['line']}"))
            continue
        name = req["name"]
        close = typosquat_of(name)
        if close:
            findings.append(_finding("high", "typosquat", "requirements.txt", 0,
                                     f"'{name}' is one letter from the popular package '{close}'"))
        packages.append({"name": name, "version": repo_dependencies.pinned_version(req["spec"])})

    if not packages:
        return findings, notes
    try:
        results = _osv_query(packages)
    except Exception as e:
        notes.append(f"OSV.dev could not be reached ({type(e).__name__}); known vulnerabilities were not checked.")
        findings.append(_finding("low", "osv_unavailable", "requirements.txt", 0, notes[-1]))
        return findings, notes
    for package, result in zip(packages, results):
        ids = [v.get("id", "") for v in (result or {}).get("vulns") or []]
        malicious = [i for i in ids if i.startswith("MAL-")]
        label = f"{package['name']}{'==' + package['version'] if package['version'] else ''}"
        if malicious:
            findings.append(_finding("critical", "malicious_package", "requirements.txt", 0,
                                     f"{label} is a known malicious package ({', '.join(malicious[:3])})"))
        elif ids and package["version"]:
            # Without a pinned version, OSV lists every advisory the package
            # ever had, most of them fixed long ago.
            findings.append(_finding("medium", "vulnerable_package", "requirements.txt", 0,
                                     f"{label} has {len(ids)} known vulnerabilit{'y' if len(ids) == 1 else 'ies'} "
                                     f"({', '.join(ids[:5])})"))
    notes.append(f"Checked {len(packages)} declared package(s) against OSV.dev.")
    return findings, notes


# config.py loads .env into os.environ, so every child process would get the
# project's API keys. Code from the internet (and pip, setup.py, a generated
# script) must not: these variables are left out of a child's environment.
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|DATABASE_URL|_DSN$|_AUTH|COOKIE|SESSION",
                         re.IGNORECASE)


def _project_variable_names() -> set:
    """Every variable .env.example lists (the project's own settings)."""
    try:
        lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {line.split("=", 1)[0].strip() for line in lines if "=" in line and not line.lstrip().startswith("#")}


PROJECT_VARIABLES = _project_variable_names()


def child_environment(extra: dict = None) -> dict:
    """os.environ without secrets or the project's own settings, plus extra."""
    return {**{k: v for k, v in os.environ.items() if not SECRET_NAME.search(k) and k not in PROJECT_VARIABLES},
            **(extra or {})}


def _git(args: list, cwd: Path = None, timeout: int = CLONE_TIMEOUT_SECONDS) -> dict:
    env = child_environment({"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace", env=env)
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": f"git {args[0]} timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": None, "stdout": "", "stderr": str(e)}


def _remove_tree(path: Path) -> None:
    # git marks its object files read-only; Windows then refuses a plain rmtree.
    import shutil
    import stat

    def _make_writable_and_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=_make_writable_and_retry)


def _slug(url: str) -> str:
    """Short folder name: a deep repository file under a long folder name
    passed Windows' path limit even with core.longpaths (a live clone)."""
    import hashlib

    name = re.sub(r"[^a-z0-9]+", "-", (url or "").rstrip("/").lower().rsplit("/", 1)[-1]).strip("-")[:24]
    return f"{name or 'repository'}-{hashlib.sha1((url or '').encode()).hexdigest()[:8]}"


def clone_for_review(repo_url: str, dest: Path) -> dict:
    """{"ok", "error", "commit_sha", "symlinks"}. Only https URLs: other git
    transports (ext::, file://, ssh) can run commands or read local files."""
    if not re.match(r"^https://[A-Za-z0-9.-]+/[^\s]+$", repo_url or ""):
        return {"ok": False, "error": f"not an https repository URL: {repo_url!r}", "commit_sha": "", "symlinks": []}
    if dest.exists():
        _remove_tree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # core.longpaths: Windows otherwise fails on paths over 260 characters
    # (a live clone of a book's notebooks did).
    clone = _git(["-c", "core.symlinks=false", "-c", "core.longpaths=true", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
                  "clone", "--depth", "1", "--no-recurse-submodules", "--quiet", repo_url, str(dest)])
    if clone["exit_code"] != 0:
        return {"ok": False, "error": (clone["stderr"] or clone["stdout"]).strip()[:500], "commit_sha": "",
                "symlinks": []}
    sha = _git(["rev-parse", "HEAD"], cwd=dest, timeout=30)
    files = _git(["ls-files", "-s"], cwd=dest, timeout=30)
    symlinks = [line.split("\t", 1)[-1] for line in files["stdout"].splitlines() if line.startswith("120000 ")]
    return {"ok": True, "error": "", "commit_sha": sha["stdout"].strip() if sha["exit_code"] == 0 else "",
            "symlinks": symlinks}


def _verdict(findings: list) -> str:
    severities = {f["severity"] for f in findings}
    if severities & set(BLOCKING_SEVERITIES):
        return "REJECTED"
    return "APPROVED_WITH_WARNINGS" if "medium" in severities else "APPROVED"


def review_repository(repository: dict) -> dict:
    """Clone (nothing executed) and review one repository. Never raises.

    Returns {"verdict", "repo_url", "commit_sha", "clone_dir", "findings",
    "counts", "stats", "notes", "summary"}."""
    repo_url = (repository or {}).get("repo_url", "")
    dest = REVIEW_DIR / _slug(repo_url)
    review = {"verdict": "REJECTED", "repo_url": repo_url, "commit_sha": "", "clone_dir": "", "findings": [],
              "counts": {}, "stats": {}, "notes": [], "summary": ""}
    try:
        findings = []
        if (repository or {}).get("provenance") not in PROVENANCE_KINDS:
            findings.append(_finding("critical", "no_provenance", "", 0,
                                     "Code Discovery recorded no evidence that this is the Parent Paper's own code"))
        clone = clone_for_review(repo_url, dest)
        if not clone["ok"]:
            findings.append(_finding("critical", "clone_failed", "", 0, clone["error"]))
        else:
            review.update(commit_sha=clone["commit_sha"], clone_dir=str(dest))
            findings += [_finding("high", "symlink", link, 0, "a symbolic link (can point outside the repository)")
                         for link in clone["symlinks"]]
            tree_findings, stats = scan_tree(dest)
            findings += tree_findings
            review["stats"] = stats
            if not stats["python_files"]:
                findings.append(_finding("critical", "no_python_code", "", 0, "no .py or .ipynb file"))
            if stats["files"] > MAX_REPO_FILES or stats["bytes"] > MAX_REPO_BYTES:
                findings.append(_finding("critical", "too_large", "", 0,
                                         f"{stats['files']} files, {stats['bytes']} bytes (limit {MAX_REPO_FILES} "
                                         f"files, {MAX_REPO_BYTES} bytes)"))
            dependency_findings, notes = check_dependencies(dest)
            findings += dependency_findings
            review["notes"] += notes
            if not (repository or {}).get("license"):
                findings.append(_finding("low", "no_license", "", 0,
                                         "no license detected: reuse and redistribution rights are unclear"))
    except Exception as e:  # a review that cannot finish is a rejection, never an approval
        findings.append(_finding("critical", "review_error", "", 0, f"{type(e).__name__}: {e}"))

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    review["counts"] = {s: sum(1 for f in findings if f["severity"] == s) for s in order}
    review["findings"] = findings[:MAX_FINDINGS]
    review["verdict"] = _verdict(findings)
    blocking = [f for f in findings if f["severity"] in BLOCKING_SEVERITIES]
    review["summary"] = (
        f"{review['verdict']}: {review['counts']['critical']} critical, {review['counts']['high']} high, "
        f"{review['counts']['medium']} medium, {review['counts']['low']} low finding(s)."
        + (" Blocking: " + "; ".join(f"{f['rule']} in {f['path'] or 'repository'}: {f['evidence']}"
                                      for f in blocking[:3]) if blocking else ""))
    if review["verdict"] == "REJECTED" and review["clone_dir"]:
        # A rejected clone is not kept: nothing may run it later by mistake.
        _remove_tree(dest)
        review["clone_dir"] = ""
    return review


def format_review(review: dict) -> str:
    """Human-readable report of one review."""
    lines = ["[Master Agent] Repository review (static: nothing in the repository was run)",
             f"Repository: {review.get('repo_url', '')} (commit {review.get('commit_sha') or 'unknown'})",
             f"Verdict:    {review.get('verdict', '')}",
             f"Summary:    {review.get('summary', '')}"]
    stats = review.get("stats") or {}
    if stats:
        lines.append(f"Scanned:    {stats.get('files', 0)} files, {stats.get('python_files', 0)} with Python code, "
                     f"{stats.get('bytes', 0)} bytes")
    lines += [f"Note: {n}" for n in review.get("notes") or []]
    if review.get("findings"):
        lines.append("Findings:")
        lines += [f"  - [{f['severity']}] {f['rule']} {f['path']}{':' + str(f['line']) if f['line'] else ''} "
                  f"-- {f['evidence']}" for f in review["findings"]]
    lines.append("A static review finds known malware patterns; it cannot prove the code is safe. Approved code "
                 "runs in an isolated venv with this user's permissions, not in a sandbox.")
    return "\n".join(lines)
