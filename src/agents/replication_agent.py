"""
Replication Agent (Milestone 10).

Establishes the baseline reproduction of the Parent Paper, branching
entirely on Code Discovery Agent's result:

  1. FOUND -- clone the discovered repository, set up an isolated
     environment, and run it against the Cleaning Agent's dataset.
  2. NOT_FOUND -- reconstruct a baseline implementation from the paper
     itself, since CLAUDE.md's Preferred Implementation Order allows this
     as the last resort. The agent reads the Parent Paper directly: its
     open-access full text when one exists (src/search/paper_fulltext.py;
     the methodology, preprocessing, split and hyperparameter passages are
     given to the generation call), else its stored abstract and metadata.
     Every inferred detail is documented; the result is never presented as
     the authors' own code.

A reconstructed baseline can be REVISED when Master Agent relays an
Evaluation Agent finding (e.g. SMOTE applied before the split, so the test
set leaked into training): the previous script and its output are archived
under replication/<slug>/revisions/round_<n>/, and one LLM call fixes only
the stated defects. A cloned repository is never revised -- the authors'
code is the baseline; a defect in it is left for the Experiment Agent.

This agent clones and executes code from the internet (or LLM-generated
code), which is a different risk category than every other agent in this
project (which only ever did read-only search, LLM calls, and file I/O on
data the project itself controls). Containment, not a human approval
gate, is the safeguard (per CLAUDE.md's Autonomous Operation policy):

  - A cloned repository is installed and run in a Docker sandbox (below);
    a reconstructed script in a fresh virtualenv under
    replication/<slug>/venv/ -- the project's own .venv is never touched.
  - Dependencies are only ever installed from a known source: the repo's
    own committed file and known packages its code imports (Path A), or a
    fixed allowlist of common
    ML libraries (Path B, since that list comes from a generative call,
    not a repo's own committed file -- a meaningfully different trust
    level).
  - Every subprocess call uses an argv list (never shell=True) and a
    bounded timeout.
  - No setup step is ever auto-elevated (sudo/admin); one that needs it
    is reported as a failure, not bypassed.

A cloned repository's code runs only in a Docker sandbox
(src/orchestration/sandbox.py): no network while it runs, no host secrets,
unprivileged, resource-limited, with only the repository copy and the
prepared CSVs mounted. Without Docker it is not run, and the baseline is
reconstructed instead. Master Agent reviews the repository first
(repo_review), and only that reviewed copy is used.

Comparing reproduced metrics against the paper's published results is
the Evaluation Agent's job, not this one's -- this agent's job ends at
"did the baseline execute and produce output."
"""
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src import llm
from src.agents.evaluation_agent import METRIC_LINE_FORMAT, parse_metric_lines
from src.search import paper_fulltext
from src.orchestration import repo_dependencies, repo_review, sandbox
from src.orchestration.handoff import request as handoff_request

REPO_ROOT = Path(__file__).parent.parent.parent
REPLICATION_DIR = REPO_ROOT / "replication"

VENV_TIMEOUT_SECONDS = 120
INSTALL_TIMEOUT_SECONDS = 600
REPLICATION_TIMEOUT_SECONDS = 1800
README_CHAR_LIMIT = 6000
# Free-model scripts often need more than one fix (an import, then a
# column name, then a dtype), so a reconstruction gets 3 repairs.
MAX_REPAIR_ATTEMPTS = 4
GENERATION_TIMEOUT_SECONDS = 600
PROFILE_SAMPLE_ROWS = 2000
PROFILE_MAX_COLUMNS = 60
STDERR_FEEDBACK_CHARS = 3000
PAPER_EXCERPT_CHARS = 12000

# A cloned repository that fails gets ONE repair (missing packages, then one
# checked LLM-proposed fix from a fixed menu) and one more run; after that
# the baseline is reconstructed from the paper (run()).
MAX_FIX_ACTIONS = 5
MAX_MISSING_MODULE_INSTALLS = 5
DATA_SPLITS = ("train", "test", "validation", "master")
SHELL_CHARACTERS = re.compile(r"[;&|<>`$\n\r]")
ALLOWED_MODULES = ("jupyter", "nbconvert")
DEPENDENCY_FILENAMES = ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml", "Pipfile")

ENTRY_POINT_CANDIDATES = ("train.py", "main.py", "run.py", "run_experiment.py")
README_FILENAMES = ("README.md", "README.rst", "README.txt", "readme.md")

# Path B's packages come from a generative LLM call, not a repo's own
# committed dependency file -- a meaningfully different trust level, so
# anything outside this fixed set is dropped rather than installed.
DEPENDENCY_ALLOWLIST = {
    "scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "catboost", "torch", "tensorflow", "matplotlib",
    "imbalanced-learn",
}
# Every reconstructed script reads CSVs and computes metrics, so these are
# installed whether or not the generated script lists them.
BASE_LIBRARIES = ("numpy", "pandas", "scikit-learn")

# A reconstructed script reads its data paths from these environment
# variables rather than from hardcoded literals, so the Experiment Agent
# can point the same script at a re-split dataset deterministically. (A
# hardcoded Windows path can come back escaped, raw, or with forward
# slashes, and a text substitution that silently misses would run the
# "experiment" on the baseline's own split.)
DATA_ENV_VARS = {
    "TRAIN_CSV": "train_csv_path",
    "TEST_CSV": "test_csv_path",
    "VALIDATION_CSV": "validation_csv_path",
}

_RESULT_DEFAULTS = {
    "status": "",                 # "OK" | "BLOCKED" | "TIMEOUT"
    "source": "",                 # "cloned_repository" | "reconstructed"
    "repo_url": "",
    "commit_sha": "",
    "replication_dir": "",
    "entry_point": "",
    "entry_point_source": "",      # "convention" | "readme_llm" | "generated"
    "command": None,
    "exit_code": None,
    "stdout_path": "",
    "stderr_path": "",
    "duration_seconds": None,
    "pip_freeze_path": "",
    "inferred_assumptions": [],
    "generated_by": [],            # model(s) that wrote/repaired a reconstructed script, in order
    "modifications": [],
    "paper_text_status": "",       # "FULL_TEXT" | "ABSTRACT_ONLY" -- what the agent could read
    "revision_round": 0,           # 0 = first baseline; n = revised on an Evaluation Agent request
    "revision_instructions": [],
    "fast_mode_from": None,        # memory id of a past script adapted in fast mode (src/memory/recall.py)
    "repository_review": None,     # Master Agent's review of a cloned repository: {"verdict", "commit_sha", "summary"}
    "repository_failed": False,    # a cloned repository did not run, even after one repair
    "sandbox": None,               # the Docker sandbox a cloned repository ran in (sandbox.Sandbox.describe)
    "repository_fallback": None,   # {"repo_url", "commit_sha", "reason", "warnings"} when it did not, and the baseline was reconstructed
    "warnings": [],
    "notes": "",
}


def _default_result(**overrides) -> dict:
    result = dict(_RESULT_DEFAULTS)
    result.update(overrides)
    return result


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60]
    return slug or "replication"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_pip(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")


def data_env(split: dict) -> dict:
    """Environment variables pointing a reconstructed script at a split's
    CSVs -- see DATA_ENV_VARS."""
    return {var: str((split or {}).get(key, "")) for var, key in DATA_ENV_VARS.items() if (split or {}).get(key)}


def _run_bounded(cmd: list, cwd: Path = None, timeout: int = REPLICATION_TIMEOUT_SECONDS, env: dict = None) -> dict:
    """Run cmd (always an argv list, never a shell string) with a bounded
    timeout, capturing output. Never raises -- every outcome (success,
    non-zero exit, timeout, or a launch failure) comes back as the same
    dict shape so callers don't need a try/except of their own.

    The child runs in UTF-8 mode: on Windows a piped child otherwise
    encodes stdout as cp1252 and crashes on the first non-ASCII character
    a generated script prints. It never gets the project's secrets
    (repo_review.child_environment): it may be code from the internet, pip
    running a setup.py, or an LLM-written script."""
    start = time.monotonic()
    child_env = repo_review.child_environment({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})})
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", env=child_env,
        )
        return {
            "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "duration_seconds": time.monotonic() - start, "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": None, "stdout": _as_text(e.stdout), "stderr": _as_text(e.stderr),
            "duration_seconds": time.monotonic() - start, "timed_out": True,
        }
    except Exception as e:
        return {
            "exit_code": None, "stdout": "", "stderr": str(e),
            "duration_seconds": time.monotonic() - start, "timed_out": False,
        }


def _as_text(output) -> str:
    # TimeoutExpired carries bytes even when text mode was requested.
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


def _remove_tree(path: Path) -> None:
    """shutil.rmtree that also deletes read-only files. git marks the files
    under .git/objects/ read-only, and on Windows a plain rmtree then fails
    with "Access is denied" (a live run lost 4 of its 5 attempts to this)."""
    def _make_writable_and_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=_make_writable_and_retry)


def _create_venv(venv_dir: Path, python: str = None) -> dict:
    return _run_bounded([python or sys.executable, "-m", "venv", str(venv_dir)], timeout=VENV_TIMEOUT_SECONDS)


def _pip_install(venv_dir: Path, cwd: Path, install_args: list) -> dict:
    return _run_bounded([str(_venv_pip(venv_dir)), *install_args], cwd=cwd, timeout=INSTALL_TIMEOUT_SECONDS)


def _pip_freeze(venv_dir: Path) -> str:
    result = _run_bounded([str(_venv_pip(venv_dir)), "freeze"], timeout=30)
    return result["stdout"] if result["exit_code"] == 0 else ""


def _select_dependency_install(dependency_files: list) -> tuple:
    """Returns (pip install argv tail, blocked_reason). Only ever
    installs from the repo's own declared file -- never guesses."""
    files = set(dependency_files or [])
    if "requirements.txt" in files:
        return ["install", "-r", "requirements.txt"], ""
    if "pyproject.toml" in files or "setup.py" in files:
        return ["install", "."], ""
    if "environment.yml" in files or "Pipfile" in files:
        return None, "Repo only declares environment.yml/Pipfile; this project has no conda/pipenv tooling for automated installs."
    return None, "No recognized dependency file found; skipping install."


def _detect_entry_point_by_convention(repo_dir: Path) -> str:
    for name in ENTRY_POINT_CANDIDATES:
        if (repo_dir / name).is_file():
            return name
    return ""


def _read_readme(repo_dir: Path) -> str:
    for name in README_FILENAMES:
        path = repo_dir / name
        if path.is_file():
            return path.read_text(errors="ignore")[:README_CHAR_LIMIT]
    return ""


def _extract_entry_point_from_readme(repo_dir: Path, paper_title: str) -> list:
    """One LLM call restricted to finding the documented run command in
    the repo's own README -- same no-guessing discipline as every other
    agent's extraction calls. Returns [] if nothing is documented."""
    readme_text = _read_readme(repo_dir)
    if not readme_text:
        return []

    prompt = f"""You are finding the documented command to run a code repository that implements an academic paper.

Paper title: "{paper_title}"

README contents:
{readme_text}

Find the command the README documents for running training/the main experiment (not installation, not tests). Return ONLY a JSON object:
{{"command": ["python", "script.py", "--flag", "value"]}}
or, if no run command is documented:
{{"command": null}}
Return ONLY valid JSON. No explanation or markdown."""

    try:
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, agent="replication_agent")
    except Exception:
        return []

    data = llm.parse_json(content, default={})
    command = data.get("command") if isinstance(data, dict) else None
    return command if isinstance(command, list) and command else []


def _wire_dataset_into_repo(repo_dir: Path, cleaning_result: dict) -> list:
    """Copy the Cleaning Agent's CSVs into the repo's own conventional
    data/ directory if one exists, rather than editing the repo's code --
    minimal modification, per CLAUDE.md. If no such directory exists,
    this is a no-op and the run is attempted unmodified."""
    modifications = []
    data_dir = repo_dir / "data"
    if not data_dir.is_dir():
        return modifications

    for key, filename in (
        ("master_csv_path", "dataset.csv"),
        ("train_csv_path", "train.csv"),
        ("test_csv_path", "test.csv"),
        ("validation_csv_path", "validation.csv"),
    ):
        src = (cleaning_result or {}).get(key)
        if src and Path(src).is_file():
            dest = data_dir / filename
            shutil.copy2(src, dest)
            modifications.append(f"Copied {src} -> {dest} (repo's existing data/ directory convention).")
    return modifications


def _save_run_output(slug_dir: Path, run_result: dict) -> tuple:
    stdout_path = slug_dir / "run_stdout.txt"
    stderr_path = slug_dir / "run_stderr.txt"
    stdout_path.write_text(run_result.get("stdout", "") or "", encoding="utf-8")
    stderr_path.write_text(run_result.get("stderr", "") or "", encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def _write_environment_doc(slug_dir: Path, details: dict) -> None:
    lines = [
        "# Replication Environment",
        f"Pipeline Python (this machine): {sys.version}",
        f"OS: {platform.platform()}",
    ]
    for key, value in details.items():
        lines.append(f"{key}: {value}")
    (slug_dir / "ENVIRONMENT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_notes_doc(recon_dir: Path, inferred_assumptions: list) -> None:
    lines = [
        "# Reconstruction Notes",
        "",
        "This code was generated from the paper's own stored text, NOT the authors' "
        "original implementation. Every detail not explicitly stated by the paper:",
    ]
    lines.extend(f"- {a}" for a in inferred_assumptions)
    (recon_dir / "RECONSTRUCTION_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def _get_repository_code(repository: dict, dest: Path) -> dict:
    """A copy of the clone Master Agent reviewed (repo_review), never a new
    download: the files that run are the files that were checked. Returns
    {"ok", "error", "commit_sha"}."""
    review = (repository or {}).get("review") or {}
    if review.get("verdict") not in repo_review.APPROVED_VERDICTS:
        return {"ok": False, "commit_sha": "",
                "error": f"Master Agent has not approved this repository ({review.get('verdict') or 'not reviewed'}"
                         + (f": {review['summary']}" if review.get("summary") else "") + ")."}
    source = Path(review.get("clone_dir") or "")
    if not review.get("clone_dir") or not source.is_dir():
        return {"ok": False, "commit_sha": "", "error": f"The reviewed clone {source} is missing."}
    shutil.copytree(source, dest, symlinks=True)
    sha = _run_bounded(["git", "rev-parse", "HEAD"], cwd=dest, timeout=30)
    commit_sha = sha["stdout"].strip() if sha["exit_code"] == 0 else ""
    if not commit_sha or commit_sha != review.get("commit_sha"):
        return {"ok": False, "commit_sha": commit_sha,
                "error": f"The copy is at commit {commit_sha or 'unknown'}, not the reviewed {review.get('commit_sha')}."}
    return {"ok": True, "error": "", "commit_sha": commit_sha}


def _declared_dependency_files(repo_dir: Path) -> list:
    """The dependency files in the copy itself (Code Discovery's GitHub
    listing can be out of date)."""
    return [name for name in DEPENDENCY_FILENAMES if (repo_dir / name).is_file()]


def _install_relaxed_requirements(box, repo_dir: Path) -> tuple:
    """Install requirements.txt one line at a time; a pinned line that fails
    is installed without its pin. Returns (ok, modifications, failed lines)."""
    modifications, failed = [], []
    for req in repo_dependencies.parse_requirements(repo_dir / "requirements.txt"):
        if req["kind"] != "pypi":
            failed.append(req["line"])
            continue
        if box.pip(["install", req["line"]], timeout=INSTALL_TIMEOUT_SECONDS)["exit_code"] == 0:
            continue
        if req["spec"] and box.pip(["install", req["name"]], timeout=INSTALL_TIMEOUT_SECONDS)["exit_code"] == 0:
            modifications.append(f"Installed {req['name']} without the pin '{req['line']}': the pinned version "
                                 "does not install in this environment.")
            continue
        failed.append(req["line"])
    return not failed, modifications, failed


def _setup_repository_environment(repo_dir: Path, box) -> dict:
    """A new venv in the Docker sandbox (src/orchestration/sandbox.py) with
    everything the repository needs: its declared dependencies, then the
    packages its code imports but does not declare.

    Declared dependencies are tried in this order, stopping at the first
    that installs:
      1. pre-built wheels only (--only-binary=:all:) on each sandbox Python
         (sandbox.SANDBOX_PYTHON_VERSIONS) -- a pin with no wheel fails in
         seconds, not after a long source build (a live run spent minutes
         compiling pandas 2.2.3 for Python 3.14);
      2. a normal install on the first Python (source builds allowed);
      3. requirements.txt one line at a time, a pin that cannot install
         relaxed (recorded, not hidden).
    A project install (pip install .) has no wheel-only step.
    Returns {"ok", "error", "python", "install", "modifications", "warnings"}."""
    env = {"ok": False, "error": "", "python": "", "install": "", "modifications": [], "warnings": []}
    versions = list(sandbox.SANDBOX_PYTHON_VERSIONS)
    install_args, not_installed = _select_dependency_install(_declared_dependency_files(repo_dir))
    if install_args is None:
        env["warnings"].append(f"{not_installed} The packages the code imports are installed instead.")
        made = box.use_python(versions[0])
        if made["exit_code"] != 0:
            env["error"] = f"The sandbox environment could not be made: {(made['stderr'] or '')[-1500:]}"
            return env
    else:
        env["install"] = " ".join(install_args)
        requirements = install_args[:2] == ["install", "-r"]
        if requirements:
            attempts = [(v, ["install", "--only-binary=:all:", *install_args[1:]]) for v in versions] \
                + [(versions[0], install_args)]
        else:
            attempts = [(v, install_args) for v in versions]
        install_result = None
        for i, (version, args) in enumerate(attempts):
            made = box.use_python(version)
            if made["exit_code"] != 0:
                install_result = made
                env["warnings"].append(f"The Python {version} sandbox could not be made: "
                                       f"{(made['stderr'] or '').strip()[-300:]}")
                continue
            install_result = box.pip(args, timeout=INSTALL_TIMEOUT_SECONDS)
            if install_result["exit_code"] == 0:
                if version != versions[0]:
                    env["modifications"].append(f"Made the environment with Python {version}: the repository's "
                                                f"dependencies do not install on Python {versions[0]}.")
                break
            if i + 1 < len(attempts):
                following = attempts[i + 1]
                how = "with source builds allowed" if "--only-binary=:all:" not in following[1] else "from wheels"
                env["warnings"].append(f"Dependency installation failed on Python {version}; trying Python "
                                       f"{following[0]} {how}.")
                print(f"[Replication Agent] {env['warnings'][-1]}")
        stderr = (install_result or {}).get("stderr") or ""
        if (install_result or {}).get("exit_code") != 0 and requirements and box.volume:
            env["warnings"].append("Dependency installation still failed; installing the requirements one at a "
                                   "time and relaxing pins that cannot install.")
            print(f"[Replication Agent] {env['warnings'][-1]}")
            ok, relaxed, failed = _install_relaxed_requirements(box, repo_dir)
            env["modifications"] += relaxed
            if not ok:
                env["error"] = f"Dependency installation failed for: {', '.join(failed)}. {stderr[-1500:]}"
                return env
        elif (install_result or {}).get("exit_code") != 0:
            env["error"] = f"Dependency installation failed: {stderr[-2000:] or 'no environment could be made'}"
            return env

    declared = {r["name"] for r in repo_dependencies.declared_requirements(repo_dir) if r["name"]}
    imported, unknown = repo_dependencies.imported_packages(repo_dir)
    for package in [p for p in imported if repo_dependencies.normalize(p) not in declared]:
        if box.pip(["install", package], timeout=INSTALL_TIMEOUT_SECONDS)["exit_code"] == 0:
            env["modifications"].append(f"Installed {package}: the code imports it, but the repository does not "
                                        "declare it.")
        else:
            env["warnings"].append(f"Could not install {package}, which the code imports.")
    if unknown:
        env["warnings"].append(f"Imports with no known package, not installed: {', '.join(unknown[:20])}.")
    env["python"] = f"{box.version} ({box.digest or box.image})"
    env["ok"] = True
    return env


def _run_installing_missing_modules(command: list, box, env: dict, modifications: list) -> dict:
    """Run the command in the sandbox; when it stops on a ModuleNotFoundError
    for a known package, install that package and run again (at most
    MAX_MISSING_MODULE_INSTALLS times)."""
    tried = set()
    while True:
        result = box.run(command, env=env, timeout=REPLICATION_TIMEOUT_SECONDS)
        if result["exit_code"] == 0 or result["timed_out"] or len(tried) >= MAX_MISSING_MODULE_INSTALLS:
            return result
        package = repo_dependencies.IMPORT_TO_PACKAGE.get(repo_dependencies.missing_module(result["stderr"]))
        if not package or package in tried:
            return result
        tried.add(package)
        print(f"[Replication Agent] The run needs {package}; installing it and running again.")
        if box.pip(["install", package], timeout=INSTALL_TIMEOUT_SECONDS)["exit_code"] != 0:
            return result
        modifications.append(f"Installed {package}: the run stopped on a ModuleNotFoundError for it.")


def _repo_file_list(repo_dir: Path, limit: int = 200) -> list:
    return sorted("/".join(p.relative_to(repo_dir).parts) for p in repo_dir.rglob("*")
                  if p.is_file() and ".git" not in p.relative_to(repo_dir).parts)[:limit]


def _propose_repository_fix(paper_title: str, repo_dir: Path, command: list, failure: str,
                            cleaning_result: dict) -> dict:
    """One LLM call: what change from a fixed menu would make the authors'
    code run? Returns {"diagnosis", "actions"}; every action is then
    checked by _check_fix_actions before anything is applied."""
    splits = {split: (cleaning_result or {}).get(f"{split}_csv_path") for split in DATA_SPLITS}
    prompt = f"""You are making a research paper's own code repository run, without changing its method.

Paper: "{paper_title}"
Command that was run: {command or "none -- no entry point was found"}
Failure (stderr/stdout tail):
{(failure or '')[-STDERR_FEEDBACK_CHARS:]}

Repository files:
{chr(10).join(_repo_file_list(repo_dir))}

README:
{_read_readme(repo_dir) or "(none)"}

The project's prepared data (CSV files you may copy into the repository): {splits}

Choose at most {MAX_FIX_ACTIONS} actions, only from these types:
  {{"type": "install", "packages": ["pypi-name", ...]}}  -- a missing third-party package
  {{"type": "command", "command": ["python", "path/in/repo.py", "--arg", "value"]}}  -- the right script/arguments
  {{"type": "copy_data", "split": "train|test|validation|master", "dest": "relative/path/in/repo.csv"}}  -- put the data where the code reads it
Never edit the repository's code, never change the model or its hyperparameters, never use a shell.
Return ONLY JSON: {{"diagnosis": "one sentence", "actions": [...]}} (actions [] if no listed action can fix it)."""
    try:
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, agent="replication_agent")
    except Exception as e:
        return {"diagnosis": f"The fix could not be proposed ({type(e).__name__}: {e}).", "actions": []}
    data = llm.parse_json(content, default={})
    data = data if isinstance(data, dict) else {}
    actions = data.get("actions") if isinstance(data.get("actions"), list) else []
    return {"diagnosis": str(data.get("diagnosis") or ""), "actions": actions[:MAX_FIX_ACTIONS]}


def _inside(repo_dir: Path, relative: str) -> Path | None:
    """repo_dir/relative when it stays inside the repository (and out of .git)."""
    if not relative or Path(relative).is_absolute() or Path(relative).drive:
        return None
    target = (repo_dir / relative).resolve()
    root = repo_dir.resolve()
    if target != root and root not in target.parents:
        return None
    if ".git" in target.relative_to(root).parts:
        return None
    return target


def _check_fix_actions(actions: list, repo_dir: Path, cleaning_result: dict) -> tuple:
    """(accepted, rejected reasons). Deterministic: a package must be a known
    one (never a guessed name), a command must run the venv's Python on a
    script inside the repository (or jupyter nbconvert) with no shell
    characters, and data may be copied only to a .csv inside it."""
    accepted, rejected = [], []
    data_paths = {str((cleaning_result or {}).get(f"{s}_csv_path") or "") for s in DATA_SPLITS} - {""}
    known = {repo_dependencies.normalize(k) for k in repo_dependencies.KNOWN_PACKAGES}
    for action in actions:
        kind = action.get("type") if isinstance(action, dict) else None
        if kind == "install":
            packages = [str(p) for p in action.get("packages") or []]
            unknown = [p for p in packages if repo_dependencies.normalize(p) not in known]
            if not packages or unknown:
                rejected.append(f"install {packages}: not a known package ({', '.join(unknown) or 'none given'})")
            else:
                accepted.append({"type": "install", "packages": packages})
        elif kind == "command":
            command = [str(a) for a in action.get("command") or []]
            ok = len(command) >= 2 and command[0] in ("python", "python3") and not any(
                SHELL_CHARACTERS.search(a) for a in command)
            if ok and command[1] == "-m":
                ok = len(command) >= 3 and command[2] in ALLOWED_MODULES
            elif ok:
                script = _inside(repo_dir, command[1])
                ok = bool(script and script.is_file() and script.suffix == ".py")
            ok = ok and all(not Path(a).is_absolute() or a in data_paths for a in command[1:])
            if ok:
                accepted.append({"type": "command", "command": command})
            else:
                rejected.append(f"command {command}: not the venv's Python on a script inside the repository")
        elif kind == "copy_data":
            split, dest = str(action.get("split") or ""), str(action.get("dest") or "")
            target = _inside(repo_dir, dest)
            source = (cleaning_result or {}).get(f"{split}_csv_path")
            if split in DATA_SPLITS and source and Path(source).is_file() and target and target.suffix == ".csv":
                accepted.append({"type": "copy_data", "split": split, "dest": dest})
            else:
                rejected.append(f"copy_data {split} -> {dest}: not a prepared split copied to a .csv in the repository")
        else:
            rejected.append(f"{action}: not an allowed action type")
    return accepted, rejected


def _apply_fix_actions(actions: list, repo_dir: Path, box, cleaning_result: dict, command: list) -> tuple:
    """Apply checked actions. Returns (command, modifications)."""
    modifications = []
    for action in actions:
        if action["type"] == "install":
            for package in action["packages"]:
                if box.pip(["install", package], timeout=INSTALL_TIMEOUT_SECONDS)["exit_code"] == 0:
                    modifications.append(f"Repair: installed {package}.")
        elif action["type"] == "command":
            command = ["python", *action["command"][1:]]
            modifications.append(f"Repair: ran {' '.join(action['command'])} instead.")
        elif action["type"] == "copy_data":
            target = _inside(repo_dir, action["dest"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cleaning_result[f"{action['split']}_csv_path"], target)
            modifications.append(f"Repair: copied the {action['split']} split to {action['dest']}.")
    return command, modifications


def _replicate_from_repository(parent_paper: dict, repository: dict, cleaning_result: dict) -> dict:
    """Run the authors' code, only in the Docker sandbox (sandbox.py): no
    network while it runs, no host secrets, no host folder but the
    repository copy. Without Docker the repository is not run at all.

    Any failure (code not approved, no sandbox, environment, no entry point,
    a failing run) gets one repair -- missing packages are installed, then
    one checked LLM-proposed fix -- and one more run. A repository that
    still does not run is marked repository_failed, and run() reconstructs
    the baseline from the paper instead."""
    paper_title = (parent_paper or {}).get("title", "")
    repo_url = repository.get("repo_url", "")
    slug_dir = REPLICATION_DIR / _slugify(paper_title or repo_url)
    repo_dir = slug_dir / "repo"
    warnings = []
    review = repository.get("review") or {}
    base = {"source": "cloned_repository", "repo_url": repo_url, "replication_dir": str(slug_dir),
            "repository_review": {k: review.get(k) for k in ("verdict", "commit_sha", "summary")} if review else None}

    def failed(notes: str, warning: str, **extra) -> dict:
        print(f"[Replication Agent] {notes} {warning}")
        return _default_result(status=extra.pop("status", "BLOCKED"), **base, warnings=warnings + [warning],
                               notes=notes, repository_failed=True, **extra)

    print(f"\n[Replication Agent] Preparing the reviewed copy of {repo_url} for baseline replication...")
    slug_dir.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        _remove_tree(repo_dir)

    code = _get_repository_code(repository, repo_dir)
    if not code["ok"]:
        return failed("The repository's code cannot be used.", code["error"], commit_sha=code["commit_sha"])
    commit = {"commit_sha": code["commit_sha"]}

    available, reason = sandbox.engine_available()
    if not available:
        return failed("The repository's code runs only in the Docker sandbox, and Docker is not available.",
                      reason, **commit)

    box = sandbox.Sandbox(repo_dir, {split: (cleaning_result or {}).get(f"{split}_csv_path") for split in DATA_SPLITS})
    try:
        return _run_in_sandbox(box, paper_title, repo_url, slug_dir, repo_dir, review, cleaning_result, commit,
                               warnings, failed, base)
    finally:
        box.close()


def _run_in_sandbox(box, paper_title, repo_url, slug_dir, repo_dir, review, cleaning_result, commit, warnings,
                    failed, base) -> dict:
    env = _setup_repository_environment(repo_dir, box)
    warnings += env["warnings"]
    pip_freeze_path = slug_dir / "pip_freeze.txt"
    pip_freeze_path.write_text(box.freeze() if box.volume else "", encoding="utf-8")
    sandbox_record = {"sandbox": box.describe()}
    if not env["ok"]:
        return failed("The repository's environment could not be set up.", env["error"], **commit,
                      pip_freeze_path=str(pip_freeze_path), modifications=env["modifications"], **sandbox_record)

    entry_point = _detect_entry_point_by_convention(repo_dir)
    entry_point_source = "convention" if entry_point else ""
    command = ["python", entry_point] if entry_point else None
    if not command:
        readme_command = _extract_entry_point_from_readme(repo_dir, paper_title)
        if readme_command:
            # A bare "python"/"python3" becomes the sandbox venv's Python (Sandbox.run).
            command = ["python", *readme_command[1:]] if readme_command[0] in ("python", "python3") \
                else readme_command
            entry_point = " ".join(readme_command)
            entry_point_source = "readme_llm"

    wired = _wire_dataset_into_repo(repo_dir, cleaning_result)
    modifications = env["modifications"] + wired
    if not wired:
        warnings.append("No conventional data/ directory found in the repo; running unmodified -- "
                        "dataset path may need manual wiring if the run fails.")
    run_env = box.data_env({var: key.replace("_csv_path", "") for var, key in DATA_ENV_VARS.items()})

    run_result = None
    if command:
        print(f"[Replication Agent] Running in the sandbox: {' '.join(command)}")
        run_result = _run_installing_missing_modules(command, box, run_env, modifications)
    if run_result is None or (run_result["exit_code"] != 0 and not run_result["timed_out"]):
        failure = (run_result["stderr"] or run_result["stdout"]) if run_result else \
            "No entry point could be determined (no conventional filename, nothing documented in the README)."
        print("[Replication Agent] The repository did not run; trying one repair.")
        fix = _propose_repository_fix(paper_title, repo_dir, command, failure, cleaning_result)
        actions, rejected = _check_fix_actions(fix["actions"], repo_dir, cleaning_result)
        warnings += [f"Repair action rejected: {r}" for r in rejected]
        if fix["diagnosis"]:
            warnings.append(f"Repair diagnosis: {fix['diagnosis']}")
        if actions:
            command, applied = _apply_fix_actions(actions, repo_dir, box, cleaning_result, command)
            modifications += applied
            if any(a["type"] == "command" for a in actions):
                entry_point, entry_point_source = " ".join(command[1:]), "repair_llm"
        if command and actions:
            print(f"[Replication Agent] Running again after the repair: {' '.join(command)}")
            run_result = _run_installing_missing_modules(command, box, run_env, modifications)
        pip_freeze_path.write_text(box.freeze(), encoding="utf-8")  # the repair may have installed more

    box_info = box.describe()
    _write_environment_doc(slug_dir, {
        "Source": "cloned_repository", "Repository": repo_url, "Commit": commit["commit_sha"],
        "Master Agent review": review.get("summary", ""), "Entry point": entry_point,
        "Command": " ".join(command or []), "Dependency install": env["install"] or "none declared",
        "Sandbox": f"Docker {box_info['image']} ({box_info['digest'] or 'digest unknown'}); run with no network, "
                   f"as uid {box_info['run_user']}, {box_info['capabilities']}, memory {box_info['memory']}, "
                   f"{box_info['cpus']} CPUs; host mounts: {', '.join(box_info['host_mounts'])}",
        "Modifications": "; ".join(modifications) or "none",
    })
    if run_result is None:
        return failed("Could not determine how to run this repository.",
                      "No entry point could be determined, and the repair found none.", **commit,
                      pip_freeze_path=str(pip_freeze_path), modifications=modifications, **sandbox_record)

    stdout_path, stderr_path = _save_run_output(slug_dir, run_result)
    result = dict(**commit, entry_point=entry_point, entry_point_source=entry_point_source, command=command,
                  exit_code=run_result["exit_code"], stdout_path=stdout_path, stderr_path=stderr_path,
                  duration_seconds=run_result["duration_seconds"], pip_freeze_path=str(pip_freeze_path),
                  modifications=modifications, sandbox=box_info)
    if run_result["timed_out"]:
        return failed("The repository's run timed out.",
                      f"No result within {REPLICATION_TIMEOUT_SECONDS}s.", status="TIMEOUT", **result)
    if run_result["exit_code"] != 0:
        return failed("The repository still fails after one repair.",
                      f"Entry point exited with code {run_result['exit_code']}.", **result)

    print(f"[Replication Agent] Done. status=OK, exit_code={run_result['exit_code']}.")
    return _default_result(status="OK", **base, warnings=warnings, **result)


def _dataset_profile(cleaning_result: dict) -> str:
    """Plain-text description of the training CSV's actual schema (column
    names, dtypes, example values, split sizes) for the generation and
    repair prompts. Without it the LLM has to guess column names -- the
    label column above all -- and the first run fails on a KeyError."""
    cleaning_result = cleaning_result or {}
    train_path = cleaning_result.get("train_csv_path", "")
    if not train_path or not Path(train_path).is_file():
        return "(The training CSV is not available to describe.)"
    try:
        sample = pd.read_csv(train_path, nrows=PROFILE_SAMPLE_ROWS, low_memory=False)
    except Exception as e:
        return f"(The training CSV could not be read: {e})"

    split_rows = cleaning_result.get("split_rows") or {}
    lines = [
        f"Rows: train={split_rows.get('train', 'unknown')}, test={split_rows.get('test', 'unknown')}, "
        f"validation={split_rows.get('validation', 'unknown')}",
        f"Columns ({len(sample.columns)}), profiled from the first {len(sample)} training rows:",
    ]
    for column in sample.columns[:PROFILE_MAX_COLUMNS]:
        series = sample[column]
        examples = ", ".join(repr(v)[:40] for v in series.dropna().unique()[:3])
        lines.append(f"  - {column} (dtype {series.dtype}, {series.nunique()} distinct): {examples}")
    if len(sample.columns) > PROFILE_MAX_COLUMNS:
        lines.append(f"  ... and {len(sample.columns) - PROFILE_MAX_COLUMNS} more columns")

    binary = [c for c in sample.columns if sample[c].dropna().nunique() == 2]
    if binary:
        lines.append(f"Two-valued columns (candidate labels): {', '.join(binary)}")
    lines.append("'_source_dataset' is a provenance column added by this project's Cleaning Agent -- "
                 "it must never be used as a feature.")
    return "\n".join(lines)


_SCRIPT_REQUIREMENTS = """Requirements for the script:
- Read the data ONLY from these environment variables (already cleaned and split -- do not re-split):
  train = pd.read_csv(os.environ["TRAIN_CSV"]); test = pd.read_csv(os.environ["TEST_CSV"]).
  os.environ["VALIDATION_CSV"] is also available if the method needs a validation set.
- Choose the prediction target from the schema above (the column that is the paper's prediction target).
- Remove from the features: the target, '_source_dataset', and identifier-like columns (row ids,
  unique per row, names, free text). Encode categorical columns and handle missing values.
- Fit on train, evaluate on test. Use random_state=42 everywhere randomness exists.
- It must finish in under 20 minutes on a CPU-only laptop. If the method cannot train on the full
  training set in that time, train on a stratified subsample and say so with an "# INFERRED:"
  comment and a printed line starting "NOTE:".
- Print each final test-set metric on its own line in exactly this form: {metric_format}
  with the value as a fraction between 0 and 1, e.g. "METRIC accuracy=0.9871".
  Print at least accuracy, precision, recall and f1 (for the positive class), plus roc_auc when the
  model gives scores. Use exactly these lower-case names.
- Mark every detail NOT explicitly stated by the paper (hyperparameters, preprocessing choices, etc.)
  with an inline comment starting "# INFERRED:", since this is a reconstruction, not the authors' code.
- Import only these libraries (plus the Python standard library): {libraries}."""


def _script_requirements() -> str:
    return _SCRIPT_REQUIREMENTS.format(
        metric_format=METRIC_LINE_FORMAT, libraries=", ".join(sorted(DEPENDENCY_ALLOWLIST)),
    )


def _paper_section(paper: dict) -> str:
    """The paper text for a prompt: methodology-focused excerpts of the
    full text, or the stored metadata, labelled as which it is."""
    paper = paper or {}
    if paper.get("status") == "FULL_TEXT":
        label = "Excerpts from the paper's own full text (methodology, preprocessing, split, training, results)"
    else:
        label = "The paper's stored abstract and metadata (the full text was not available)"
    body = paper_fulltext.excerpt(paper.get("text", ""), paper_fulltext.METHOD_KEYWORDS,
                                  max_chars=PAPER_EXCERPT_CHARS)
    return f"{label}:\n\"\"\"\n{body or '(none)'}\n\"\"\""


def _parse_generation(content: str) -> dict:
    code = llm.parse_code(content)
    if not code:
        return {}
    meta = llm.parse_json(content, default={})
    meta = meta if isinstance(meta, dict) else {}
    assumptions = meta.get("inferred_assumptions") or []
    libraries = meta.get("libraries_required") or []
    return {
        "code": code,
        "inferred_assumptions": [str(a) for a in assumptions] if isinstance(assumptions, list) else [],
        "libraries_required": [str(l) for l in libraries] if isinstance(libraries, list) else [],
        # A router (FreeLLMAPI) may serve each call from a different model;
        # the replication records which one wrote its code.
        "generated_by": llm.last_served_by(),
    }


def _memory_context(parent_paper: dict, cleaning_result: dict = None, text: str = "", failure: str = None):
    """Past cases for one of this agent's LLM calls (src/memory/recall.py):
    good and bad baseline scripts for the same model family and task, and --
    with a failure -- earlier failures and Master Agent diagnoses with the
    same failure signature. None when agents do not use memory."""
    from src.memory import features, recall
    task = features.task_features({"parent_paper": parent_paper or {}, "cleaning": cleaning_result,
                                   "replication": {"source": "reconstructed"}}, "Replication Agent")
    agents = ["Replication Agent", "Master Agent"] if failure else ["Replication Agent"]
    return recall.context_for("Replication Agent", task, text=text, search_agents=agents, failure=failure)


def _revise_baseline_script(code: str, instructions: list, parent_paper: dict, paper: dict,
                            dataset_profile: str) -> dict:
    """One LLM call fixing the defects an Evaluation Agent review found in
    the reconstructed baseline (relayed by Master Agent), changing nothing
    else. Returns {} on any failure."""
    fixes = "\n".join(f"- {i}" for i in instructions) or "- (none given)"
    prompt = f"""You are revising a reconstructed baseline machine learning script. A senior data analyst reviewed it
and found defects that make its results invalid or weaker than the paper's method. Fix ONLY these defects; keep the
paper's method, the model, and every other part of the script the same.

Paper title: "{parent_paper.get('title', '')}"

{_paper_section(paper)}

The data the script runs on:
{dataset_profile}

Defects to fix:
{fixes}

Current script:
```python
{code}
```

Rules for the fix: resampling (e.g. SMOTE), scaling, imputation, encoding and feature selection are fitted on the
TRAINING data only and then applied to the test data; the test set is never used to train, tune, select features
or stop training early.

{_script_requirements()}

Respond with exactly two fenced blocks and nothing else. First, a JSON block:
```json
{{"inferred_assumptions": ["<one sentence per detail you inferred while fixing>"], "libraries_required": ["<pip package name>"]}}
```
Then the full revised script:
```python
<the script>
```"""
    try:
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, effort="high", agent="replication_agent",
                           memory=_memory_context(parent_paper, text=" ".join(instructions or [])))
    except Exception as e:
        print(f"[Replication Agent] Revision call failed: {e}")
        return {}
    return _parse_generation(content)


def _generate_baseline_script(parent_paper: dict, cleaning_result: dict, dataset_profile: str = "",
                              paper: dict = None) -> dict:
    """One LLM call asking for a self-contained baseline script
    reconstructed from the paper itself (its full text when available),
    since no implementation could be located. Returns {} on any failure."""
    paper = paper or {"status": "ABSTRACT_ONLY", "text": paper_fulltext._metadata_text(parent_paper or {})}
    prompt = f"""You are reconstructing a baseline machine learning implementation from an academic paper, because no public code implementation could be found. Read the paper text below carefully: follow its stated preprocessing order, data split, resampling, model, hyperparameters and evaluation exactly where it states them.

Paper title: "{parent_paper.get('title', '')}"
Dataset the paper describes: "{parent_paper.get('dataset', '') or ''}"

{_paper_section(paper)}

The data this reconstruction runs on:
{dataset_profile or _dataset_profile(cleaning_result)}

Write a single, self-contained Python training script that implements the method described above as closely as the paper's own text allows.

{_script_requirements()}

Respond with exactly two fenced blocks and nothing else. First, a JSON block:
```json
{{"inferred_assumptions": ["<one sentence per inferred detail>"], "libraries_required": ["<pip package name>"]}}
```
Then the full script:
```python
<the script>
```"""

    try:
        context = _memory_context(parent_paper, cleaning_result,
                                  text=f"{parent_paper.get('title', '')} {parent_paper.get('methodology', '')}")
        # Fast mode (plan 7.5): a kept past script for the same model family and
        # task is adapted rather than written anew. Every check still runs.
        case = context.fast_mode_case() if context else None
        if case:
            print(f"[Replication Agent] Fast mode: adapting past case {case['label']} "
                  f"(memory {case['memory']['id'][:8]}).")
            prompt += (f"\n\nFAST MODE: past case [{case['label']}] below is a kept baseline script for the same "
                       "model family and task. Adapt that script to THIS paper and THIS data (column names, label, "
                       "preprocessing and model settings the paper states) instead of writing one from nothing. "
                       "Every requirement above still applies, and the paper's own text wins where they differ.")
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, effort="high", agent="replication_agent",
                           memory=context)
    except Exception as e:
        print(f"[Replication Agent] Script generation call failed: {e}")
        return {}
    result = _parse_generation(content)
    if result and case:
        result["fast_mode_from"] = case["memory"]["id"]
        result["fast_mode_usage_id"] = context.usage_ids.get(case["memory"]["id"])
    return result


def _repair_generated_script(code: str, stderr: str, parent_paper: dict, stdout: str = "",
                             dataset_profile: str = "", installed_libraries: list = None) -> str:
    """One LLM call feeding back the failure, asking for a fixed script.
    Returns "" if no fix could be parsed out."""
    prompt = f"""The following Python script, a reconstruction of an academic paper's method, did not complete successfully.

Paper title: "{parent_paper.get('title', '')}"

The data it runs on:
{dataset_profile or "(not available)"}

Libraries installed in its environment: {", ".join(installed_libraries or []) or "unknown"}

Script:
```python
{code}
```

Error output (stderr, last part):
{(stderr or "(empty)")[-STDERR_FEEDBACK_CHARS:]}

Standard output (last part):
{(stdout or "(empty)")[-1500:]}

Fix the script so it runs successfully and meets every requirement below, keeping the same method and the inline "# INFERRED:" comments.

{_script_requirements()}

Respond with the full fixed script in one fenced block and nothing else:
```python
<the fixed script>
```"""

    try:
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, effort="high", agent="replication_agent",
                           memory=_memory_context(parent_paper, text=(stderr or "")[-1500:], failure=stderr))
    except Exception as e:
        print(f"[Replication Agent] Repair call failed: {e}")
        return ""
    return llm.parse_code(content)


def _install_libraries(venv_dir: Path, cwd: Path, libraries: list) -> tuple:
    """Install libraries into the venv: all at once, and on failure one at
    a time so one unavailable package (e.g. no wheel for this Python
    version) doesn't leave the environment empty. Returns
    (installed, failed)."""
    if not libraries:
        return [], []
    if _pip_install(venv_dir, cwd, ["install", *libraries])["exit_code"] == 0:
        return list(libraries), []
    installed, failed = [], []
    for library in libraries:
        ok = _pip_install(venv_dir, cwd, ["install", library])["exit_code"] == 0
        (installed if ok else failed).append(library)
    return installed, failed


def _archive_previous_baseline(slug_dir: Path, recon_dir: Path, revision_round: int) -> Path | None:
    """Keep the baseline being revised (script, notes, output) under
    revisions/round_<n>/, so the revision never destroys it (CLAUDE.md:
    Baseline Preservation). Returns the archived script, or None."""
    archive = slug_dir / "revisions" / f"round_{revision_round}"
    if (archive / "train.py").is_file():
        # A retry of the same round: the archive already holds the baseline
        # from before the round; never overwrite it with a failed attempt.
        return archive / "train.py"
    script = recon_dir / "train.py"
    if not script.is_file():
        return None
    archive.mkdir(parents=True, exist_ok=True)
    for source in (script, recon_dir / "RECONSTRUCTION_NOTES.md", slug_dir / "run_stdout.txt",
                   slug_dir / "run_stderr.txt"):
        if source.is_file():
            shutil.copy2(source, archive / source.name)
    return archive / "train.py"


def _replicate_by_reconstruction(parent_paper: dict, cleaning_result: dict, revision: dict = None) -> dict:
    paper_title = (parent_paper or {}).get("title", "")
    slug_dir = REPLICATION_DIR / _slugify(paper_title or "reconstructed")
    recon_dir = slug_dir / "reconstructed"
    attempts_dir = recon_dir / "attempts"
    venv_dir = slug_dir / "venv"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    revision = revision or {}
    revision_round = int(revision.get("round") or 0)
    instructions = [str(i) for i in revision.get("instructions") or []]

    paper = paper_fulltext.get_parent_paper_text(parent_paper or {})
    dataset_profile = _dataset_profile(cleaning_result)
    previous_assumptions = []
    archived = _archive_previous_baseline(slug_dir, recon_dir, revision_round) if revision_round else None

    if archived and revision.get("reuse_script"):
        # More data, same code: only the data changes (one variable at a time).
        print(f"\n[Replication Agent] Re-running the same baseline script for '{paper_title}' on the new data "
              f"(round {revision_round})...")
        previous_assumptions = [str(a) for a in revision.get("previous_inferred_assumptions") or []]
        generated = {"code": archived.read_text(encoding="utf-8", errors="ignore"), "inferred_assumptions": [],
                     "libraries_required": [],
                     "generated_by": ", ".join(map(str, revision.get("previous_generated_by") or []))}
        failure = "The previous baseline script could not be read."
    elif archived and instructions:
        print(f"\n[Replication Agent] Revising the reconstructed baseline for '{paper_title}' "
              f"(round {revision_round}): {'; '.join(instructions)}")
        previous_assumptions = [str(a) for a in revision.get("previous_inferred_assumptions") or []]
        generated = _revise_baseline_script(archived.read_text(encoding="utf-8", errors="ignore"), instructions,
                                            parent_paper, paper, dataset_profile)
        failure = "LLM could not revise the baseline script."
    else:
        print(f"\n[Replication Agent] No implementation found; reconstructing a baseline from the paper "
              f"for '{paper_title}' ({paper.get('status', 'ABSTRACT_ONLY')})...")
        if revision_round and not archived:
            warnings.append("No previous reconstructed script to revise; generated a new one with the requested fixes.")
        generated = _generate_baseline_script(parent_paper, cleaning_result, dataset_profile, paper)
        failure = "LLM could not generate a baseline script from the paper."
    if not generated:
        return _default_result(
            status="BLOCKED", source="reconstructed", replication_dir=str(slug_dir),
            paper_text_status=paper.get("status", ""), revision_round=revision_round,
            revision_instructions=instructions, warnings=[failure],
            notes="Reconstruction failed at generation.",
        )
    generated["inferred_assumptions"] = previous_assumptions + generated.get("inferred_assumptions", [])

    requested_libs = set(generated.get("libraries_required") or [])
    rejected_libs = sorted(requested_libs - DEPENDENCY_ALLOWLIST)
    if rejected_libs:
        warnings.append(f"Dropped non-allowlisted libraries requested by the generated script: {', '.join(rejected_libs)}.")
    libraries = sorted((requested_libs & DEPENDENCY_ALLOWLIST) | set(BASE_LIBRARIES))

    venv_result = _create_venv(venv_dir)
    if venv_result["exit_code"] != 0:
        return _default_result(
            status="BLOCKED", source="reconstructed", replication_dir=str(slug_dir),
            inferred_assumptions=generated.get("inferred_assumptions", []),
            warnings=warnings + [f"Virtualenv creation failed: {venv_result['stderr'] or venv_result['stdout']}"],
            notes="Could not create an isolated environment for the reconstructed script.",
        )

    print(f"[Replication Agent] Installing {', '.join(libraries)} into {venv_dir}...")
    installed_libs, failed_libs = _install_libraries(venv_dir, recon_dir, libraries)
    if failed_libs:
        warnings.append(f"Could not install: {', '.join(failed_libs)}.")

    pip_freeze_path = slug_dir / "pip_freeze.txt"
    pip_freeze_path.write_text(_pip_freeze(venv_dir), encoding="utf-8")

    script_path = recon_dir / "train.py"
    command = [str(_venv_python(venv_dir)), str(script_path)]
    env = data_env(cleaning_result)
    code = generated["code"]
    attempts = 0
    run_result = None
    metrics = {}
    generated_by = [m for m in [generated.get("generated_by", "")] if m]

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        attempts = attempt
        script_path.write_text(code, encoding="utf-8")
        # Every attempt is kept, so the changes needed to make the
        # reconstruction run are on record (CLAUDE.md: record changes
        # required to make the implementation run).
        (attempts_dir / f"attempt_{attempt}.py").write_text(code, encoding="utf-8")
        print(f"[Replication Agent] Running reconstructed script (attempt {attempt}/{MAX_REPAIR_ATTEMPTS})...")
        run_result = _run_bounded(command, cwd=recon_dir, timeout=REPLICATION_TIMEOUT_SECONDS, env=env)
        (attempts_dir / f"attempt_{attempt}_stderr.txt").write_text(run_result.get("stderr") or "", encoding="utf-8")
        metrics = parse_metric_lines(run_result.get("stdout", ""))
        if attempt == 1 and generated.get("fast_mode_from") and \
                (run_result["timed_out"] or not (run_result["exit_code"] == 0 and metrics)):
            # The adapted past script failed its first run: its memory ranks lower
            # (judged now, before any repair can make the attempt pass).
            from src.memory import learning
            learning.report_fast_mode_failure(generated.get("fast_mode_usage_id"), generated["fast_mode_from"])
        if run_result["timed_out"] or (run_result["exit_code"] == 0 and metrics):
            break
        if run_result["exit_code"] == 0:
            stderr_feedback = (f"The script exited successfully but printed no lines in the required "
                               f"'{METRIC_LINE_FORMAT}' form.\n{run_result.get('stderr') or ''}")
        else:
            stderr_feedback = run_result.get("stderr") or ""
        print(f"[Replication Agent] Attempt {attempt} failed: {stderr_feedback.strip().splitlines()[-1:] or ['(no output)']}")
        if attempt < MAX_REPAIR_ATTEMPTS:
            repaired = _repair_generated_script(
                code, stderr_feedback, parent_paper, stdout=run_result.get("stdout", ""),
                dataset_profile=dataset_profile, installed_libraries=installed_libs,
            )
            if not repaired:
                warnings.append(f"Repair call after attempt {attempt} returned no script.")
                break
            if llm.last_served_by():
                generated_by.append(llm.last_served_by())
            code = repaired

    _write_notes_doc(recon_dir, generated.get("inferred_assumptions", []))
    stdout_path, stderr_path = _save_run_output(slug_dir, run_result)
    _write_environment_doc(slug_dir, {
        "Source": "reconstructed", "Entry point": "train.py", "Command": " ".join(command),
        "Data environment": ", ".join(f"{k}={v}" for k, v in env.items()) or "none",
        "Libraries installed": ", ".join(installed_libs) or "none",
        "Script generated/repaired by (in order)": ", ".join(generated_by) or "unknown",
    })

    if run_result["timed_out"]:
        status = "TIMEOUT"
    elif run_result["exit_code"] == 0 and metrics:
        status = "OK"
    else:
        status = "BLOCKED"
        if run_result["exit_code"] == 0:
            warnings.append(f"Reconstructed script ran but printed no '{METRIC_LINE_FORMAT}' lines "
                            f"after {attempts} attempt(s).")
        else:
            warnings.append(f"Reconstructed script still failing after {attempts} attempt(s).")

    print(f"[Replication Agent] Done. status={status}, exit_code={run_result['exit_code']}, "
          f"metrics={metrics or 'none'}.")

    return _default_result(
        status=status, source="reconstructed", replication_dir=str(slug_dir),
        entry_point="train.py", entry_point_source="generated", command=command,
        exit_code=run_result["exit_code"], stdout_path=stdout_path, stderr_path=stderr_path,
        duration_seconds=run_result["duration_seconds"], pip_freeze_path=str(pip_freeze_path),
        inferred_assumptions=generated.get("inferred_assumptions", []), warnings=warnings,
        notes=f"Repair attempts: {attempts}.", generated_by=generated_by,
        paper_text_status=paper.get("status", ""), revision_round=revision_round,
        revision_instructions=instructions, fast_mode_from=generated.get("fast_mode_from"),
    )


# Repositories that did not run in this process, by (replication folder,
# URL, reviewed commit) -> their repository_fallback record.
_FAILED_REPOSITORIES = {}


def run(parent_paper: dict, code_discovery_result: dict, cleaning_result: dict, revision: dict = None) -> dict:
    """
    Establish the baseline reproduction: run the discovered repository if
    Code Discovery Agent found one, else reconstruct from the paper itself.
    Every branch returns the same fully-populated result shape.

    revision ({"round", "instructions", "reuse_script",
    "previous_inferred_assumptions", "previous_generated_by"}) is given only
    by Master Agent, relaying an Evaluation Agent request: with
    instructions, a reconstructed baseline is revised to fix exactly those
    defects; with reuse_script (after the Data Agent collected more data),
    the same script runs again on the new data. A cloned repository is
    never revised (it is the authors' code); it is simply run again.

    A cloned repository that does not run, even after one repair (see
    _replicate_from_repository), is not usable code, so the baseline is
    reconstructed from the paper instead (CLAUDE.md's Preferred
    Implementation Order, step 6); the repository's failure is kept in
    repository_fallback and the warnings. A repository that failed once is
    not tried again in this process: Master Agent's later attempts go
    straight to the reconstruction.
    """
    code_discovery_result = code_discovery_result or {}
    cleaning_result = cleaning_result or {}
    repository = code_discovery_result.get("repository")
    if code_discovery_result.get("status") == "FOUND" and repository:
        key = (str(REPLICATION_DIR), repository.get("repo_url", ""),
               (repository.get("review") or {}).get("commit_sha", ""))
        repository_fallback = _FAILED_REPOSITORIES.get(key)
        result = None
        if repository_fallback is None:
            result = _replicate_from_repository(parent_paper, repository, cleaning_result)
            if result.get("repository_failed"):
                repository_fallback = {"repo_url": result["repo_url"], "commit_sha": result["commit_sha"],
                                       "reason": result["notes"], "warnings": list(result["warnings"])}
                _FAILED_REPOSITORIES[key] = repository_fallback
        if repository_fallback is not None:
            print(f"[Replication Agent] The repository does not run ({repository_fallback['reason']}); "
                  "reconstructing the baseline from the paper instead.")
            result = _replicate_by_reconstruction(parent_paper, cleaning_result, revision)
            result["repository_fallback"] = repository_fallback
            result["warnings"] = [
                f"The discovered repository {repository_fallback['repo_url']} did not run "
                f"({repository_fallback['reason']}), so the baseline was reconstructed from the paper."
            ] + repository_fallback["warnings"] + list(result.get("warnings") or [])
        elif revision and revision.get("instructions"):
            result["warnings"] = list(result.get("warnings") or []) + [
                "Revision not applied: the cloned repository is the authors' own code and is preserved as the "
                "baseline. Requested fixes: " + "; ".join(map(str, revision["instructions"]))]
    else:
        result = _replicate_by_reconstruction(parent_paper, cleaning_result, revision)
    result["dataset_sampled"] = bool(cleaning_result.get("sampled"))
    result["dataset_semantic_mappings"] = sum(
        1 for m in cleaning_result.get("column_mappings", []) for c in m.get("columns", []) if c.get("accepted")
    )
    return result


def handoff(result: dict, context: dict) -> list:
    """The Replication Agent's request to Master Agent when it finishes: run
    the Evaluation Agent on the new baseline."""
    return [handoff_request("replication_agent", "Evaluation Agent", {},
                            f"Baseline ran (status {(result or {}).get('status', '')}, source "
                            f"{(result or {}).get('source', '')}); it must be compared with the paper.")]
