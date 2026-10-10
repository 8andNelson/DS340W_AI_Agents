"""
Replication Agent (Milestone 10).

Establishes the baseline reproduction of the Parent Paper, branching
entirely on Code Discovery Agent's result:

  1. FOUND -- clone the discovered repository, set up an isolated
     environment, and run it against the Cleaning Agent's dataset.
  2. NOT_FOUND -- reconstruct a baseline implementation from the paper's
     own stored text (title/abstract/methodology/results_summary), since
     CLAUDE.md's Preferred Implementation Order allows this as the last
     resort. Every inferred detail is documented; the result is never
     presented as the authors' own code.

This agent clones and executes code from the internet (or LLM-generated
code), which is a different risk category than every other agent in this
project (which only ever did read-only search, LLM calls, and file I/O on
data the project itself controls). Containment, not a human approval
gate, is the safeguard (per CLAUDE.md's Autonomous Operation policy):

  - Every install/run happens inside a fresh virtualenv under
    replication/<slug>/venv/ -- the project's own .venv is never touched.
  - Dependencies are only ever installed from a known, declared source:
    the repo's own committed file (Path A) or a fixed allowlist of common
    ML libraries (Path B, since that list comes from a generative call,
    not a repo's own committed file -- a meaningfully different trust
    level).
  - Every subprocess call uses an argv list (never shell=True) and a
    bounded timeout.
  - No setup step is ever auto-elevated (sudo/admin); one that needs it
    is reported as a failure, not bypassed.

Comparing reproduced metrics against the paper's published results is
the Evaluation Agent's job, not this one's -- this agent's job ends at
"did the baseline execute and produce output."
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL

REPO_ROOT = Path(__file__).parent.parent.parent
REPLICATION_DIR = REPO_ROOT / "replication"

CLONE_TIMEOUT_SECONDS = 120
VENV_TIMEOUT_SECONDS = 120
INSTALL_TIMEOUT_SECONDS = 600
REPLICATION_TIMEOUT_SECONDS = 1800
README_CHAR_LIMIT = 6000
MAX_REPAIR_ATTEMPTS = 2

ENTRY_POINT_CANDIDATES = ("train.py", "main.py", "run.py", "run_experiment.py")
README_FILENAMES = ("README.md", "README.rst", "README.txt", "readme.md")

# Path B's packages come from a generative LLM call, not a repo's own
# committed dependency file -- a meaningfully different trust level, so
# anything outside this fixed set is dropped rather than installed.
DEPENDENCY_ALLOWLIST = {
    "scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "catboost", "torch", "tensorflow", "matplotlib",
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
    "modifications": [],
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


def _run_bounded(cmd: list, cwd: Path = None, timeout: int = REPLICATION_TIMEOUT_SECONDS) -> dict:
    """Run cmd (always an argv list, never a shell string) with a bounded
    timeout, capturing output. Never raises -- every outcome (success,
    non-zero exit, timeout, or a launch failure) comes back as the same
    dict shape so callers don't need a try/except of their own."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "duration_seconds": time.monotonic() - start, "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": None, "stdout": e.stdout or "", "stderr": e.stderr or "",
            "duration_seconds": time.monotonic() - start, "timed_out": True,
        }
    except Exception as e:
        return {
            "exit_code": None, "stdout": "", "stderr": str(e),
            "duration_seconds": time.monotonic() - start, "timed_out": False,
        }


def _clone_repo(repo_url: str, dest: Path) -> dict:
    clone_result = _run_bounded(["git", "clone", "--depth", "1", repo_url, str(dest)], timeout=CLONE_TIMEOUT_SECONDS)
    if clone_result["exit_code"] != 0:
        return {"ok": False, "error": clone_result["stderr"] or clone_result["stdout"], "commit_sha": ""}
    sha_result = _run_bounded(["git", "rev-parse", "HEAD"], cwd=dest, timeout=30)
    commit_sha = sha_result["stdout"].strip() if sha_result["exit_code"] == 0 else ""
    return {"ok": True, "error": "", "commit_sha": commit_sha}


def _create_venv(venv_dir: Path) -> dict:
    return _run_bounded([sys.executable, "-m", "venv", str(venv_dir)], timeout=VENV_TIMEOUT_SECONDS)


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
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}

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
    stdout_path.write_text(run_result.get("stdout", "") or "")
    stderr_path.write_text(run_result.get("stderr", "") or "")
    return str(stdout_path), str(stderr_path)


def _write_environment_doc(slug_dir: Path, details: dict) -> None:
    lines = [
        "# Replication Environment",
        f"Python: {sys.version}",
        f"OS: {platform.platform()}",
    ]
    for key, value in details.items():
        lines.append(f"{key}: {value}")
    (slug_dir / "ENVIRONMENT.md").write_text("\n".join(lines))


def _write_notes_doc(recon_dir: Path, inferred_assumptions: list) -> None:
    lines = [
        "# Reconstruction Notes",
        "",
        "This code was generated from the paper's own stored text, NOT the authors' "
        "original implementation. Every detail not explicitly stated by the paper:",
    ]
    lines.extend(f"- {a}" for a in inferred_assumptions)
    (recon_dir / "RECONSTRUCTION_NOTES.md").write_text("\n".join(lines))


def _replicate_from_repository(parent_paper: dict, repository: dict, cleaning_result: dict) -> dict:
    paper_title = (parent_paper or {}).get("title", "")
    repo_url = repository.get("repo_url", "")
    slug_dir = REPLICATION_DIR / _slugify(paper_title or repo_url)
    repo_dir = slug_dir / "repo"
    venv_dir = slug_dir / "venv"
    warnings = []

    print(f"\n[Replication Agent] Cloning {repo_url} for baseline replication...")
    slug_dir.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    clone = _clone_repo(repo_url, repo_dir)
    if not clone["ok"]:
        print(f"[Replication Agent] Clone failed: {clone['error']}")
        return _default_result(
            status="BLOCKED", source="cloned_repository", repo_url=repo_url, replication_dir=str(slug_dir),
            warnings=[f"Clone failed: {clone['error']}"], notes="Could not clone the discovered repository.",
        )

    venv_result = _create_venv(venv_dir)
    if venv_result["exit_code"] != 0:
        return _default_result(
            status="BLOCKED", source="cloned_repository", repo_url=repo_url, commit_sha=clone["commit_sha"],
            replication_dir=str(slug_dir),
            warnings=[f"Virtualenv creation failed: {venv_result['stderr'] or venv_result['stdout']}"],
            notes="Could not create an isolated environment for the cloned repo.",
        )

    install_args, blocked_reason = _select_dependency_install(repository.get("dependency_files", []))
    if install_args is None:
        return _default_result(
            status="BLOCKED", source="cloned_repository", repo_url=repo_url, commit_sha=clone["commit_sha"],
            replication_dir=str(slug_dir), warnings=[blocked_reason],
            notes="Dependency installation could not proceed.",
        )

    install_result = _pip_install(venv_dir, repo_dir, install_args)
    pip_freeze_path = slug_dir / "pip_freeze.txt"
    pip_freeze_path.write_text(_pip_freeze(venv_dir))
    if install_result["exit_code"] != 0:
        return _default_result(
            status="BLOCKED", source="cloned_repository", repo_url=repo_url, commit_sha=clone["commit_sha"],
            replication_dir=str(slug_dir), pip_freeze_path=str(pip_freeze_path),
            warnings=[f"Dependency installation failed: {(install_result['stderr'] or '')[-2000:]}"],
            notes="pip install failed for the discovered dependency file.",
        )

    entry_point = _detect_entry_point_by_convention(repo_dir)
    entry_point_source = "convention" if entry_point else ""
    command = [str(_venv_python(venv_dir)), entry_point] if entry_point else None

    if not command:
        readme_command = _extract_entry_point_from_readme(repo_dir, paper_title)
        if readme_command:
            # The LLM returns a bare "python"/"python3" token -- substitute
            # the isolated venv's own interpreter so the dependencies just
            # installed into it are actually the ones used, same as the
            # convention-detected case above.
            if readme_command[0] in ("python", "python3"):
                command = [str(_venv_python(venv_dir)), *readme_command[1:]]
            else:
                command = readme_command
            entry_point = " ".join(readme_command)
            entry_point_source = "readme_llm"

    if not command:
        return _default_result(
            status="BLOCKED", source="cloned_repository", repo_url=repo_url, commit_sha=clone["commit_sha"],
            replication_dir=str(slug_dir), pip_freeze_path=str(pip_freeze_path),
            warnings=["No entry point could be determined (no conventional filename, nothing documented in the README)."],
            notes="Could not determine how to run this repository.",
        )

    modifications = _wire_dataset_into_repo(repo_dir, cleaning_result)
    if not modifications:
        warnings.append("No conventional data/ directory found in the repo; running unmodified -- "
                         "dataset path may need manual wiring if the run fails.")

    print(f"[Replication Agent] Running: {' '.join(command)}")
    run_result = _run_bounded(command, cwd=repo_dir, timeout=REPLICATION_TIMEOUT_SECONDS)
    stdout_path, stderr_path = _save_run_output(slug_dir, run_result)
    _write_environment_doc(slug_dir, {
        "Source": "cloned_repository", "Repository": repo_url, "Commit": clone["commit_sha"],
        "Entry point": entry_point, "Command": " ".join(command), "Dependency install": " ".join(install_args),
    })

    if run_result["timed_out"]:
        status = "TIMEOUT"
    elif run_result["exit_code"] == 0:
        status = "OK"
    else:
        status = "BLOCKED"
        warnings.append(f"Entry point exited with code {run_result['exit_code']}.")

    print(f"[Replication Agent] Done. status={status}, exit_code={run_result['exit_code']}.")

    return _default_result(
        status=status, source="cloned_repository", repo_url=repo_url, commit_sha=clone["commit_sha"],
        replication_dir=str(slug_dir), entry_point=entry_point, entry_point_source=entry_point_source,
        command=command, exit_code=run_result["exit_code"], stdout_path=stdout_path, stderr_path=stderr_path,
        duration_seconds=run_result["duration_seconds"], pip_freeze_path=str(pip_freeze_path),
        modifications=modifications, warnings=warnings,
    )


def _generate_baseline_script(parent_paper: dict, cleaning_result: dict) -> dict:
    """One LLM call asking for a self-contained baseline script
    reconstructed purely from the paper's own stored text, since no
    implementation could be located. Returns {} on any failure."""
    prompt = f"""You are reconstructing a baseline machine learning implementation purely from an academic paper's own text, because no public code implementation could be found.

Paper title: "{parent_paper.get('title', '')}"
Abstract: "{parent_paper.get('abstract', '') or ''}"
Methodology: "{parent_paper.get('methodology', '') or ''}"
Results summary: "{parent_paper.get('results_summary', '') or ''}"
Dataset the paper describes: "{parent_paper.get('dataset', '') or ''}"

Write a single, self-contained Python training script that:
- Reads the training data from "{cleaning_result.get('train_csv_path', 'train.csv')}" and the test data from "{cleaning_result.get('test_csv_path', 'test.csv')}" (both already split -- do not re-split).
- Implements the method described above as closely as the paper's own text allows.
- Prints its evaluation metrics (accuracy, precision, recall, F1 at minimum) to stdout in a clearly labeled format.
- Marks every detail NOT explicitly stated by the paper (hyperparameters, preprocessing choices, etc.) with an inline comment starting "# INFERRED:", since this is a reconstruction, not the authors' own code.
- Uses only well-established libraries (e.g. scikit-learn, pandas, numpy, xgboost, lightgbm, catboost, torch, tensorflow).

Return ONLY a JSON object of this exact shape, no markdown:
{{
  "code": "<the full Python script as a single string, with \\n for newlines>",
  "inferred_assumptions": ["<one sentence per inferred detail>"],
  "libraries_required": ["<pip package name>"]
}}"""

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}

    return data if isinstance(data, dict) and data.get("code") else {}


def _repair_generated_script(code: str, stderr: str, parent_paper: dict) -> str:
    """One LLM call feeding back the failure, asking for a fixed script.
    Returns "" if no fix could be parsed out."""
    prompt = f"""The following Python script failed when executed.

Paper title: "{parent_paper.get('title', '')}"

Script:
{code}

Error output:
{stderr[-3000:]}

Fix the script so it runs successfully, keeping the same approach and inline "# INFERRED:" comments. Return ONLY a JSON object:
{{"code": "<the full fixed Python script as a single string, with \\n for newlines>"}}"""

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}

    fixed = data.get("code") if isinstance(data, dict) else None
    return fixed if isinstance(fixed, str) and fixed.strip() else ""


def _replicate_by_reconstruction(parent_paper: dict, cleaning_result: dict) -> dict:
    paper_title = (parent_paper or {}).get("title", "")
    slug_dir = REPLICATION_DIR / _slugify(paper_title or "reconstructed")
    recon_dir = slug_dir / "reconstructed"
    venv_dir = slug_dir / "venv"
    recon_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    print(f"\n[Replication Agent] No implementation found; reconstructing a baseline from the paper "
          f"for '{paper_title}'...")

    generated = _generate_baseline_script(parent_paper, cleaning_result)
    if not generated:
        return _default_result(
            status="BLOCKED", source="reconstructed", replication_dir=str(slug_dir),
            warnings=["LLM could not generate a baseline script from the paper's stored text."],
            notes="Reconstruction failed at generation.",
        )

    requested_libs = set(generated.get("libraries_required") or [])
    allowed_libs = sorted(requested_libs & DEPENDENCY_ALLOWLIST)
    rejected_libs = sorted(requested_libs - DEPENDENCY_ALLOWLIST)
    if rejected_libs:
        warnings.append(f"Dropped non-allowlisted libraries requested by the generated script: {', '.join(rejected_libs)}.")

    venv_result = _create_venv(venv_dir)
    if venv_result["exit_code"] != 0:
        return _default_result(
            status="BLOCKED", source="reconstructed", replication_dir=str(slug_dir),
            inferred_assumptions=generated.get("inferred_assumptions", []),
            warnings=warnings + [f"Virtualenv creation failed: {venv_result['stderr'] or venv_result['stdout']}"],
            notes="Could not create an isolated environment for the reconstructed script.",
        )

    if allowed_libs:
        install_result = _pip_install(venv_dir, recon_dir, ["install", *allowed_libs])
        if install_result["exit_code"] != 0:
            warnings.append(f"Dependency installation reported errors: {(install_result['stderr'] or '')[-2000:]}")

    pip_freeze_path = slug_dir / "pip_freeze.txt"
    pip_freeze_path.write_text(_pip_freeze(venv_dir))

    script_path = recon_dir / "train.py"
    command = [str(_venv_python(venv_dir)), str(script_path)]
    code = generated["code"]
    attempts = 0
    run_result = None

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        script_path.write_text(code)
        attempts = attempt
        print(f"[Replication Agent] Running reconstructed script (attempt {attempt}/{MAX_REPAIR_ATTEMPTS})...")
        run_result = _run_bounded(command, cwd=recon_dir, timeout=REPLICATION_TIMEOUT_SECONDS)
        if run_result["timed_out"] or run_result["exit_code"] == 0:
            break
        if attempt < MAX_REPAIR_ATTEMPTS:
            repaired = _repair_generated_script(code, run_result["stderr"] or "", parent_paper)
            if not repaired:
                break
            code = repaired

    _write_notes_doc(recon_dir, generated.get("inferred_assumptions", []))
    stdout_path, stderr_path = _save_run_output(slug_dir, run_result)
    _write_environment_doc(slug_dir, {
        "Source": "reconstructed", "Entry point": "train.py", "Command": " ".join(command),
        "Libraries installed": ", ".join(allowed_libs) or "none",
    })

    if run_result["timed_out"]:
        status = "TIMEOUT"
    elif run_result["exit_code"] == 0:
        status = "OK"
    else:
        status = "BLOCKED"
        warnings.append(f"Reconstructed script still failing after {attempts} attempt(s).")

    print(f"[Replication Agent] Done. status={status}, exit_code={run_result['exit_code']}.")

    return _default_result(
        status=status, source="reconstructed", replication_dir=str(slug_dir),
        entry_point="train.py", entry_point_source="generated", command=command,
        exit_code=run_result["exit_code"], stdout_path=stdout_path, stderr_path=stderr_path,
        duration_seconds=run_result["duration_seconds"], pip_freeze_path=str(pip_freeze_path),
        inferred_assumptions=generated.get("inferred_assumptions", []), warnings=warnings,
        notes=f"Repair attempts: {attempts}.",
    )


def run(parent_paper: dict, code_discovery_result: dict, cleaning_result: dict) -> dict:
    """
    Establish the baseline reproduction: run the discovered repository if
    Code Discovery Agent found one, else reconstruct from the paper's own
    text. Every branch returns the same fully-populated result shape.
    """
    code_discovery_result = code_discovery_result or {}
    if code_discovery_result.get("status") == "FOUND" and code_discovery_result.get("repository"):
        return _replicate_from_repository(parent_paper, code_discovery_result["repository"], cleaning_result or {})
    return _replicate_by_reconstruction(parent_paper, cleaning_result or {})
