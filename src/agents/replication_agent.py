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
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src import llm
from src.agents.evaluation_agent import METRIC_LINE_FORMAT, parse_metric_lines
from src.search import paper_fulltext
from src.orchestration.handoff import request as handoff_request

REPO_ROOT = Path(__file__).parent.parent.parent
REPLICATION_DIR = REPO_ROOT / "replication"

CLONE_TIMEOUT_SECONDS = 120
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
    a generated script prints."""
    start = time.monotonic()
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
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
        f"Python: {sys.version}",
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
    pip_freeze_path.write_text(_pip_freeze(venv_dir), encoding="utf-8")
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
    """
    code_discovery_result = code_discovery_result or {}
    cleaning_result = cleaning_result or {}
    if code_discovery_result.get("status") == "FOUND" and code_discovery_result.get("repository"):
        result = _replicate_from_repository(parent_paper, code_discovery_result["repository"], cleaning_result)
        if revision and revision.get("instructions"):
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
