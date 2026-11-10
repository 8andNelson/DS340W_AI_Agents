"""
Local web dashboard for the multi-agent pipeline.

    python -m src.dashboard            # then open http://127.0.0.1:8050

The dashboard and the pipeline are separate programs. The pipeline does not
import, call or know about the dashboard, and runs the same with or without
it. The dashboard:

  * starts a run the same way a user would (`python run.py "<topic>"`), as a
    detached process that keeps going if the dashboard stops (runner.py);
  * only *reads* the pipeline's outputs through readers.py, which never
    locks a file or the database against the pipeline;
  * writes only to its own folder, dashboard_files/ (fetched paper PDFs,
    script copies, run consoles).

Everything is shown per run. The run database (database/project_runs.db,
written by src/orchestration/run_store.py) records each run's agent
attempts -- with their structured output and report -- and every file the
run produced, with its contents. Each endpoint takes `?run=<run_id>`
(default: the current run, the newest one), and shows only that run's
papers, datasets, scripts, documents, reports and Master Agent decisions.
Files on disk are shared by all runs and overwritten by the next one, so a
file is served from disk only while it is still the version the run
recorded, and otherwise from the copy in the run database.

It imports nothing from the pipeline's packages. It binds to 127.0.0.1 only.
"""
import csv
import io
import json
import re
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from src.dashboard import editor, pdf_fetch, readers
from src.dashboard.freellmapi import FreeLLMAPIManager
from src.dashboard.runner import CONSOLE_DIR, PipelineRunner, RunError

REPO_ROOT = Path(__file__).parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"

# Pipeline outputs (read only)
STATE_FILE = REPO_ROOT / "logs" / "project_state.json"
MASTER_LOG = REPO_ROOT / "logs" / "master_agent_log.json"
RUN_DB = REPO_ROOT / "database" / "project_runs.db"

# The dashboard's own files
DASHBOARD_FILES = REPO_ROOT / "dashboard_files"
PDF_DIR = DASHBOARD_FILES / "pdfs"
SCRIPT_COPY_DIR = DASHBOARD_FILES / "scripts"  # recorded versions opened in VS Code
CONSOLE_INDEX = CONSOLE_DIR / "index.json"     # run_id -> console file of a dashboard-started run

# Recorded files are read from disk only inside these folders.
FILE_ROOTS = [REPO_ROOT / d for d in
              ("data", "papers", "reports", "presentations", "replication", "experiments", "dashboard_files")]
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", "node_modules"}

PREVIEW_MAX_ROWS = 500
SCRIPT_VIEW_MAX_BYTES = 2 * 1024 * 1024
CODE_SUFFIXES = {".py", ".ipynb", ".r", ".sh"}
# The text files run_store keeps from replication/ and experiments/.
SCRIPT_TEXT_SUFFIXES = CODE_SUFFIXES | {".md", ".txt", ".cfg", ".toml", ".yml", ".yaml", ".json", ".ini"}
TABLE_SUFFIXES = {".csv", ".tsv"}
# A run row still marked "running" with no new agent attempt for this long
# is treated as a run that was killed before it could record its end.
STALE_RUN_HOURS = 6

# Pipeline order and the workflow phase each agent belongs to. Report and
# Presentation are listed so the user can see the full plan; not built yet.
PIPELINE = [
    {"name": "Intake Agent", "phase": "TOPIC_ANALYSIS",
     "role": "Turns the topic into keywords, a domain and the prediction target."},
    {"name": "Research Agent", "phase": "LITERATURE_SEARCH",
     "role": "Searches peer-reviewed literature (2022 to now) for candidate papers."},
    {"name": "Validation Agent", "phase": "MASTER_APPROVAL",
     "role": "Verifies each paper, ranks the qualifying ones and recommends the Parent Paper."},
    {"name": "Data Agent", "phase": "DATA_DISCOVERY",
     "role": "Finds, downloads and checks up to 5 datasets for the Parent Paper."},
    {"name": "Cleaning Agent", "phase": "DATA_CLEANING",
     "role": "Merges the datasets into one master CSV and splits it 70/20/10."},
    {"name": "Code Discovery Agent", "phase": "CODE_DISCOVERY",
     "role": "Looks for the Parent Paper's implementation code."},
    {"name": "Replication Agent", "phase": "BASELINE_REPLICATION",
     "role": "Runs (or reconstructs) the implementation to set a baseline."},
    {"name": "Evaluation Agent", "phase": "BASELINE_EVALUATION",
     "role": "Compares the reproduced metrics with the paper's published results."},
    {"name": "Experiment Agent", "phase": "EXPERIMENTATION",
     "role": "Runs one controlled, single-variable experiment against the baseline."},
    {"name": "Report Agent", "phase": "REPORT_GENERATION",
     "role": "Writes the final research report.", "planned": True},
    {"name": "Presentation Agent", "phase": "PRESENTATION_GENERATION",
     "role": "Builds the presentation slides.", "planned": True},
]
PIPELINE_ORDER = [s["name"] for s in PIPELINE if not s.get("planned")]


# --- Paths and files ---

def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _display_path(path: Path) -> str:
    return _relative(path) if path.resolve().is_relative_to(REPO_ROOT.resolve()) else str(path)


def _within_roots(path: Path) -> bool:
    resolved = path.resolve()
    return (any(resolved.is_relative_to(r.resolve()) for r in FILE_ROOTS)
            and not SKIP_DIRS.intersection(resolved.parts))


def _slug(text: str, max_len: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:max_len] or "paper"


def _paper_pdf_path(paper: dict) -> Path:
    return PDF_DIR / f"{_slug(paper.get('title', ''))}.pdf"


def _fetched_pdf(name: str) -> Path:
    """A fetched paper PDF by file name; aborts on anything else."""
    if not re.fullmatch(r"[a-z0-9-]+\.pdf", name or ""):
        abort(400, "Invalid PDF name.")
    path = PDF_DIR / name
    if not path.is_file():
        abort(404, "PDF not found.")
    return path


def _parse_time(iso: str | None) -> datetime | None:
    try:
        moment = datetime.fromisoformat(iso) if iso else None
    except ValueError:
        return None
    if moment and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


# --- Which run is shown ---

def _run_overview(runner: PipelineRunner) -> tuple[dict, dict | None, bool]:
    """(runner info, latest run from the run database, whether a run is active now)."""
    info = runner.info()
    try:
        run = readers.latest_run(RUN_DB)
    except Exception:  # an unreadable database only hides history
        run = None
    if info["running"]:
        return info, run, True
    if not run or run.get("status") != "running":
        return info, run, False
    # The database says "running". If this dashboard started that run and it
    # has exited, it was stopped or crashed before recording its end.
    started_here = info["started_at"] and run["started_at"][:19] >= info["started_at"][:19]
    if started_here:
        run["status"] = "stopped" if info["stopped_by_user"] else "crashed"
        return info, run, False
    hours = readers.hours_since(readers.last_activity(run))
    if hours is not None and hours > STALE_RUN_HOURS:
        run["status"] = "unknown"
        run["note"] = (f"Marked running, but no agent has finished in {hours:.0f} hours; "
                       "it was probably stopped before it could record its end.")
        return info, run, False
    run["note"] = "This run was started outside the dashboard (for example with python run.py)."
    return info, run, True


def _link_console(runner: PipelineRunner, latest: dict | None) -> None:
    """Once the pipeline has opened its run record, remember which console
    file belongs to it, so the console can be read again from run history."""
    if not (runner.started_at and runner.console_log and latest):
        return
    if latest["started_at"][:19] < runner.started_at[:19] or runner.run_id == latest["run_id"]:
        return
    runner.run_id = latest["run_id"]
    index = readers.read_json(CONSOLE_INDEX, {})
    index[latest["run_id"]] = runner.console_log.name
    try:
        CONSOLE_INDEX.parent.mkdir(parents=True, exist_ok=True)
        CONSOLE_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except OSError:
        pass  # only costs the console of this run in history


def _resolve_run(runner: PipelineRunner) -> dict | None:
    """The run a request asks for (`?run=<id>`), else the current run.

    The current run is the newest one in the run database -- or, in the
    seconds between starting `run.py` and the pipeline opening its run
    record, a placeholder with no results, so no older run's files show.
    """
    info, latest, active = _run_overview(runner)
    _link_console(runner, latest)
    wanted = request.args.get("run") or "current"

    if wanted == "current":
        if info["running"] and (not latest or latest["started_at"][:19] < info["started_at"][:19]):
            return {"run_id": None, "topic": runner.topic, "status": "starting", "started_at": info["started_at"],
                    "finished_at": None, "final_phase": "", "stages": [], "is_current": True, "active": True}
        if not latest:
            return None
        return {**latest, "is_current": True, "active": active}

    if latest and wanted == latest["run_id"]:
        return {**latest, "is_current": True, "active": active}
    run = readers.get_run(RUN_DB, wanted)
    if not run:
        abort(404, "Run not found.")
    stages = [{k: a[k] for k in ("agent", "attempt", "decision", "report_text", "recorded_at")}
              for a in readers.stage_attempts(RUN_DB, wanted)]
    return {**run, "stages": stages, "is_current": False, "active": False}


def _public(run: dict | None) -> dict | None:
    return {k: v for k, v in run.items() if k != "stages"} if run else None


def _attempts(run: dict | None) -> list[dict]:
    return readers.stage_attempts(RUN_DB, run["run_id"]) if run and run.get("run_id") else []


def _artifacts(run: dict | None) -> list[dict]:
    return readers.list_artifacts(RUN_DB, run["run_id"]) if run and run.get("run_id") else []


def _output(attempts: list[dict], agent: str):
    """The agent's last recorded output in this run (an approved one if any)."""
    mine = [a for a in attempts if a["agent"] == agent and a["output"] is not None]
    approved = [a for a in mine if a["decision"] == "Approved"]
    chosen = (approved or mine)[-1:] or [None]
    return chosen[0]["output"] if chosen[0] else None


def _papers(attempts: list[dict]) -> tuple[list[dict], dict]:
    validation = _output(attempts, "Validation Agent") or {}
    papers = validation.get("all") or _output(attempts, "Research Agent") or []
    return (papers if isinstance(papers, list) else []), (validation.get("recommended_parent_paper") or {})


def _summary(attempts: list[dict]) -> dict:
    papers, parent = _papers(attempts)
    data = _output(attempts, "Data Agent") or {}
    cleaning = _output(attempts, "Cleaning Agent") or {}
    experiment = _output(attempts, "Experiment Agent") or {}
    return {
        "papers": len(papers),
        "papers_verified": sum(1 for p in papers if p.get("validation_status") == "VERIFIED"),
        "parent_paper": parent.get("title", ""),
        "datasets": len(data.get("selected_datasets") or []),
        "dataset_entries": data.get("total_entries"),
        "master_rows": cleaning.get("rows_total"),
        "master_target_met": cleaning.get("target_met"),
        "master_target_rows": cleaning.get("target_rows"),
        "data_target": data.get("target_entries"),
        "data_plan": (data.get("data_plan") or {}).get("reasoning", ""),
        "baseline": (_output(attempts, "Replication Agent") or {}).get("status", ""),
        "evaluation": (_output(attempts, "Evaluation Agent") or {}).get("status", ""),
        "experiment": experiment.get("overall") or experiment.get("status", ""),
    }


# --- Recorded files (run_store's artifacts) ---

def _open_artifact(artifact_id: int) -> tuple[dict, object, str]:
    """(record, binary file object, "disk" | "database") for one recorded file:
    the pipeline's file while it is still the recorded version, else the copy
    in the run database. Aborts if neither is available."""
    record = readers.get_artifact_record(RUN_DB, artifact_id)
    if not record:
        abort(404, "File not found in the run database.")
    original = Path(record["original_path"])
    if _within_roots(original) and readers.file_matches(original, record["sha256"]) == "same":
        try:
            return record, readers.open_shared(original), "disk"
        except OSError:
            pass  # replaced between the check and the open: use the stored copy
    blob = readers.open_blob(RUN_DB, record["sha256"]) if record["content_stored"] else None
    if blob is None:
        abort(404, "This file was too large to keep in the run database, and it is no longer on disk "
                   "in the version this run produced.")
    return record, blob, "database"


def _artifact_status(record: dict) -> str:
    original = Path(record["original_path"])
    disk = readers.file_matches(original, record["sha256"]) if _within_roots(original) else "missing"
    return disk


# --- Scripts ---

def _is_script_file(artifact: dict) -> bool:
    return (artifact["kind"] in ("script", "source_file")
            and Path(artifact["original_path"]).suffix.lower() in SCRIPT_TEXT_SUFFIXES)


def _script_role(path: Path, agent: str, output: dict) -> tuple[str, int]:
    """A plain-language role for one file, and its sort order (code first)."""
    name = path.name.lower()
    if agent == "Experiment Agent" and path.suffix.lower() in CODE_SUFFIXES:
        return "Experiment script (one change from the baseline)", 0
    if path.parent.name == "attempts" and name.startswith("attempt_"):
        version = name.split("_")[1].split(".")[0]
        if name.endswith("_stderr.txt"):
            return f"Error output of generated version {version}", 4
        return f"Generated version {version}, before repair", 2
    if path.name == output.get("entry_point") and path.parent.name != "attempts":
        return "Baseline script (the version that ran)", 0
    if path.suffix.lower() in CODE_SUFFIXES:
        return "Source file", 1
    if name in ("run_stdout.txt", "run_stderr.txt") or name.endswith("_stderr.txt"):
        return "Run output", 4
    if name in ("environment.md", "pip_freeze.txt"):
        return "Environment record", 5
    return "Notes", 3


_CONTEXT_KEYS = {
    "Replication Agent": ("source", "repo_url", "commit_sha", "entry_point", "exit_code", "generated_by",
                          "inferred_assumptions"),
    "Experiment Agent": ("category", "hypothesis", "what_changed", "overall", "exit_code", "generated_by"),
}


def _script_groups(run: dict | None) -> list[dict]:
    attempts = _attempts(run)
    groups: dict[str, dict] = {}
    baseline_ids = {}
    for a in _artifacts(run):
        if not _is_script_file(a):
            continue
        path = Path(a["original_path"])
        output = _output(attempts, a["agent"]) or {}
        role, order = _script_role(path, a["agent"], output)
        group = groups.setdefault(a["agent"], {
            "agent": a["agent"], "files": [],
            "context": {k: output.get(k) for k in _CONTEXT_KEYS.get(a["agent"], ()) if output.get(k) not in (None, "", [])},
        })
        group["files"].append({
            "id": a["id"], "name": path.name, "path": _display_path(path),
            "role": role, "order": order, "is_code": path.suffix.lower() in CODE_SUFFIXES,
            "size_bytes": a["size_bytes"], "disk": _artifact_status(a), "stored": bool(a["content_stored"]),
        })
        if a["agent"] == "Replication Agent" and order == 0:
            baseline_ids[path.name] = a["id"]
    for group in groups.values():
        group["files"].sort(key=lambda f: (f["order"], f["path"]))
        for f in group["files"]:
            if group["agent"] == "Experiment Agent" and f["is_code"]:
                f["baseline_id"] = baseline_ids.get(f["name"])
    order = {name: i for i, name in enumerate(PIPELINE_ORDER)}
    return sorted(groups.values(), key=lambda g: order.get(g["agent"], len(order)))


def _script_bytes(artifact_id: int) -> tuple[dict, bytes]:
    record, handle, _ = _open_artifact(artifact_id)
    if not _is_script_file(record):
        handle.close()
        abort(404, "Script not found.")
    with handle:
        return record, handle.read()


def run_in_progress(runner: PipelineRunner) -> bool:
    """True while a pipeline run is active, started here or outside the dashboard."""
    return _run_overview(runner)[2]


def create_app(runner: PipelineRunner | None = None, router: FreeLLMAPIManager | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["RUNNER"] = runner or PipelineRunner()
    app.config["FREELLMAPI"] = router or FreeLLMAPIManager()

    def current_runner() -> PipelineRunner:
        return app.config["RUNNER"]

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def _error(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    @app.get("/")
    def index():
        return send_file(STATIC_DIR / "index.html")

    # --- Running the pipeline ---

    @app.post("/api/run")
    def start_run():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400, "Send the topic as JSON.")
        runner = current_runner()
        _, run, active = _run_overview(runner)
        if active and not runner.is_running():
            abort(409, f"A pipeline run is already in progress (run {run['run_id']}, started outside the "
                       "dashboard). Wait for it to finish; two runs would overwrite each other's files.")
        router = app.config["FREELLMAPI"].status()
        if router["enabled"] and not router["healthy"] and not runner.is_running():
            # With LLM_PROVIDER=freellmapi every agent call would fail without the router.
            abort(409, "The FreeLLMAPI router is not running, so the agents cannot reach a model. "
                       + ("It is starting now; try again when it shows Running." if router["busy"]
                          else "Select Start router first."))
        try:
            runner.start(body.get("topic", ""), reset=bool(body.get("reset")))
        except RunError as e:
            abort(409 if "already" in str(e) else 400, str(e))
        return jsonify(runner.info())

    @app.post("/api/stop")
    def stop_run():
        try:
            current_runner().stop()
        except RunError:
            abort(409, "No run started from this dashboard is in progress. "
                       "A run started from the command line must be stopped there.")
        return jsonify(current_runner().info())

    @app.get("/api/status")
    def status():
        runner = current_runner()
        run = _resolve_run(runner)
        info = runner.info()
        is_runner_run = bool(run and run["is_current"] and info["running"])
        console = runner.console_text() if is_runner_run else ""
        progress = readers.agent_progress(run, PIPELINE_ORDER, console, run_active=bool(run and run["active"]))
        attempted = {s["agent"] for s in (run or {}).get("stages", [])}
        agents = [{**stage, "live": progress.get(stage["name"]), "has_output": stage["name"] in attempted}
                  for stage in PIPELINE]
        phase = ""
        if run and run["is_current"] and run.get("run_id"):
            state = readers.read_json(STATE_FILE, {})
            phase = state.get("phase", "") if isinstance(state, dict) else ""
        elif run:
            phase = run.get("final_phase") or ""
        return jsonify({"runner": info, "run": _public(run), "active": bool(run and run["active"]),
                        "agents": agents, "phase": phase})

    @app.get("/api/run-detail")
    def run_detail():
        run = _resolve_run(current_runner())
        attempts = _attempts(run)
        return jsonify({"run": _public(run), "summary": _summary(attempts), "attempts": attempts})

    @app.get("/api/runs")
    def runs():
        runner = current_runner()
        try:
            listed = readers.list_runs(RUN_DB)
        except Exception as e:  # the run database is optional, like Slack
            return jsonify({"error": f"Run database unavailable: {e}"}), 503
        current = _resolve_run(runner)
        if current and current.get("run_id") is None:
            listed.insert(0, {**_public(current), "parent_paper_title": None})
        for r in listed:
            if current and r["run_id"] == current.get("run_id"):
                r.update(status=current["status"], is_current=True)
            else:
                r["is_current"] = r["run_id"] is None
        return jsonify(listed)

    # --- FreeLLMAPI router (Docker) ---

    @app.get("/api/freellmapi")
    def freellmapi_status():
        return jsonify(app.config["FREELLMAPI"].status())

    @app.post("/api/freellmapi/start")
    def freellmapi_start():
        if not app.config["FREELLMAPI"].start_async():
            abort(409, "The router is already starting or stopping.")
        return jsonify(app.config["FREELLMAPI"].status())

    @app.post("/api/freellmapi/stop")
    def freellmapi_stop():
        if current_runner().is_running():
            abort(409, "A pipeline run is using the router. Stop the run first.")
        if not app.config["FREELLMAPI"].stop_async():
            abort(409, "The router is already starting or stopping.")
        return jsonify(app.config["FREELLMAPI"].status())

    # --- Console (kept per run for runs started from the dashboard) ---

    @app.get("/api/console")
    def console():
        runner = current_runner()
        run = _resolve_run(runner)
        offset = max(request.args.get("offset", default=0, type=int), 0)
        if run and run["is_current"] and runner.console_log and (run.get("run_id") in (None, runner.run_id)):
            return jsonify({**runner.read_console(offset), "available": True})
        name = readers.read_json(CONSOLE_INDEX, {}).get((run or {}).get("run_id") or "")
        if name and re.fullmatch(r"[0-9TZ]+\.log", name):
            return jsonify({**runner.read_console(offset, path=CONSOLE_DIR / name), "available": True})
        return jsonify({"text": "", "offset": 0, "reset": False, "available": False})

    # --- Master Agent decisions in this run ---

    @app.get("/api/activity")
    def activity():
        run = _resolve_run(current_runner())
        if not run:
            return jsonify([])
        start = _parse_time(run.get("started_at"))
        end = _parse_time(run.get("finished_at")) or datetime.now(timezone.utc)
        entries = readers.read_json(MASTER_LOG, [])
        if not isinstance(entries, list) or start is None:
            return jsonify([])
        # The log has no run id; a decision belongs to the run whose time span holds it.
        mine = [e for e in entries if (t := _parse_time(e.get("timestamp"))) and start <= t <= end]
        return jsonify(list(reversed(mine)))

    # --- Papers ---

    @app.get("/api/papers")
    def papers():
        papers_list, parent = _papers(_attempts(_resolve_run(current_runner())))
        out = []
        for i, p in enumerate(papers_list):
            pdf = _paper_pdf_path(p)
            out.append({
                "index": i, "is_parent": bool(parent) and p.get("title") == parent.get("title"),
                "title": p.get("title", ""), "authors": p.get("authors", []), "year": p.get("year"),
                "venue": p.get("venue", ""), "doi": p.get("doi", ""), "paper_url": p.get("paper_url", ""),
                "peer_reviewed": p.get("peer_reviewed"), "validation_status": p.get("validation_status", ""),
                "rejection_reason": p.get("rejection_reason", ""), "dataset": p.get("dataset", ""),
                "methodology": p.get("methodology", ""), "results_summary": p.get("results_summary", ""),
                "abstract": p.get("abstract", ""), "has_code": p.get("has_code"), "code_url": p.get("code_url", ""),
                "parent_paper_score": p.get("parent_paper_score"),
                "pdf_name": pdf.name if pdf.is_file() else "",
            })
        return jsonify(out)

    @app.post("/api/papers/<int:index>/pdf")
    def fetch_paper_pdf(index: int):
        """Download a paper's legal open-access PDF into dashboard_files/pdfs/."""
        papers_list, _ = _papers(_attempts(_resolve_run(current_runner())))
        if not 0 <= index < len(papers_list):
            abort(404, "Paper not found.")
        paper = papers_list[index]
        dest = _paper_pdf_path(paper)
        if dest.is_file():
            return jsonify({"pdf_name": dest.name})
        url = pdf_fetch.open_access_pdf_url(paper.get("doi", ""), paper.get("title", ""))
        if not url:
            abort(404, "No open-access PDF is listed for this paper. Open the publisher page instead.")
        try:
            content = pdf_fetch.download_pdf(url)
        except pdf_fetch.PdfError as e:
            abort(404, str(e))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return jsonify({"pdf_name": dest.name, "source_url": url})

    @app.get("/api/pdfs/<name>/view")
    def view_fetched_pdf(name: str):
        path = _fetched_pdf(name)
        return send_file(readers.open_shared(path), mimetype="application/pdf", download_name=path.name)

    @app.get("/api/pdfs/<name>/download")
    def download_fetched_pdf(name: str):
        path = _fetched_pdf(name)
        return send_file(readers.open_shared(path), as_attachment=True, download_name=path.name)

    # --- Documents: PDFs this run produced, and fetched copies of its papers ---

    @app.get("/api/documents")
    def documents():
        run = _resolve_run(current_runner())
        docs = []
        for a in _artifacts(run):
            if a["original_path"].lower().endswith(".pdf"):
                docs.append({"name": Path(a["original_path"]).name, "origin": f"Produced by the {a['agent']}",
                             "size_bytes": a["size_bytes"],
                             "view_url": f"/api/artifacts/{a['id']}/view",
                             "download_url": f"/api/artifacts/{a['id']}/download"})
        papers_list, parent = _papers(_attempts(run))
        for p in papers_list:
            pdf = _paper_pdf_path(p)
            if pdf.is_file():
                is_parent = bool(parent) and p.get("title") == parent.get("title")
                docs.append({"name": p.get("title") or pdf.name,
                             "origin": "Parent Paper, open-access copy" if is_parent else "Candidate paper, open-access copy",
                             "size_bytes": pdf.stat().st_size,
                             "view_url": f"/api/pdfs/{pdf.name}/view",
                             "download_url": f"/api/pdfs/{pdf.name}/download"})
        return jsonify(docs)

    # --- Datasets this run downloaded and produced ---

    @app.get("/api/datasets")
    def datasets():
        run = _resolve_run(current_runner())
        attempts = _attempts(run)
        meta = {}
        for d in (_output(attempts, "Data Agent") or {}).get("selected_datasets") or []:
            if d.get("local_csv_path"):
                meta[str(Path(d["local_csv_path"]))] = d
        cleaning = _output(attempts, "Cleaning Agent") or {}
        split_rows = cleaning.get("split_rows") or {}
        rows_by_kind = {"master_csv_path": cleaning.get("rows_total"), "train_csv_path": split_rows.get("train"),
                        "test_csv_path": split_rows.get("test"), "validation_csv_path": split_rows.get("validation")}
        out = []
        for a in _artifacts(run):
            path = Path(a["original_path"])
            if path.suffix.lower() not in TABLE_SUFFIXES or a["agent"] not in ("Data Agent", "Cleaning Agent"):
                continue
            d = meta.get(str(path), {})
            out.append({
                "id": a["id"], "name": path.name, "path": _display_path(path),
                "group": "processed" if a["agent"] == "Cleaning Agent" else "raw",
                "display_name": d.get("display_name", ""), "source": d.get("source", ""),
                "source_url": d.get("source_url", ""), "license": d.get("license", ""),
                "entry_count": d.get("entry_count") if a["agent"] == "Data Agent" else rows_by_kind.get(a["kind"]),
                "size_bytes": a["size_bytes"], "disk": _artifact_status(a), "stored": bool(a["content_stored"]),
            })
        return jsonify(out)

    # --- Any recorded file: preview, download, view ---

    @app.get("/api/artifacts/<int:artifact_id>/preview")
    def artifact_preview(artifact_id: int):
        record, handle, source = _open_artifact(artifact_id)
        suffix = Path(record["original_path"]).suffix.lower()
        if suffix not in TABLE_SUFFIXES:
            handle.close()
            abort(400, "Only CSV/TSV files can be previewed.")
        offset = max(request.args.get("offset", default=0, type=int), 0)
        limit = min(max(request.args.get("limit", default=100, type=int), 1), PREVIEW_MAX_ROWS)
        with io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if suffix == ".tsv" else ",")
            columns = next(reader, [])
            rows = list(islice(reader, offset, offset + limit + 1))
        return jsonify({"columns": columns, "rows": rows[:limit], "offset": offset,
                        "has_more": len(rows) > limit, "source": source})

    @app.get("/api/artifacts/<int:artifact_id>/download")
    def artifact_download(artifact_id: int):
        record, handle, _ = _open_artifact(artifact_id)
        return send_file(handle, as_attachment=True, download_name=Path(record["original_path"]).name)

    @app.get("/api/artifacts/<int:artifact_id>/view")
    def artifact_view(artifact_id: int):
        record, handle, _ = _open_artifact(artifact_id)
        if not record["original_path"].lower().endswith(".pdf"):
            handle.close()
            abort(400, "Only PDF files can be viewed inline.")
        return send_file(handle, mimetype="application/pdf", download_name=Path(record["original_path"]).name)

    # --- Scripts this run wrote ---

    @app.get("/api/scripts")
    def scripts():
        return jsonify({"groups": _script_groups(_resolve_run(current_runner()))})

    @app.get("/api/scripts/<int:artifact_id>")
    def script_text(artifact_id: int):
        record, content = _script_bytes(artifact_id)
        return jsonify({"id": artifact_id, "name": Path(record["original_path"]).name,
                        "language": "python" if record["original_path"].lower().endswith(".py") else "text",
                        "text": content[:SCRIPT_VIEW_MAX_BYTES].decode("utf-8", errors="replace"),
                        "truncated": len(content) > SCRIPT_VIEW_MAX_BYTES})

    @app.post("/api/scripts/<int:artifact_id>/open")
    def script_open(artifact_id: int):
        """Open the script in VS Code: the pipeline's own file if it is still the
        version this run recorded, else a copy of that recorded version."""
        record, content = _script_bytes(artifact_id)
        original = Path(record["original_path"])
        if _within_roots(original) and readers.file_matches(original, record["sha256"]) == "same":
            target, is_copy = original, False
        else:
            target = SCRIPT_COPY_DIR / record["run_id"] / f"{artifact_id}-{original.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            is_copy = True
        try:
            editor.open_in_vscode(target)
        except editor.EditorError as e:
            abort(409, str(e))
        return jsonify({"opened": _display_path(target), "is_copy": is_copy})

    return app
