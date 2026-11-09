import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.dashboard import editor, pdf_fetch, readers, server
from src.dashboard.runner import PipelineRunner, RunError, _complete_utf8, validate_topic
from src.orchestration import run_store, state_manager

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point every path the dashboard reads at a temp repo."""
    monkeypatch.setattr(server, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(server, "FILE_ROOTS", [tmp_path / d for d in ("data", "replication", "experiments", "dashboard_files")])
    monkeypatch.setattr(server, "PDF_DIR", tmp_path / "dashboard_files" / "pdfs")
    monkeypatch.setattr(server, "SCRIPT_COPY_DIR", tmp_path / "dashboard_files" / "scripts")
    monkeypatch.setattr(server, "CONSOLE_DIR", tmp_path / "dashboard_files" / "consoles")
    monkeypatch.setattr(server, "CONSOLE_INDEX", tmp_path / "dashboard_files" / "consoles" / "index.json")
    monkeypatch.setattr(server, "STATE_FILE", state_manager.STATE_FILE)  # conftest's temp state file
    monkeypatch.setattr(server, "MASTER_LOG", tmp_path / "logs" / "master_agent_log.json")
    monkeypatch.setattr(server, "RUN_DB", run_store.DB_FILE)  # conftest's temp run database
    # The real .env may select FreeLLMAPI; these tests are not about the router
    # (tests/test_dashboard_freellmapi.py is).
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    return tmp_path


def _stand_in(tmp_path, body: str, name: str = "stand_in_run.py") -> list:
    """A command that stands in for run.py, so no test starts the real pipeline."""
    script = tmp_path / name
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


@pytest.fixture
def runner(repo):
    r = PipelineRunner(command_prefix=_stand_in(repo, "import time\ntime.sleep(30)\n"),
                       console_dir=repo / "dashboard_files" / "consoles")
    yield r
    if r.is_running():
        r.stop()


@pytest.fixture
def client(runner):
    return server.create_app(runner).test_client()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _wait_until_done(runner, seconds=10):
    deadline = time.time() + seconds
    while runner.is_running() and time.time() < deadline:
        time.sleep(0.05)


def _set_times(run_id, started, finished=None, status=None):
    """run_store stamps times to the second; tests set them so runs order clearly."""
    with sqlite3.connect(run_store.DB_FILE) as conn:
        conn.execute("UPDATE runs SET started_at = ?, finished_at = ?, status = COALESCE(?, status) WHERE run_id = ?",
                     (started, finished, status, run_id))


def _paper(title, status="VERIFIED"):
    return {"title": title, "authors": ["A. Author"], "year": 2024, "venue": "Journal", "validation_status": status}


def _record_full_run(repo, tag, started, finished):
    """A finished run recorded the way the pipeline records it (run_store.record_stage
    stores each output and every file its *_path keys point to)."""
    raw = repo / "data" / "raw" / f"{tag}.csv"
    master = repo / "data" / "processed" / "master_dataset.csv"
    _write(raw, f"amount,fraud\n1,0\n2,1\n")
    _write(master, f"amount,fraud,source\n1,0,{tag}\n2,1,{tag}\n")
    run_id = run_store.start_run(f"topic {tag}")
    papers = [_paper(f"Paper {tag} one"), _paper(f"Paper {tag} two", "REJECTED")]
    run_store.record_stage(run_id, "Research Agent", 1, "Approved", papers, "found 2")
    run_store.record_stage(run_id, "Validation Agent", 1, "Approved",
                           {"all": papers, "recommended_parent_paper": papers[0]}, "parent chosen")
    run_store.record_stage(run_id, "Data Agent", 1, "Approved",
                           {"selected_datasets": [{"local_csv_path": str(raw), "display_name": f"Fraud {tag}",
                                                   "source": "Kaggle", "entry_count": 2}],
                            "total_entries": 2}, "")
    run_store.record_stage(run_id, "Cleaning Agent", 1, "Approved",
                           {"master_csv_path": str(master), "rows_total": 2, "target_met": False}, "")
    run_store.finish_run(run_id, "success", "DATA_CLEANING", parent_paper=papers[0])
    _set_times(run_id, started, finished)
    return run_id


@pytest.fixture
def two_runs(repo):
    """Run A finished yesterday with datasets; run B (current) has only papers."""
    a = _record_full_run(repo, "A", "2026-01-01T10:00:00+00:00", "2026-01-01T11:00:00+00:00")
    b = run_store.start_run("topic B")
    run_store.record_stage(b, "Research Agent", 1, "Approved", [_paper("Paper B one")], "found 1")
    _set_times(b, "2026-01-02T10:00:00+00:00", None, "running")
    return {"a": a, "b": b}


# --- The dashboard and the pipeline are separate programs ---

def test_dashboard_imports_no_pipeline_module():
    code = ("import sys, src.dashboard.server; "
            "print([m for m in sys.modules if m.startswith('src.') and not m.startswith('src.dashboard')])")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_pipeline_never_imports_the_dashboard():
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "dashboard" in path.parts:
            continue
        assert "src.dashboard" not in path.read_text(encoding="utf-8"), path
    assert "dashboard" not in (REPO_ROOT / "run.py").read_text(encoding="utf-8")


def test_shared_open_does_not_block_pipeline_writes_or_deletes(tmp_path):
    # The pipeline rewrites its files in place (open "w") and deletes whole
    # folders (Replication Agent's shutil.rmtree before a fresh clone). On
    # Windows a normal open() by the dashboard would make the delete fail.
    victim = tmp_path / "replication" / "repo" / "data.csv"
    _write(victim, "a\n1\n")
    state = tmp_path / "project_state.json"
    _write(state, "{}")

    with readers.open_shared(victim) as f, readers.open_shared(state):
        assert f.read(1) == b"a"
        state.write_text('{"new": 1}')
        shutil.rmtree(tmp_path / "replication")
    assert not victim.exists()
    assert json.loads(state.read_text()) == {"new": 1}


@pytest.mark.skipif(os.name != "nt", reason="POSIX never blocks deletes of open files")
def test_plain_open_would_block_the_pipeline(tmp_path):
    """Shows why open_shared exists: the test above fails with a plain open()."""
    victim = tmp_path / "replication" / "data.csv"
    _write(victim, "a\n")
    with open(victim, "rb"):
        with pytest.raises(PermissionError):
            shutil.rmtree(tmp_path / "replication")


def test_run_database_reads_never_write(tmp_path):
    missing = tmp_path / "database" / "none.db"
    assert readers.list_runs(missing) == []
    assert readers.latest_run(missing) is None
    assert not missing.exists()  # run_store._connect would have created it

    run_id = run_store.start_run("topic")
    before = run_store.DB_FILE.read_bytes()
    assert readers.latest_run(run_store.DB_FILE)["run_id"] == run_id
    assert readers.stage_attempts(run_store.DB_FILE, run_id) == []
    assert run_store.DB_FILE.read_bytes() == before
    # The pipeline can still write while the dashboard keeps reading.
    run_store.record_stage(run_id, "Intake Agent", 1, "Approved", {"ok": True}, "report")
    assert readers.latest_run(run_store.DB_FILE)["stages"][0]["agent"] == "Intake Agent"


def test_json_read_during_a_pipeline_write_uses_last_good_copy(tmp_path):
    path = tmp_path / "project_state.json"
    _write(path, '{"phase": "DATA_DISCOVERY"}')
    assert readers.read_json(path, {}) == {"phase": "DATA_DISCOVERY"}
    _write(path, '{"phase": "DATA_CLEA')  # half-written
    assert readers.read_json(path, {}) == {"phase": "DATA_DISCOVERY"}


def test_run_keeps_going_after_the_dashboard_exits(tmp_path):
    marker = tmp_path / "pipeline_finished.txt"
    child = _stand_in(tmp_path, f"""
        import time
        time.sleep(1.5)
        open({str(marker)!r}, "w").write("done")
    """)
    # A short-lived "dashboard" that starts the run and exits at once.
    dashboard = textwrap.dedent(f"""
        import os
        from pathlib import Path
        from src.dashboard.runner import PipelineRunner
        runner = PipelineRunner(console_log=Path({str(tmp_path / 'console.log')!r}), command_prefix={child!r})
        runner.start("topic")
        os._exit(0)
    """)
    subprocess.run([sys.executable, "-c", dashboard], cwd=REPO_ROOT, check=True, timeout=30)
    assert not marker.exists()  # the dashboard is gone, the run is still going

    deadline = time.time() + 15
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert marker.read_text() == "done"


def test_dashboard_refuses_a_second_run_while_a_cli_run_is_active(client):
    run_store.start_run("started with python run.py")  # recorded as "running"

    status = client.get("/api/status").get_json()
    assert status["active"] is True
    assert "outside the dashboard" in status["run"]["note"]

    response = client.post("/api/run", json={"topic": "another topic"})
    assert response.status_code == 409
    assert "already in progress" in response.get_json()["error"]
    assert client.post("/api/stop").status_code == 409  # it is not the dashboard's to stop


def test_stale_running_row_does_not_block_new_runs(client, monkeypatch):
    run_store.start_run("killed long ago")
    monkeypatch.setattr(readers, "hours_since", lambda moment: 48.0)

    status = client.get("/api/status").get_json()
    assert status["active"] is False
    assert status["run"]["status"] == "unknown"
    assert client.post("/api/run", json={"topic": "new topic"}).status_code == 200


# --- Every view shows one run ---

def test_current_run_shows_only_its_own_records(client, two_runs):
    papers = client.get("/api/papers").get_json()
    assert [p["title"] for p in papers] == ["Paper B one"]
    assert client.get("/api/datasets").get_json() == []  # run A's files on disk are not shown
    status = client.get("/api/status").get_json()
    assert status["run"]["run_id"] == two_runs["b"]
    assert status["run"]["is_current"] is True


def test_past_run_shows_its_own_papers_and_datasets(client, two_runs):
    q = f"?run={two_runs['a']}"
    papers = client.get(f"/api/papers{q}").get_json()
    assert [p["title"] for p in papers] == ["Paper A one", "Paper A two"]
    assert papers[0]["is_parent"] is True

    datasets = {d["name"]: d for d in client.get(f"/api/datasets{q}").get_json()}
    assert datasets["A.csv"]["display_name"] == "Fraud A"
    assert datasets["A.csv"]["group"] == "raw"
    assert datasets["master_dataset.csv"]["entry_count"] == 2
    assert datasets["master_dataset.csv"]["group"] == "processed"

    status = client.get(f"/api/status{q}").get_json()
    assert status["run"]["is_current"] is False
    assert status["phase"] == "DATA_CLEANING"
    assert {a["name"]: (a["live"] or {}).get("status") for a in status["agents"]}["Cleaning Agent"] == "approved"


def test_run_detail_has_attempts_and_summary(client, two_runs):
    body = client.get(f"/api/run-detail?run={two_runs['a']}").get_json()
    assert [a["agent"] for a in body["attempts"]] == ["Research Agent", "Validation Agent", "Data Agent", "Cleaning Agent"]
    assert body["attempts"][0]["report_text"] == "found 2"
    assert body["summary"]["papers"] == 2
    assert body["summary"]["papers_verified"] == 1
    assert body["summary"]["parent_paper"] == "Paper A one"
    assert body["summary"]["master_rows"] == 2


def test_overwritten_dataset_is_served_from_the_run_record(client, two_runs, repo):
    # A later run rewrites the shared master CSV on disk.
    _write(repo / "data" / "processed" / "master_dataset.csv", "other,columns\n9,9\n")
    datasets = {d["name"]: d for d in client.get(f"/api/datasets?run={two_runs['a']}").get_json()}
    master = datasets["master_dataset.csv"]
    assert master["disk"] == "changed"

    preview = client.get(f"/api/artifacts/{master['id']}/preview?limit=5").get_json()
    assert preview["columns"] == ["amount", "fraud", "source"]
    assert preview["rows"] == [["1", "0", "A"], ["2", "1", "A"]]
    assert preview["source"] == "database"
    download = client.get(f"/api/artifacts/{master['id']}/download")
    assert download.data == b"amount,fraud,source\n1,0,A\n2,1,A\n"
    assert "attachment" in download.headers["Content-Disposition"]


def test_unchanged_dataset_is_read_from_disk(client, two_runs):
    raw = {d["name"]: d for d in client.get(f"/api/datasets?run={two_runs['a']}").get_json()}["A.csv"]
    preview = client.get(f"/api/artifacts/{raw['id']}/preview?limit=1").get_json()
    assert preview["source"] == "disk"
    assert preview["rows"] == [["1", "0"]]
    assert preview["has_more"] is True


def test_activity_shows_only_decisions_made_during_the_run(client, two_runs, repo):
    _write(repo / "logs" / "master_agent_log.json", json.dumps([
        {"action": "Approval", "target": "Research Agent", "timestamp": "2026-01-01T10:30:00+00:00", "n": "A"},
        {"action": "Approval", "target": "Research Agent", "timestamp": "2026-01-02T10:30:00+00:00", "n": "B"},
    ]))
    assert [e["n"] for e in client.get(f"/api/activity?run={two_runs['a']}").get_json()] == ["A"]
    assert [e["n"] for e in client.get("/api/activity").get_json()] == ["B"]


def test_documents_list_fetched_pdfs_only_for_the_runs_papers(client, two_runs, repo):
    _write(repo / "dashboard_files" / "pdfs" / "paper-a-one.pdf", "%PDF-1.4 a")
    docs_a = client.get(f"/api/documents?run={two_runs['a']}").get_json()
    assert [d["name"] for d in docs_a] == ["Paper A one"]
    assert docs_a[0]["origin"] == "Parent Paper, open-access copy"
    assert client.get("/api/documents").get_json() == []  # run B has no such paper

    view = client.get(docs_a[0]["view_url"])
    assert view.mimetype == "application/pdf"
    view.close()
    assert client.get("/api/pdfs/..%2F..%2Fsecret.pdf/view").status_code in (400, 404)


def test_run_history_marks_the_current_run(client, two_runs):
    runs = client.get("/api/runs").get_json()
    assert [(r["run_id"], r["is_current"]) for r in runs] == [(two_runs["b"], True), (two_runs["a"], False)]


def test_unknown_run_is_not_found(client, two_runs):
    assert client.get("/api/papers?run=no-such-run").status_code == 404


def test_a_new_run_starts_with_an_empty_view(client, two_runs, runner):
    run_store.finish_run(two_runs["b"], "success", "LITERATURE_SEARCH")
    _set_times(two_runs["b"], "2026-01-02T10:00:00+00:00", "2026-01-02T11:00:00+00:00", "success")
    runner.start("brand new topic")  # the pipeline has not opened its run record yet

    status = client.get("/api/status").get_json()
    assert status["run"]["run_id"] is None
    assert status["run"]["status"] == "starting"
    assert status["run"]["topic"] == "brand new topic"
    assert client.get("/api/papers").get_json() == []
    assert client.get("/api/datasets").get_json() == []
    assert client.get("/api/runs").get_json()[0]["status"] == "starting"


def test_console_of_a_dashboard_run_is_kept_for_run_history(repo):
    console_dir = repo / "dashboard_files" / "consoles"
    runner = PipelineRunner(command_prefix=_stand_in(repo, "print('hello from the run')\n", "say.py"),
                            console_dir=console_dir)
    client = server.create_app(runner).test_client()
    runner.start("topic")
    run_id = run_store.start_run("topic")  # what the pipeline does first
    _wait_until_done(runner)
    client.get("/api/status")  # links the console file to the run record

    # A second dashboard run gets its own console file.
    runner.command_prefix = _stand_in(repo, "print('second run')\n", "say2.py")
    time.sleep(1.1)  # console files are named by the second
    runner.start("next topic")
    _wait_until_done(runner)

    old = client.get(f"/api/console?run={run_id}").get_json()
    assert old["available"] is True
    assert "hello from the run" in old["text"]
    assert len(list(console_dir.glob("*.log"))) == 2


def test_console_is_unavailable_for_runs_started_elsewhere(client, two_runs):
    body = client.get(f"/api/console?run={two_runs['a']}").get_json()
    assert body["available"] is False


# --- Live agent progress, from the pipeline's own records ---

def _run(stages):
    return {"run_id": "r", "started_at": "2026-01-01T00:00:00+00:00", "status": "running",
            "stages": [{"agent": a, "attempt": n, "decision": d, "report_text": f"{a} report",
                        "recorded_at": f"2026-01-01T00:0{i + 1}:00+00:00"} for i, (a, n, d) in enumerate(stages)]}


ORDER = ["Intake Agent", "Research Agent", "Validation Agent"]


def test_progress_infers_next_agent_from_run_database():
    progress = readers.agent_progress(_run([("Intake Agent", 1, "Approved")]), ORDER, run_active=True)
    assert progress["Intake Agent"]["status"] == "approved"
    assert progress["Intake Agent"]["summary"] == "Intake Agent report"
    assert progress["Research Agent"]["status"] == "running"
    assert "Validation Agent" not in progress


def test_progress_after_retry_shows_second_attempt_running():
    progress = readers.agent_progress(_run([("Intake Agent", 1, "Retrying")]), ORDER, run_active=True)
    assert progress["Intake Agent"]["status"] == "running"
    assert progress["Intake Agent"]["attempt"] == 2


def test_progress_prefers_the_console_line():
    console = "[Master Agent] Running Data Agent (attempt 1/2)...\n"  # e.g. a Slack-relayed re-run
    progress = readers.agent_progress(_run([("Intake Agent", 1, "Approved")]), ORDER, console, run_active=True)
    assert progress["Data Agent"]["status"] == "running"
    assert progress["Data Agent"]["max_attempts"] == 2
    assert "Research Agent" not in progress


def test_progress_of_finished_run_has_no_running_agent():
    progress = readers.agent_progress(_run([("Intake Agent", 1, "Rejected")]), ORDER, run_active=False)
    assert list(progress) == ["Intake Agent"]
    assert progress["Intake Agent"]["status"] == "rejected"


# --- Paper PDFs (saved in the dashboard's own folder) ---

class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.content


def test_fetch_pdf_uses_the_viewed_runs_papers(client, two_runs, repo, monkeypatch):
    monkeypatch.setattr(pdf_fetch, "open_access_pdf_url", lambda doi, title: "https://oa.example/p.pdf")
    monkeypatch.setattr(pdf_fetch.requests, "get", lambda *a, **k: _FakeResponse(b"%PDF-1.7 body"))

    body = client.post(f"/api/papers/1/pdf?run={two_runs['a']}").get_json()
    assert body["pdf_name"] == "paper-a-two.pdf"
    assert (repo / "dashboard_files" / "pdfs" / "paper-a-two.pdf").read_bytes() == b"%PDF-1.7 body"
    assert not (repo / "papers").exists()  # the pipeline's folder is untouched
    assert client.get(f"/api/papers?run={two_runs['a']}").get_json()[1]["pdf_name"] == "paper-a-two.pdf"


def test_fetch_pdf_rejects_html_landing_page(client, two_runs, repo, monkeypatch):
    monkeypatch.setattr(pdf_fetch, "open_access_pdf_url", lambda doi, title: "https://oa.example/p")
    monkeypatch.setattr(pdf_fetch.requests, "get", lambda *a, **k: _FakeResponse(b"<html>login</html>"))

    assert client.post("/api/papers/0/pdf").status_code == 404
    assert not list(repo.rglob("*.pdf"))


def test_fetch_pdf_without_open_access_link(client, two_runs, monkeypatch):
    monkeypatch.setattr(pdf_fetch, "open_access_pdf_url", lambda doi, title: "")
    response = client.post("/api/papers/0/pdf")
    assert response.status_code == 404
    assert "open-access" in response.get_json()["error"]


def test_title_match_must_be_same_paper(monkeypatch):
    class R:
        status_code = 200

        def json(self):
            return {"results": [{"title": "A Different Paper", "best_oa_location": {"pdf_url": "u"}}]}

    monkeypatch.setattr(pdf_fetch.requests, "get", lambda *a, **k: R())
    assert pdf_fetch.open_access_pdf_url(title="My Paper") == ""


def test_reference_list_pdfs_are_skipped():
    work = {"best_oa_location": {"pdf_url": "https://www.nature.com/articles/x_reference.pdf"},
            "locations": [{"pdf_url": "https://repo.example/x.pdf"}]}
    assert pdf_fetch._pdf_url(work) == "https://repo.example/x.pdf"


# --- Runner ---

def test_validate_topic():
    assert validate_topic("  fraud   detection ") == "fraud detection"
    for bad in ("", "   ", "--reset", "x" * 501):
        with pytest.raises(RunError):
            validate_topic(bad)


def test_runner_passes_topic_and_reset_like_the_cli(tmp_path):
    prefix = _stand_in(tmp_path, """
        import sys
        print("topic=" + sys.argv[-1])
        print("reset" if "--reset" in sys.argv else "keep")
    """)
    runner = PipelineRunner(console_log=tmp_path / "console.log", command_prefix=prefix)
    runner.start("fraud detection", reset=True)
    _wait_until_done(runner)

    assert runner.info()["exit_code"] == 0
    assert runner.read_console()["text"].split() == ["topic=fraud", "detection", "reset"]


def test_runner_refuses_second_concurrent_run_and_can_stop(tmp_path):
    runner = PipelineRunner(console_log=tmp_path / "console.log",
                            command_prefix=_stand_in(tmp_path, "import time\ntime.sleep(30)\n"))
    runner.start("first")
    try:
        with pytest.raises(RunError, match="already"):
            runner.start("second")
    finally:
        runner.stop()
    assert runner.info()["running"] is False
    assert runner.info()["stopped_by_user"] is True


def test_console_reads_stop_at_whole_characters():
    euro = "€".encode()
    assert _complete_utf8(b"a" + euro) == b"a" + euro
    assert _complete_utf8(b"a" + euro[:2]) == b"a"


def test_run_endpoint_requires_json(client):
    assert client.post("/api/run", data="topic=x").status_code == 400
    assert client.post("/api/run", json={"topic": ""}).status_code == 400


# --- Scripts (from the run database's artifact records) ---

BASELINE = "import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()\n"
EXPERIMENT = "import pandas as pd\nfrom sklearn.ensemble import HistGradientBoostingClassifier\nclf = HistGradientBoostingClassifier()\n"


@pytest.fixture
def scripted_run(repo):
    rep_dir = repo / "replication" / "paper"
    exp_dir = repo / "experiments" / "paper" / "exp_001"
    _write(rep_dir / "reconstructed" / "train.py", BASELINE)
    _write(rep_dir / "reconstructed" / "attempts" / "attempt_1.py", "broken = (\n")
    _write(rep_dir / "run_stdout.txt", "accuracy: 0.9\n")
    _write(exp_dir / "train.py", EXPERIMENT)
    run_id = run_store.start_run("fraud detection")
    run_store.record_stage(run_id, "Replication Agent", 1, "Approved",
                           {"replication_dir": str(rep_dir), "entry_point": "train.py", "source": "reconstructed",
                            "generated_by": ["model-a"]}, "")
    run_store.record_stage(run_id, "Experiment Agent", 1, "Approved",
                           {"experiment_dir": str(exp_dir), "category": "alternative_model",
                            "hypothesis": "Boosting helps.", "generated_by": "model-b"}, "")
    return {"run_id": run_id, "rep_dir": rep_dir, "exp_dir": exp_dir}


def _scripts_by_name(client, run_id=None):
    body = client.get(f"/api/scripts{f'?run={run_id}' if run_id else ''}").get_json()
    return body, {(g["agent"], f["name"]): f for g in body["groups"] for f in g["files"]}


def test_scripts_are_grouped_by_agent_with_roles_and_provenance(client, scripted_run):
    body, files = _scripts_by_name(client)
    assert [g["agent"] for g in body["groups"]] == ["Replication Agent", "Experiment Agent"]
    assert body["groups"][0]["context"]["generated_by"] == ["model-a"]
    assert body["groups"][1]["context"]["hypothesis"] == "Boosting helps."

    baseline = files[("Replication Agent", "train.py")]
    assert baseline["role"] == "Baseline script (the version that ran)"
    assert files[("Replication Agent", "attempt_1.py")]["role"].startswith("Generated version 1")
    assert files[("Replication Agent", "run_stdout.txt")]["role"] == "Run output"
    experiment = files[("Experiment Agent", "train.py")]
    assert experiment["baseline_id"] == baseline["id"]  # enables "Changes from baseline"
    assert experiment["disk"] == "same"


def test_scripts_of_one_run_are_not_shown_for_another(client, scripted_run):
    later = run_store.start_run("later topic")
    _set_times(later, "2099-01-01T00:00:00+00:00")
    assert _scripts_by_name(client)[0]["groups"] == []
    assert _scripts_by_name(client, scripted_run["run_id"])[0]["groups"]


def test_script_text_and_download_come_from_the_recorded_version(client, scripted_run):
    _, files = _scripts_by_name(client)
    script_id = files[("Experiment Agent", "train.py")]["id"]
    # A later run overwrites the file; the dashboard still shows what this run wrote.
    _write(scripted_run["exp_dir"] / "train.py", "print('a later run')\n")

    body = client.get(f"/api/scripts/{script_id}").get_json()
    assert body["text"] == EXPERIMENT
    assert body["language"] == "python"
    response = client.get(f"/api/artifacts/{script_id}/download")
    assert response.data == EXPERIMENT.encode()
    assert "attachment" in response.headers["Content-Disposition"]
    assert _scripts_by_name(client)[1][("Experiment Agent", "train.py")]["disk"] == "changed"


def test_open_in_vscode_uses_pipeline_file_when_unchanged(client, scripted_run, monkeypatch):
    opened = []
    monkeypatch.setattr(editor, "open_in_vscode", opened.append)
    _, files = _scripts_by_name(client)

    body = client.post(f"/api/scripts/{files[('Replication Agent', 'train.py')]['id']}/open").get_json()
    assert body["is_copy"] is False
    assert opened == [scripted_run["rep_dir"] / "reconstructed" / "train.py"]


def test_open_in_vscode_uses_a_copy_when_the_file_changed(client, scripted_run, repo, monkeypatch):
    opened = []
    monkeypatch.setattr(editor, "open_in_vscode", opened.append)
    _, files = _scripts_by_name(client)
    script_id = files[("Experiment Agent", "train.py")]["id"]
    (scripted_run["exp_dir"] / "train.py").unlink()

    body = client.post(f"/api/scripts/{script_id}/open").get_json()
    assert body["is_copy"] is True
    assert opened[0].is_relative_to(repo / "dashboard_files" / "scripts")  # never written into the pipeline's folders
    assert opened[0].read_text() == EXPERIMENT


def test_open_in_vscode_reports_a_missing_editor(client, scripted_run, monkeypatch):
    def missing(path):
        raise editor.EditorError("VS Code was not found.")
    monkeypatch.setattr(editor, "open_in_vscode", missing)
    _, files = _scripts_by_name(client)
    response = client.post(f"/api/scripts/{files[('Replication Agent', 'train.py')]['id']}/open")
    assert response.status_code == 409
    assert "not found" in response.get_json()["error"]


def test_only_script_artifacts_are_served_as_scripts(client, two_runs):
    master = {d["name"]: d for d in client.get(f"/api/datasets?run={two_runs['a']}").get_json()}["master_dataset.csv"]
    assert client.get(f"/api/scripts/{master['id']}").status_code == 404
    assert client.post(f"/api/scripts/{master['id']}/open").status_code == 404


@pytest.mark.skipif(os.name != "nt", reason="the batch-file workaround is Windows-only")
def test_vscode_is_started_as_code_exe_not_the_batch_file(tmp_path, monkeypatch):
    install = tmp_path / "Microsoft VS Code"
    _write(install / "bin" / "code.cmd", "@echo off")
    _write(install / "Code.exe", "")
    monkeypatch.setattr(editor.shutil, "which", lambda name: str(install / "bin" / "code.cmd"))
    assert editor.find_vscode() == (install / "Code.exe").resolve()
