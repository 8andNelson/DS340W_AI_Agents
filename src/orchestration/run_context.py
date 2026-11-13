"""
The pipeline run in progress, so log lines can name the run they belong to.

run_store.start_run sets it and run_store.finish_run clears it. A process
runs one pipeline run at a time (the dashboard refuses to start a second
run while one is active), so a module-level value is correct here --
unlike per-invocation context such as supervisor notes, which differ
between worker threads of the same run.

The values recorded at the start (git dirty state, Python version, OS)
describe the conditions the run actually ran under; run_metrics reads
them when it records the run's conditions.
"""
_run: dict = {}


def set_run(run_id: str | None, **info) -> None:
    _run.clear()
    if run_id:
        _run.update(info, run_id=run_id)


def clear(run_id: str | None = None) -> None:
    """Forget the current run; with run_id, only if it is still that run."""
    if run_id is None or _run.get("run_id") == run_id:
        _run.clear()


def current_run_id() -> str | None:
    return _run.get("run_id")


def info(run_id: str) -> dict:
    """What was recorded at the start of run_id, or {} for another run."""
    return dict(_run) if _run.get("run_id") == run_id else {}


def tag(entry: dict) -> dict:
    """Add the current run id to a log entry (in place) and return it."""
    if _run.get("run_id"):
        entry["run_id"] = _run["run_id"]
    return entry
