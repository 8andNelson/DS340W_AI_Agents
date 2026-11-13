"""
The pipeline run in progress, so log lines can name the run they belong to.

run_store.start_run sets it and run_store.finish_run clears it. A process
runs one pipeline run at a time (the dashboard refuses to start a second
run while one is active), so a module-level value is correct here --
unlike per-invocation context such as supervisor notes, which differ
between worker threads of the same run.

The values recorded at the start (git dirty state, Python version, OS,
the memory snapshot) describe the conditions the run actually ran under;
run_metrics reads them when it records the run's conditions.

The agent attempt in progress is different: first-pass stages run in
parallel worker threads, so it is a ContextVar, set by Master Agent's
_supervise around each attempt (invocation()). It carries the attempt's
memory use, which the memory recall code adds to as the agent works.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_run: dict = {}
_invocation: ContextVar = ContextVar("invocation", default=None)


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


@contextmanager
def invocation(invocation_id: str | None, agent: str, attempt: int):
    """While active, this thread (and workers started with
    llm.submit_with_context) works for one agent attempt. Yields the
    attempt's record, whose "memory" dict the recall code fills in."""
    record = {"invocation_id": invocation_id, "agent": agent, "attempt": attempt,
              "memory": {"recalls": 0, "recalls_with_cases": 0, "recall_errors": 0, "shown": [], "used": [],
                         "invalid_refs": [], "requests": 0, "prompt_chars": 0}}
    token = _invocation.set(record)
    try:
        yield record
    finally:
        _invocation.reset(token)


def current_invocation() -> dict | None:
    return _invocation.get()


def start_invocation(agent: str, attempt: int) -> dict:
    """Make this thread's current attempt a new one (Master Agent's
    _supervise, at the start of each attempt; the attempt's diagnosis after
    a failure still belongs to it). end_invocation() clears it."""
    record = {"invocation_id": None, "agent": agent, "attempt": attempt,
              "memory": {"recalls": 0, "recalls_with_cases": 0, "recall_errors": 0, "shown": [], "used": [],
                         "invalid_refs": [], "requests": 0, "prompt_chars": 0}}
    _invocation.set(record)
    return record


def end_invocation() -> None:
    _invocation.set(None)


def tag(entry: dict) -> dict:
    """Add the current run id to a log entry (in place) and return it."""
    if _run.get("run_id"):
        entry["run_id"] = _run["run_id"]
    return entry
