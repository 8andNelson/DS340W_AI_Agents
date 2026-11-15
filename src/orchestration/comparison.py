"""
Controlled comparison: does memory help? (milestone M8;
docs/CLAUDE_BACKEND_PLAN.md, section 11.8.)

  plan     writes the test plan BEFORE the first run, once: the topics, the
           conditions, the repeats, the run order (conditions alternate, so
           a change over time -- a free model that stops -- does not affect
           only one group), the primary metrics, the exclusion rules, the
           code commit and LLM provider every run must use, the frozen
           memory snapshot and the held-out topics. The file is never
           changed afterwards; progress goes to a separate status file.
  run      runs the next planned run (or all): `python run.py "<topic>"`
           with the run's condition in its environment (MEMORY_MODE,
           MEMORY_WARNINGS, MEMORY_COMPARISON_GROUP). Refused on another
           code commit, another LLM provider, or with the LLM router down.
  status   what has run, and what is next.
  report   the comparison of the group's runs only, grouped by ASSIGNED
           condition (never moved after a result), with each run's actual
           memory exposure, n, spread, the test when each group has at
           least 3 runs, and the conditions that differ. Failed runs stay in.

Conditions: "off" (no cross-run memory: the control), "read" (memory,
bank frozen), and optionally "read_no_warnings" (good cases only).

While a test group is set (MEMORY_COMPARISON_GROUP), every run of the
group recalls only memories created before the plan's snapshot, and never
one made from a run on a held-out topic (the test topics themselves), so
no run can get its answer from an earlier run of the same task, and test
runs cannot feed each other (plan 11.8).

The plan also freezes the bank itself (review finding 1): make_plan writes
<name>.bank.json once -- every memory created before the snapshot, with its
state, quality and outcome counts at that moment -- and records its sha256
in the plan. A test run ranks by these frozen values (recall.frozen_bank),
not by the live bank, which other runs keep changing; and the "read"
condition records no outcome at all (learning.bank_learns).

CLI:
  python -m src.orchestration.comparison plan <name> --topic "<t>" [--topic ...]
        [--conditions off read] [--repeats 3] [--no-warnings-condition]
  python -m src.orchestration.comparison run <name> [--all] [--allow-dirty]
  python -m src.orchestration.comparison status <name>
  python -m src.orchestration.comparison report <name>
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.orchestration import run_store

COMPARISONS_DIR = run_store.REPO_ROOT / "reports" / "comparisons"
CONDITIONS = {
    "off": {"MEMORY_MODE": "off", "MEMORY_WARNINGS": "true"},
    "read": {"MEMORY_MODE": "read", "MEMORY_WARNINGS": "true"},
    "read_no_warnings": {"MEMORY_MODE": "read", "MEMORY_WARNINGS": "false"},
}
PRIMARY_METRICS = ("final_phase_reached", "reached_measurable_gap", "final_gap_points", "leakage_findings_final",
                   "retries", "exceptions", "first_attempt_approval_rate", "run_seconds", "llm_calls",
                   "memories_used", "memories_helped")
EXCLUSION_RULES = [
    "No run is excluded because of its result; halted and crashed runs stay in every count.",
    "A run is invalid -- and run again -- only when its recorded memory mode or comparison group differs from "
    "its assigned condition, or it ran on another code commit or LLM provider than the plan (the runner "
    "refuses these before starting).",
]
RUN_TIMEOUT_SECONDS = 4 * 3600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan_path(name: str) -> Path:
    return COMPARISONS_DIR / f"{name}.json"


def status_path(name: str) -> Path:
    return COMPARISONS_DIR / f"{name}.status.json"


def bank_path(name: str) -> Path:
    return COMPARISONS_DIR / f"{name}.bank.json"


# What a frozen bank keeps of each memory: everything ranking reads that can
# change after the memory is written.
BANK_FIELDS = ("state", "quality", "quality_reasons", "times_helped", "times_hurt", "superseded_by")


def write_bank(name: str, snapshot: str) -> dict:
    """Write the frozen bank of a plan (once). Returns {"file", "sha256", "count"}."""
    import hashlib
    from src import memory
    path = bank_path(name)
    if path.exists():
        raise FileExistsError(f"{path} exists: a frozen bank is written once and never changed")
    rows = [m for m in memory.find_memories(states=None, limit=10_000_000) if (m.get("created_at") or "") <= snapshot]
    bank = {"snapshot": snapshot, "memories": {m["id"]: {k: m.get(k) for k in BANK_FIELDS} for m in rows}}
    data = json.dumps(bank, indent=0, sort_keys=True, default=str).encode("utf-8")
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)  # bytes: write_text would turn \n into \r\n on Windows and change the hash
    return {"file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "count": len(rows)}


def read_bank(path: str) -> dict:
    """{memory id: frozen fields} of a frozen bank file."""
    return json.loads(Path(path).read_text(encoding="utf-8")).get("memories") or {}


def bank_intact(plan: dict) -> bool:
    """True when the plan's frozen bank file still has the sha256 the plan recorded."""
    import hashlib
    path = plan.get("memory_bank_file")
    if not path or not Path(path).exists():
        return False
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() == plan.get("memory_bank_sha256")


def load_plan(name: str) -> dict | None:
    path = plan_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def group_tag(name: str) -> str:
    return f"test:{name}"


def active_plan() -> dict | None:
    """The plan of the test this run belongs to (MEMORY_COMPARISON_GROUP), or None."""
    from src.config import MEMORY_COMPARISON_GROUP
    return load_plan(MEMORY_COMPARISON_GROUP) if MEMORY_COMPARISON_GROUP else None


def run_order(topics: list, conditions: list, repeats: int) -> list:
    """[(topic, condition)]: every topic under every condition, repeats times,
    with the conditions' order alternating between repeats and topics."""
    order = []
    for r in range(repeats):
        for t, topic in enumerate(topics):
            conds = conditions if (r + t) % 2 == 0 else list(reversed(conditions))
            order.extend((topic, c) for c in conds)
    return order


def make_plan(name: str, topics: list, conditions: list = ("off", "read"), repeats: int = 3,
              notes: str = "") -> dict:
    """Write the test plan, once. Raises if it already exists (a plan is
    never changed after it is written)."""
    from src import config
    from src.orchestration.run_metrics import topic_key
    if plan_path(name).exists():
        raise FileExistsError(f"{plan_path(name)} exists: a test plan is written once and never changed")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown or len(conditions) < 2 or not topics or repeats < 1:
        raise ValueError(f"need >= 2 known conditions ({', '.join(CONDITIONS)}), topics and repeats >= 1")
    snapshot = _now()
    bank = write_bank(name, snapshot)
    plan = {
        "name": name, "comparison_group": group_tag(name), "created_at": snapshot,
        "memory_snapshot": snapshot,
        "memory_bank_file": bank["file"], "memory_bank_sha256": bank["sha256"], "memory_bank_count": bank["count"],
        "held_out_topics": sorted({topic_key(t) for t in topics}),
        "topics": list(topics), "conditions": {c: CONDITIONS[c] for c in conditions}, "repeats": repeats,
        "run_order": [{"n": i + 1, "topic": t, "condition": c} for i, (t, c) in enumerate(run_order(topics, list(conditions), repeats))],
        "primary_metrics": list(PRIMARY_METRICS), "exclusion_rules": EXCLUSION_RULES,
        "code_commit": run_store._project_git_commit() or None, "llm_provider": config.LLM_PROVIDER,
        "notes": notes,
    }
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path(name).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    status_path(name).write_text(json.dumps({"runs": []}, indent=1), encoding="utf-8")
    return plan


def load_status(name: str) -> dict:
    path = status_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}


def _save_status(name: str, status: dict) -> None:
    status_path(name).write_text(json.dumps(status, indent=1), encoding="utf-8")


def next_run(name: str) -> dict | None:
    plan, done = load_plan(name), {r["n"] for r in load_status(name)["runs"] if r.get("valid", True)}
    return next((r for r in plan["run_order"] if r["n"] not in done), None) if plan else None


def preflight(plan: dict, allow_dirty: bool = False) -> list:
    """Reasons the next run must not start (empty: it may)."""
    from src import config
    problems = []
    commit = run_store._project_git_commit()
    if plan.get("code_commit") and commit != plan["code_commit"]:
        problems.append(f"the code is at commit {commit[:8]}, the plan needs {plan['code_commit'][:8]}")
    if not allow_dirty and run_store._project_git_dirty():
        problems.append("tracked files have changes not in the commit (use --allow-dirty to accept, it is recorded)")
    if config.LLM_PROVIDER != plan.get("llm_provider"):
        problems.append(f"LLM_PROVIDER is {config.LLM_PROVIDER}, the plan needs {plan.get('llm_provider')}")
    if config.LLM_PROVIDER == "freellmapi":
        import requests
        base = config.FREELLMAPI_BASE_URL.rsplit("/v1", 1)[0]
        try:
            requests.get(f"{base}/api/ping", timeout=5).raise_for_status()
        except Exception as e:
            problems.append(f"the FreeLLMAPI router is not reachable at {base} ({type(e).__name__})")
    from datetime import timedelta
    recent = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(timespec="seconds")
    with run_store._connect() as conn:  # same ISO format as runs.started_at
        if conn.execute("SELECT 1 FROM runs WHERE status = 'running' AND started_at > ?", (recent,)).fetchone():
            problems.append("another pipeline run is active")
    return problems


def run_one(name: str, allow_dirty: bool = False) -> dict:
    """Run the next planned run. Returns its status record."""
    plan = load_plan(name)
    if plan is None:
        raise FileNotFoundError(f"no test plan {plan_path(name)}")
    item = next_run(name)
    if item is None:
        return {"done": True}
    problems = preflight(plan, allow_dirty)
    if problems:
        raise RuntimeError("not starting run %d: %s" % (item["n"], "; ".join(problems)))
    env = dict(os.environ, **plan["conditions"][item["condition"]], MEMORY_COMPARISON_GROUP=name)
    started = _now()

    def run_ids() -> set:
        with run_store._connect() as conn:
            return {r[0] for r in conn.execute("SELECT run_id FROM runs WHERE topic = ?", (item["topic"],))}

    before = run_ids()
    print(f"[Comparison] Run {item['n']}/{len(plan['run_order'])}: '{item['topic']}' under '{item['condition']}'...")
    proc = subprocess.run([sys.executable, "run.py", item["topic"]], cwd=run_store.REPO_ROOT, env=env,
                          timeout=RUN_TIMEOUT_SECONDS)
    new = sorted(run_ids() - before)  # the run this command started (one run at a time, checked above)
    record = {**item, "run_id": new[-1] if len(new) == 1 else None, "exit_code": proc.returncode,
              "started_at": started, "finished_at": _now()}
    record.update(_validity(record, plan))
    status = load_status(name)
    status["runs"].append(record)
    _save_status(name, status)
    return record


def _validity(record: dict, plan: dict) -> dict:
    """A run is invalid only when it did not run under its assigned condition."""
    if not record.get("run_id"):
        return {"valid": False, "invalid_reason": "no run was recorded"}
    from src.orchestration import run_metrics
    run = next((r for r in run_metrics.load_runs() if r["conditions"]["run_id"] == record["run_id"]), None)
    if run is None:
        return {"valid": False, "invalid_reason": "the run has no measurements"}
    cond = run["conditions"]
    expected = plan["conditions"][record["condition"]]["MEMORY_MODE"]
    if cond.get("memory_mode") != expected:
        return {"valid": False, "invalid_reason": f"recorded memory mode {cond.get('memory_mode')}, assigned {expected}"}
    if cond.get("comparison_group") != plan["comparison_group"]:
        return {"valid": False, "invalid_reason": f"recorded group {cond.get('comparison_group')}"}
    return {"valid": True}


def report(name: str) -> tuple[dict, list]:
    """(the comparison of the group's valid runs by assigned condition, those runs)."""
    from src.orchestration import run_metrics
    plan = load_plan(name)
    status = load_status(name)
    assigned = {r["run_id"]: r["condition"] for r in status["runs"] if r.get("valid") and r.get("run_id")}
    runs = [r for r in run_metrics.load_runs() if r["conditions"]["run_id"] in assigned]
    for r in runs:
        r["conditions"]["assigned_condition"] = assigned[r["conditions"]["run_id"]]
    result = run_metrics.compare_runs(runs, group_by="assigned_condition", metrics=tuple(plan["primary_metrics"]))
    result.update(plan=name, planned_runs=len(plan["run_order"]), valid_runs=len(runs),
                  invalid_runs=[r for r in status["runs"] if not r.get("valid")],
                  exposure={r["conditions"]["run_id"]: r["conditions"].get("memory_exposure") for r in runs})
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARISONS_DIR / f"{name}.report.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result, runs


def _main(argv: list) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.orchestration.comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("--topic", action="append", required=True)
    p.add_argument("--conditions", nargs="+", default=["off", "read"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-warnings-condition", action="store_true", help="add read_no_warnings")
    p.add_argument("--notes", default="")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--all", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    for cmd in ("status", "report"):
        sub.add_parser(cmd).add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "plan":
        conditions = list(args.conditions) + (["read_no_warnings"] if args.no_warnings_condition else [])
        plan = make_plan(args.name, args.topic, conditions, args.repeats, args.notes)
        print(f"Wrote {plan_path(args.name)}: {len(plan['run_order'])} runs, snapshot {plan['memory_snapshot']}, "
              f"held out: {plan['held_out_topics']}")
        return 0
    if args.command == "run":
        while True:
            record = run_one(args.name, args.allow_dirty)
            if record.get("done"):
                print("All planned runs are done.")
                return 0
            print(f"[Comparison] Run {record['n']}: run_id {record['run_id']}, "
                  + ("valid" if record["valid"] else f"INVALID ({record['invalid_reason']})"))
            if not args.all:
                return 0
    if args.command == "status":
        plan, status = load_plan(args.name), load_status(args.name)
        done = {r["n"]: r for r in status["runs"]}
        for item in plan["run_order"]:
            r = done.get(item["n"])
            state = "pending" if r is None else ("valid " + str(r["run_id"]) if r["valid"] else "INVALID: " + r["invalid_reason"])
            print(f"  {item['n']:>2}. {item['condition']:<17} {item['topic'][:50]:<50} {state}")
        return 0
    result, runs = report(args.name)
    from src.orchestration import run_metrics
    print(f"Plan {args.name}: {result['valid_runs']} valid run(s) of {result['planned_runs']} planned "
          f"({len(result['invalid_runs'])} invalid).")
    if runs:
        run_metrics._print_compare(runs, result)
    print(f"\nWrote {COMPARISONS_DIR / (args.name + '.report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
