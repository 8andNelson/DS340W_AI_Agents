"""
The memory parts of the final report (milestone M9; docs/CLAUDE_BACKEND_PLAN.md,
sections 11.8 and 14) -- for the Report Agent (project Milestone 13), which
does not exist yet.

  memory_used(run_id)  which past cases this run's agents were shown and
                       said they used, with the outcome that followed; a
                       script adapted in fast mode is named as adapted from
                       its source run, never as new work.
  comparison(name)     the controlled memory test (src/orchestration/
                       comparison.py) as rows of numbers, each with the
                       run_metrics rows it comes from.
  render_markdown(...) both as a report section.

Every number comes from recorded rows -- run_metrics, memory_usage,
memories -- never from an LLM, and each carries its source so it can be
traced (plan 15.10: "each number ... can be traced to a run_metrics row").
"""
import json


def memory_used(run_id: str) -> list[dict]:
    """[{memory_id, title, agent, kind, source_run, shown_to, used, outcome, fast_mode}] for one run."""
    from src import memory
    from src.orchestration import run_store
    c = memory.client()
    if c is None or c.local is None:
        return []
    usage = c.local.rows("memory_usage", run_id=run_id)
    fast = set()  # memories adapted in fast mode by any Replication attempt of the run
    with run_store._connect() as conn:
        for (output_json,) in conn.execute("SELECT output_json FROM stages WHERE run_id = ? AND agent = ?",
                                           (run_id, "Replication Agent")):
            output = json.loads(output_json or "null")
            if isinstance(output, dict) and output.get("fast_mode_from"):
                fast.add(output["fast_mode_from"])
    rows = {}
    for u in usage:
        entry = rows.setdefault(u["memory_id"], {"memory_id": u["memory_id"], "shown_to": set(), "used": False,
                                                 "helped": None})
        entry["shown_to"].add(u.get("agent") or "")
        entry["used"] = entry["used"] or bool(u.get("used"))
        if u.get("helped") is not None:
            entry["helped"] = bool(u["helped"]) if entry["helped"] is None else entry["helped"] and bool(u["helped"])
    out = []
    for memory_id, entry in rows.items():
        m = memory.get_memory(memory_id) or {}
        outcome = {True: "followed by a good outcome", False: "followed by a bad outcome"}.get(entry["helped"],
                                                                                              "not judged")
        out.append({"memory_id": memory_id, "title": m.get("title"), "agent": m.get("agent"), "kind": m.get("kind"),
                    "source_run": m.get("run_id"), "shown_to": sorted(entry["shown_to"]), "used": entry["used"],
                    "outcome": outcome if entry["used"] else "not used", "fast_mode": memory_id in fast})
    return sorted(out, key=lambda r: (not r["used"], r["agent"] or "", r["title"] or ""))


def comparison(name: str) -> dict:
    """The controlled test's numbers, each with its source rows."""
    from src.orchestration import comparison as test
    from src.orchestration.run_metrics import METRICS_VERSION
    result, runs = test.report(name)
    by_condition = {}
    for r in runs:
        by_condition.setdefault(r["conditions"]["assigned_condition"], []).append(r)
    rows = []
    for block in result["blocks"]:
        for metric in result["metrics"]:
            for condition, group in block["groups"].items():
                s = group["metrics"][metric]
                sources = [{"run_id": r["conditions"]["run_id"], "metric": metric,
                            "metrics_version": r["conditions"]["metrics_version"],
                            "value": r["metrics"].get(("run", metric), {}).get("value")}
                           for r in by_condition.get(condition, [])
                           if isinstance(r["metrics"].get(("run", metric), {}).get("value"), (int, float))]
                rows.append({"task": block["task"], "metric": metric, "condition": condition, "n": s["n"],
                             "runs": s["runs"], "median": s.get("median"), "mean": s.get("mean"),
                             "min": s.get("min"), "max": s.get("max"), "sources": sources})
    return {"name": name, "kind": result["kind"], "planned_runs": result["planned_runs"],
            "valid_runs": result["valid_runs"], "invalid_runs": len(result["invalid_runs"]),
            "tests": [{"task": b["task"], "metric": m, **t} for b in result["blocks"] for m, t in b["tests"].items()],
            "confounders": [{"task": b["task"], "field": f, "values": v} for b in result["blocks"]
                            for f, v in b["confounders"].items()],
            "exposure": result["exposure"], "rows": rows, "metrics_version": METRICS_VERSION}


def _cite(sources: list) -> str:
    return "; ".join(f"{s['run_id']}/{s['metric']}" for s in sources) or "no run has a value"


def render_markdown(run_id: str = None, test_name: str = None) -> str:
    """A report section: the memory this run used, and the controlled test."""
    lines = []
    if run_id:
        used = memory_used(run_id)
        lines += ["## Memory used", ""]
        if not used:
            lines.append("No past case from the shared memory was shown to this run's agents.")
        for r in used:
            how = (f"adapted from the script of run {r['source_run']} (fast mode), not written anew" if r["fast_mode"]
                   else "used" if r["used"] else "shown, not used")
            lines.append(f"- {r['agent']}: \"{r['title']}\" (memory {r['memory_id'][:8]}, from run {r['source_run']}) -- "
                         f"{how}; {r['outcome']}. Shown to: {', '.join(r['shown_to'])}.")
        lines += ["", "\"Used\" is the agents' own report; an outcome is what followed, not proof that the memory "
                      "caused it.", ""]
    if test_name:
        data = comparison(test_name)
        lines += [f"## Controlled memory test: {test_name}", "",
                  f"{data['valid_runs']} valid run(s) of {data['planned_runs']} planned ({data['invalid_runs']} invalid); "
                  f"a {data['kind']} comparison. Each value cites its source rows in the run database's run_metrics "
                  f"table (run id / metric).", "",
                  "| Task | Metric | Condition | n | Median | Range | Sources |", "|---|---|---|---|---|---|---|"]
        for r in data["rows"]:
            if r["n"]:
                lines.append(f"| {r['task']} | {r['metric']} | {r['condition']} | {r['n']}/{r['runs']} | "
                             f"{r['median']:.4g} | {r['min']:.4g} to {r['max']:.4g} | {_cite(r['sources'])} |")
            else:
                lines.append(f"| {r['task']} | {r['metric']} | {r['condition']} | 0/{r['runs']} | n/a | n/a | "
                             "no run has a value |")
        if data["tests"]:
            lines += ["", "Permutation tests (two-sided, on the mean; Cliff's delta, first group vs second):"]
            lines += [f"- {t['metric']}: {t['groups'][0]} vs {t['groups'][1]}, p = {t['p_value']:.4f}, "
                      f"delta = {t['cliffs_delta']:+.2f}, n = {t['n']}" for t in data["tests"]]
        else:
            lines += ["", "No test: each condition needs at least 3 runs with a value."]
        if data["confounders"]:
            lines += ["", "Conditions that differ between the groups (possible confounders):"]
            lines += [f"- {c['field']}: " + "; ".join(f"{g} = {', '.join(v)}" for g, v in c["values"].items())
                      for c in data["confounders"]]
        lines.append("")
    return "\n".join(lines)
