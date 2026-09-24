"""
Master Agent (Milestone 5+).

Orchestrates and supervises the pipeline. Master Agent is the only module
that invokes intake_agent, research_agent, validation_agent (which now also
performs Parent Paper selection -- see validation_agent.rank_and_recommend),
and data_agent (which builds a >=10k-entry dataset pool for the Parent
Paper). It does not re-score, re-select, or re-search anything itself; that
logic stays inside the specialized agents. Master Agent's job is to act as
the boss: run each agent, independently check its output against the hard
academic requirements in CLAUDE.md, and hold it accountable when it fails.

For every stage:
  1. Run the agent.
  2. Write a saveable, human-readable report of what it produced
     (logs/agent_reports/<agent>_attempt<N>.txt).
  3. Check the output. No issues -> log an Approval entry, advance.
  4. Issues found and a retry remains -> log a Correction entry describing
     exactly what failed, log that it is retrying, and re-run the agent once.
  5. Issues found and no retries remain -> log a final Correction/rejection
     and halt the pipeline.

Per CLAUDE.md's "Autonomous Operation" policy, Master Agent never blocks on
a human prompt -- retry-once-then-halt is itself the final decision, and
every step is logged (both the structured JSON log and the saveable text
reports) so a person can review the full trail after the fact.

Slack is the human-observable layer on top of that same trail (Milestone
7): every agent invocation posts its own start/finish message (via
_supervise), and every Master Agent Correction/Approval also posts (via
_print_entry) -- see src/slack/slack_client.py and message_formatter.py.
Routine status (agent start/finish, Master Agent Approvals) goes to the
main channel; Master Agent Corrections and agent exceptions go to the
errors_corrections channel, so there's one place to check for anything
that needs attention. Slack posting always fails soft, so a missing/
unreachable Slack channel never changes pipeline behavior, only its
visibility.
"""
import json
from pathlib import Path

from src.slack import slack_client, message_formatter

MIN_QUALIFYING = 5
MAX_ATTEMPTS = 2  # first attempt + one retry
DATA_REQUEST_TARGET_INCREMENT = 5000  # fallback bump when a Slack request
                                       # doesn't state an explicit target=<int>

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = REPO_ROOT / "logs" / "master_agent_log.json"
REPORT_DIR = REPO_ROOT / "logs" / "agent_reports"
SLACK_SUMMARY_MAX_CHARS = 600


def _log_entry(action: str, target: str, issue: str, decision: str, next_step: str) -> dict:
    return {
        "sender": "master_agent",
        "action": action,      # "Correction" | "Approval"
        "target": target,      # which agent's output is being acted on
        "issue": issue,
        "decision": decision,  # "Approved" | "Rejected" | "Retrying"
        "next": next_step,
    }


def _append_log(entries: list) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.extend(entries)
    LOG_FILE.write_text(json.dumps(existing, indent=2))


def _print_entry(entry: dict) -> None:
    print("\n[Master Agent]")
    print(f"ACTION: {entry['action']}")
    print(f"TARGET: {entry['target']}")
    print(f"ISSUE: {entry['issue']}")
    print(f"DECISION: {entry['decision']}")
    print(f"NEXT: {entry['next']}")
    # Per CLAUDE.md's "Master Agent Slack Requirement": every correction,
    # override, or approval Master Agent makes is posted to Slack, not
    # just logged locally, so the supervision trail is observable live.
    # Corrections go to the errors_corrections channel (one place to check
    # for anything that needs attention); Approvals stay on the main
    # channel alongside routine agent status.
    text = message_formatter.format_master_entry(entry)
    if entry["action"] == "Correction":
        slack_client.post_error_message(text)
    else:
        slack_client.post_message(text)


def _summarize_for_slack(text: str) -> str:
    """Keep an agent's Slack "finished" post short -- the full report is
    already saved to logs/agent_reports/ by _write_report, so Slack only
    needs enough to be useful at a glance."""
    text = text.strip()
    if len(text) <= SLACK_SUMMARY_MAX_CHARS:
        return text
    return text[:SLACK_SUMMARY_MAX_CHARS].rstrip() + " …(see saved report for full detail)"


def _write_report(agent_name: str, attempt: int, text: str) -> Path:
    """Save a human-readable report of one agent invocation to disk."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = agent_name.lower().replace(" ", "_")
    path = REPORT_DIR / f"{safe_name}_attempt{attempt}.txt"
    path.write_text(text)
    return path


# --- Per-stage report formatters ---

def _format_intake_report(structured: dict) -> str:
    lines = [
        "[Intake Agent] Structured research request",
        f"Domain:        {structured.get('domain', '')}",
        f"ML Task:       {structured.get('ml_task', '')}",
        f"Keywords:      {', '.join(structured.get('keywords', []))}",
        f"Search Queries: {', '.join(structured.get('search_queries', []))}",
    ]
    return "\n".join(lines)


def _format_research_report(papers: list) -> str:
    qualifying = [p for p in papers if p.get("peer_reviewed") and p.get("year", 0) >= 2022]
    lines = [f"[Research Agent] {len(papers)} candidate papers found ({len(qualifying)} qualifying)."]
    for i, p in enumerate(papers, 1):
        peer = "peer-reviewed" if p.get("peer_reviewed") else "NOT peer-reviewed"
        lines.append(f"  [{i}] {p.get('title', '')} ({p.get('year', '?')}, {peer})")
        if p.get("paper_url"):
            lines.append(f"       URL: {p['paper_url']}")
    return "\n".join(lines)


def _format_validation_report(result: dict) -> str:
    lines = [
        "[Validation Agent] Validation + Parent Paper selection",
        f"VERIFIED:   {len(result.get('verified', []))}",
        f"UNVERIFIED: {len(result.get('unverified', []))}",
        f"REJECTED:   {len(result.get('rejected', []))}",
        f"Candidate pool source: {result.get('candidate_pool_source')} "
        f"(size {result.get('candidate_pool_size', 0)})",
    ]
    chosen = result.get("recommended_parent_paper")
    if chosen:
        lines.append(f"Recommended Parent Paper: {chosen.get('title', '')}")
        lines.append(f"Justification:\n{result.get('justification', '')}")
    else:
        lines.append(f"No Parent Paper recommended. Reason: {result.get('degraded_reason', '')}")
    return "\n".join(lines)


def _format_data_report(result: dict) -> str:
    lines = [
        "[Data Agent] Dataset pool for Parent Paper replication",
        f"Status:        {result.get('status', '')}",
        f"Search stage:  {result.get('search_stage', '')}",
        f"Total entries: {result.get('total_entries', 0)} / {result.get('target_entries', 10000)} "
        f"(target met: {result.get('target_met', False)})",
        f"Datasets used: {len(result.get('selected_datasets', []))} "
        f"(considered {result.get('datasets_considered', 0)} papers)",
    ]
    for d in result.get("selected_datasets", []):
        label = d.get("display_name") or d.get("name", "")
        lines.append(f"  - {label} ({d.get('source', '')}): {d.get('entry_count', '?')} entries -> {d.get('name', '')}")
    if result.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result["warnings"])
    return "\n".join(lines)


def _format_cleaning_report(result: dict) -> str:
    lines = [
        "[Cleaning Agent] Master CSV for Parent Paper replication",
        f"Status:            {result.get('status', '')}",
        f"Datasets merged:   {result.get('datasets_merged', 0)} / {result.get('datasets_attempted', 0)}",
        f"Rows / columns:    {result.get('rows_total', 0)} / {result.get('columns_total', 0)} "
        f"(target: {result.get('target_rows', 10000)}, target met: {result.get('target_met', False)})",
        f"Master CSV:        {result.get('master_csv_path', '')}",
    ]
    if result.get("conflicts"):
        lines.append("Conflicts:")
        for c in result["conflicts"]:
            lines.append(f"  - {c.get('dataset', '')} ({c.get('link', '')}): {c.get('reason', '')}")
    return "\n".join(lines)


# --- Per-stage checkers: each returns a list of blocker log entries ---

def _check_intake(structured: dict) -> list:
    domain = (structured or {}).get("domain", "")
    keywords = (structured or {}).get("keywords", [])
    search_queries = (structured or {}).get("search_queries", [])

    if not domain and not keywords and not search_queries:
        return [_log_entry(
            "Correction", "Intake Agent",
            "Intake Agent produced no usable domain, keywords, or search queries from the topic.",
            "Rejected",
            "Re-run Intake Agent; if it fails again, escalate for manual topic clarification.",
        )]
    return []


def _check_research(papers: list) -> list:
    if not papers:
        return [_log_entry(
            "Correction", "Research Agent",
            "Research Agent returned no candidate papers.",
            "Rejected",
            "Re-run Research Agent with broader search terms.",
        )]

    qualifying = [p for p in papers if p.get("peer_reviewed") and p.get("year", 0) >= 2022]
    if len(qualifying) < MIN_QUALIFYING:
        return [_log_entry(
            "Correction", "Research Agent",
            f"Only {len(qualifying)} of {len(papers)} papers meet basic peer-reviewed + "
            f"2022-present requirements; the project requires at least {MIN_QUALIFYING}.",
            "Rejected",
            "Broaden search queries or add academic sources to reach the 5-paper minimum.",
        )]
    return []


def _check_validation_and_selection(recommendation: dict) -> list:
    """
    Run deterministic hard-constraint checks against the Validation Agent's
    Parent Paper recommendation. Returns a list of blocker log entries --
    empty if the recommendation passes every check.
    """
    blockers = []

    if recommendation.get("status") == "NO_CANDIDATES":
        blockers.append(_log_entry(
            "Correction", "Validation Agent",
            "No non-rejected candidate papers were available to recommend a Parent Paper.",
            "Rejected",
            "Re-run the Research Agent with broader search terms and re-validate before retrying selection.",
        ))
        return blockers  # nothing else to meaningfully check

    pool_source = recommendation.get("candidate_pool_source")
    if pool_source != "verified":
        blockers.append(_log_entry(
            "Correction", "Validation Agent",
            f"Parent Paper candidates were drawn from '{pool_source}', not 'verified' -- Layer 3 "
            "independent validation did not confirm peer-reviewed status for any candidate.",
            "Rejected",
            "Re-run paper validation (check LLM connectivity/model) until at least one paper reaches VERIFIED status.",
        ))

    pool_size = recommendation.get("candidate_pool_size", 0)
    if pool_size < MIN_QUALIFYING:
        blockers.append(_log_entry(
            "Correction", "Research Agent",
            f"Only {pool_size} qualifying candidate papers were available; the project requires at least {MIN_QUALIFYING}.",
            "Rejected",
            "Broaden search queries or add academic sources to reach the 5-paper minimum.",
        ))

    chosen = recommendation.get("recommended_parent_paper")
    row = recommendation["comparison_table"][0] if recommendation.get("comparison_table") else None

    if chosen and row:
        if not row.get("peer_reviewed"):
            blockers.append(_log_entry(
                "Correction", "Validation Agent",
                f"Recommended paper '{chosen.get('title', '')}' is not confirmed peer-reviewed.",
                "Rejected",
                "Select the next-ranked candidate that is confirmed peer-reviewed, or escalate for manual review.",
            ))

        has_replication_detail = (
            row.get("dataset_availability") != "Not specified"
            or bool((chosen.get("methodology") or "").strip())
            or bool((chosen.get("results_summary") or "").strip())
        )
        if not has_replication_detail:
            blockers.append(_log_entry(
                "Correction", "Validation Agent",
                f"Recommended paper '{chosen.get('title', '')}' does not clearly identify a dataset, "
                "methodology, or results -- insufficient detail to realistically replicate.",
                "Rejected",
                "Select the next-ranked candidate with clearer implementation detail, or escalate for manual review.",
            ))

        audit = recommendation.get("code_availability_audit", {})
        if recommendation.get("code_availability_deciding_factor") and audit.get("policy_violation"):
            blockers.append(_log_entry(
                "Correction", "Validation Agent",
                f"Recommended paper '{chosen.get('title', '')}' was ranked #1 primarily because of code "
                "availability, while multiple qualifying papers report available code. Project policy requires "
                "academic merit to drive selection, not code convenience.",
                "Rejected",
                "Re-rank with code availability excluded as a deciding factor, or justify the academic merits independently.",
            ))

    return blockers


def _check_data(result: dict) -> list:
    """
    Data Agent is only blocked when it found zero usable datasets. A
    DEGRADED result (1+ usable datasets but under the entry target) is
    approved and passed through -- the shortfall is still visible in the
    saved report and the returned dataset_result, per CLAUDE.md's Results
    Integrity policy of surfacing real outcomes rather than hiding them.
    """
    if not result.get("selected_datasets"):
        return [_log_entry(
            "Correction", "Data Agent",
            "Data Agent found zero usable datasets (verified, in-scope, downloaded, and passing "
            "the reproducibility-fit check) across the Parent Paper and its ranked alternates. "
            f"Warnings: {'; '.join(result.get('warnings', [])) or 'none reported'}.",
            "Rejected",
            "Re-run Data Agent; if it fails again, halt for manual dataset review.",
        )]
    return []


def _check_cleaning(result: dict) -> list:
    """
    Cleaning Agent is only blocked when it could not merge a single
    dataset into a master CSV. A DEGRADED result (1+ datasets merged, but
    some logged as conflicts) is approved and passed through -- the
    conflicts stay visible in the saved report and the returned
    cleaning_result, per CLAUDE.md's Results Integrity policy.
    """
    if not result.get("datasets_merged"):
        conflict_reasons = "; ".join(c.get("reason", "") for c in result.get("conflicts", [])) or "none reported"
        return [_log_entry(
            "Correction", "Cleaning Agent",
            f"Cleaning Agent could not merge any dataset into a master CSV. Conflicts: {conflict_reasons}.",
            "Rejected",
            "Re-run Cleaning Agent; if it fails again, halt for manual dataset review.",
        )]
    return []


def _supervise(agent_name: str, run_fn, checker_fn, report_fn, max_attempts: int = MAX_ATTEMPTS):
    """
    Invoke run_fn() up to max_attempts times, independently checking its
    output with checker_fn after each attempt. Writes a saveable report after
    every attempt and logs Correction/Approval entries to LOG_FILE.

    Returns (output, blockers). blockers is empty when the stage passed
    (on the first attempt or after a retry); non-empty means the stage is
    rejected and the pipeline should halt.
    """
    output = None
    blockers = []

    for attempt in range(1, max_attempts + 1):
        print(f"\n[Master Agent] Running {agent_name} (attempt {attempt}/{max_attempts})...")
        slack_client.post_message(message_formatter.format_started(
            agent_name, f"Stage starting (attempt {attempt}/{max_attempts})."
        ))

        try:
            output = run_fn()
        except Exception as e:
            slack_client.post_error_message(message_formatter.format_finished(
                agent_name, "Failed", f"Raised an exception: {e}",
            ))
            report_path = _write_report(agent_name, attempt, f"{agent_name} raised an exception:\n{e}")
            decision = "Retrying" if attempt < max_attempts else "Rejected"
            next_step = (
                f"Re-running {agent_name} (attempt {attempt + 1}/{max_attempts})."
                if attempt < max_attempts
                else "Halting pipeline; manual review required."
            )
            entry = _log_entry(
                "Correction", agent_name,
                f"{agent_name} raised an exception on attempt {attempt}: {e}. Report saved to {report_path}.",
                decision, next_step,
            )
            _print_entry(entry)
            _append_log([entry])
            blockers = [entry]
            if attempt < max_attempts:
                continue
            return None, blockers

        report_text = report_fn(output)
        report_path = _write_report(agent_name, attempt, report_text)
        slack_client.post_message(message_formatter.format_finished(
            agent_name, "Completed", _summarize_for_slack(report_text),
        ))

        blockers = checker_fn(output)

        if not blockers:
            approval = _log_entry(
                "Approval", agent_name,
                f"{agent_name} output passed all checks on attempt {attempt}. Report saved to {report_path}.",
                "Approved",
                "Advance to next stage.",
            )
            _print_entry(approval)
            _append_log([approval])
            return output, []

        for b in blockers:
            _print_entry(b)
        _append_log(blockers)

        if attempt < max_attempts:
            retry_note = _log_entry(
                "Correction", agent_name,
                f"{agent_name} failed {len(blockers)} check(s) on attempt {attempt}; see above. Report saved to {report_path}.",
                "Retrying",
                f"Re-running {agent_name} (attempt {attempt + 1}/{max_attempts}).",
            )
            _print_entry(retry_note)
            _append_log([retry_note])
        else:
            final = _log_entry(
                "Correction", agent_name,
                f"{agent_name} still failing after {max_attempts} attempt(s). See issues above.",
                "Rejected",
                "Halting pipeline; manual review required.",
            )
            _print_entry(final)
            _append_log([final])

    return output, blockers


def _max_ts(a: str | None, b: str | None) -> str | None:
    """Numeric max of two Slack timestamps (either may be None/empty)."""
    if not a:
        return b
    if not b:
        return a
    return a if float(a) >= float(b) else b


def check_agent_requests(parent_paper: dict = None, ranked_pool: list = None,
                          floor_ts: str = None) -> dict | None:
    """
    Read any Slack messages posted since the last check and act on the one
    request shape Master Agent currently knows how to fulfill: another
    agent (today, no such agent exists yet -- this is the hook a future
    Modeling/Experiment Agent uses) asking for Data Agent to be rerun with
    a larger target, via message_formatter.format_request /
    "REQUEST: Data Agent". This is Master Agent "overseeing the channel":
    it does not poll continuously, but re-checks once per call, which is
    enough as long as something (run_pipeline, a scheduled job, a manual
    re-check) calls it periodically.

    floor_ts is a hard lower bound on top of the persisted cursor
    (logs/project_state.json's slack_last_checked_ts) -- pass the current
    run's own "run started" banner ts (see run_pipeline) so this run can
    never act on a message left over from a *different*, earlier topic's
    run. The persisted cursor still advances normally across runs (so old
    history isn't endlessly re-scanned); floor_ts is what actually
    prevents cross-topic contamination, since a request only ever makes
    sense in the context of the parent_paper/ranked_pool it was posted
    against.

    Anything Master Agent doesn't yet know how to act on -- a request
    targeting an agent other than Data Agent, or a plain non-request
    message -- is acknowledged in Slack rather than silently ignored, per
    CLAUDE.md's "never silently drop" policy, but does not raise or block.

    Returns None when Slack isn't configured, there is nothing new, or
    nothing in the new messages was an actionable request. Otherwise
    returns {"dataset_result": ..., "cleaning_result": ..., "blockers": [...]}
    for the last acted-on request (blockers non-empty means that re-run
    itself failed Master Agent's checks).
    """
    from src.agents import data_agent, cleaning_agent
    from src.orchestration.state_manager import load_state, update_state

    if not slack_client.is_configured():
        return None

    state = load_state()
    effective_oldest_ts = _max_ts(state.get("slack_last_checked_ts"), floor_ts)
    messages = slack_client.fetch_new_messages(oldest_ts=effective_oldest_ts)
    if not messages:
        return None

    update_state({"slack_last_checked_ts": messages[-1]["ts"]})

    acted = None
    for msg in messages:
        request = message_formatter.parse_request(msg.get("text", ""))
        if not request:
            continue

        if request["target"] != "Data Agent":
            slack_client.post_message(message_formatter.format_finished(
                "Master Agent", "Noted",
                f"Received a request from {request['sender']} for '{request['target']}', but Master "
                "Agent does not yet know how to act on that agent -- no action taken.",
            ))
            continue

        if not parent_paper or not ranked_pool:
            slack_client.post_message(message_formatter.format_finished(
                "Master Agent", "Blocked",
                f"Received a Data Agent request from {request['sender']} ({request['reason']}), but no "
                "Parent Paper/ranked candidate pool is available in this run to re-search against.",
            ))
            continue

        new_target = (
            message_formatter.parse_target_from_detail(request["detail"])
            or data_agent.TARGET_ENTRIES + DATA_REQUEST_TARGET_INCREMENT
        )

        slack_client.post_message(message_formatter.format_started(
            "Master Agent",
            f"Relaying {request['sender']}'s request ({request['reason']}): "
            f"re-running Data Agent with target={new_target}.",
        ))

        new_dataset_result, blockers = _supervise(
            "Data Agent",
            lambda: data_agent.run(parent_paper, ranked_pool, target_entries=new_target),
            _check_data,
            _format_data_report,
        )
        if blockers:
            acted = {"dataset_result": None, "cleaning_result": None, "blockers": blockers}
            continue

        new_cleaning_result, blockers = _supervise(
            "Cleaning Agent",
            lambda: cleaning_agent.run(new_dataset_result["selected_datasets"]),
            _check_cleaning,
            _format_cleaning_report,
        )
        update_state({
            "dataset": {
                "selected_datasets": new_dataset_result["selected_datasets"],
                "total_entries": new_dataset_result["total_entries"],
                "target_met": new_dataset_result["target_met"],
                "is_combined": new_dataset_result["is_combined"],
            },
            "cleaning": {
                "master_csv_path": new_cleaning_result["master_csv_path"],
                "datasets_merged": new_cleaning_result["datasets_merged"],
                "rows_total": new_cleaning_result["rows_total"],
                "target_met": new_cleaning_result["target_met"],
                "conflicts": new_cleaning_result["conflicts"],
            },
        })
        acted = {"dataset_result": new_dataset_result, "cleaning_result": new_cleaning_result, "blockers": blockers}

    return acted


def _relay_and_apply(result: dict, ranked_pool: list, floor_ts: str) -> None:
    """Call check_agent_requests with this run's context/floor and, if it
    successfully acted on a request, fold the fresh dataset_result/
    cleaning_result into result. A relay attempt that itself failed
    Master Agent's checks (blockers non-empty) is left alone -- it's
    already been logged/posted by the nested _supervise calls, and must
    not silently overwrite this run's last good dataset_result/
    cleaning_result with None."""
    relay = check_agent_requests(result["parent_paper"], ranked_pool, floor_ts=floor_ts)
    if relay and not relay["blockers"]:
        result["dataset_result"] = relay["dataset_result"]
        result["cleaning_result"] = relay["cleaning_result"]


def run_pipeline(topic: str) -> dict:
    """
    Drive the full pipeline (Intake -> Research -> Validation & Selection ->
    Data Agent -> Cleaning Agent), supervising every stage. Master Agent
    invokes each agent itself, checks its output, retries once on failure,
    and halts (with a logged rejection) if the retry also fails.

    Returns:
        {
          "success": bool,
          "phase": str,
          "structured": dict | None,
          "papers": list | None,
          "validation_result": dict | None,
          "parent_paper": dict | None,
          "dataset_result": dict | None,
          "cleaning_result": dict | None,
          "blockers": [...],   # non-empty only when success is False
        }
    """
    # Imported here (not at module load) so master_agent stays the sole
    # orchestrator without creating an import cycle with src.main.
    from src.agents import intake_agent, research_agent, validation_agent, data_agent, cleaning_agent
    from src.orchestration.state_manager import update_state
    from src.orchestration import dataset_cache

    # This run's own banner ts is the floor check_agent_requests uses below
    # -- see check_agent_requests' floor_ts docstring for why this matters
    # once the pipeline is run repeatedly for different topics.
    run_floor_ts = slack_client.post_message(message_formatter.format_run_started(topic))

    result = {
        "success": False,
        "phase": None,
        "structured": None,
        "papers": None,
        "validation_result": None,
        "parent_paper": None,
        "dataset_result": None,
        "cleaning_result": None,
        "blockers": [],
    }

    # --- Stage 1: Intake ---
    structured, blockers = _supervise(
        "Intake Agent",
        lambda: intake_agent.run(topic),
        _check_intake,
        _format_intake_report,
    )
    if blockers:
        result.update(phase="TOPIC_ANALYSIS", blockers=blockers)
        return result

    structured["project_topic"] = topic
    update_state({
        "project_topic": topic,
        "research_question": structured.get("domain", ""),
        "phase": "TOPIC_ANALYSIS",
    })
    result["structured"] = structured

    # --- Stage 2: Research ---
    papers, blockers = _supervise(
        "Research Agent",
        lambda: research_agent.run(structured),
        _check_research,
        _format_research_report,
    )
    if blockers:
        result.update(phase="LITERATURE_SEARCH", blockers=blockers)
        return result

    update_state({"papers": papers, "phase": "LITERATURE_SEARCH"})
    result["papers"] = papers

    # --- Stage 3: Validation & Parent Paper Selection ---
    validation_result, blockers = _supervise(
        "Validation Agent",
        lambda: validation_agent.run(papers),
        _check_validation_and_selection,
        _format_validation_report,
    )
    if blockers:
        update_state({
            "papers": validation_result["all"] if validation_result else papers,
            "parent_paper": None,
            "parent_paper_approved": False,
            "phase": "PARENT_PAPER_SELECTION",
        })
        result.update(phase="PARENT_PAPER_SELECTION", blockers=blockers)
        return result

    update_state({
        "papers": validation_result["all"],
        "parent_paper": validation_result["recommended_parent_paper"],
        "parent_paper_approved": True,
        "phase": "MASTER_APPROVAL",
    })
    result["validation_result"] = validation_result
    result["parent_paper"] = validation_result["recommended_parent_paper"]

    # A Data Agent request only becomes actionable once parent_paper/
    # ranked_pool exist -- i.e. from here on. Checking any earlier would
    # let this run's own first check permanently mark a not-yet-actionable
    # request as "seen" (advancing the cursor) before it ever had context
    # to act on, losing it for the rest of this run.
    _relay_and_apply(result, validation_result["ranked_pool"], run_floor_ts)

    # --- Stage 4: Data Agent (dataset pool discovery + verification) ---
    dataset_cache.cache_validated_papers(validation_result["ranked_pool"])

    dataset_result, blockers = _supervise(
        "Data Agent",
        lambda: data_agent.run(validation_result["recommended_parent_paper"], validation_result["ranked_pool"]),
        _check_data,
        _format_data_report,
    )
    if blockers:
        update_state({"phase": "DATA_DISCOVERY"})
        result.update(phase="DATA_DISCOVERY", blockers=blockers)
        return result

    update_state({
        "dataset": {
            "selected_datasets": dataset_result["selected_datasets"],
            "total_entries": dataset_result["total_entries"],
            "target_met": dataset_result["target_met"],
            "is_combined": dataset_result["is_combined"],
        },
        "phase": "DATA_DISCOVERY",
    })
    result["dataset_result"] = dataset_result

    _relay_and_apply(result, validation_result["ranked_pool"], run_floor_ts)

    # --- Stage 5: Cleaning Agent (merge downloaded datasets into a master CSV) ---
    cleaning_result, blockers = _supervise(
        "Cleaning Agent",
        lambda: cleaning_agent.run(dataset_result["selected_datasets"]),
        _check_cleaning,
        _format_cleaning_report,
    )
    if blockers:
        update_state({"phase": "DATA_CLEANING"})
        result.update(phase="DATA_CLEANING", blockers=blockers)
        return result

    update_state({
        "cleaning": {
            "master_csv_path": cleaning_result["master_csv_path"],
            "datasets_merged": cleaning_result["datasets_merged"],
            "rows_total": cleaning_result["rows_total"],
            "target_met": cleaning_result["target_met"],
            "conflicts": cleaning_result["conflicts"],
        },
        "phase": "DATA_CLEANING",
    })

    result.update(
        success=True,
        phase="DATA_CLEANING",
        cleaning_result=cleaning_result,
    )

    _relay_and_apply(result, validation_result["ranked_pool"], run_floor_ts)

    return result
