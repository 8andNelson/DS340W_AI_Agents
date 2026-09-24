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
"""
import json
from pathlib import Path

MIN_QUALIFYING = 5
MAX_ATTEMPTS = 2  # first attempt + one retry

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = REPO_ROOT / "logs" / "master_agent_log.json"
REPORT_DIR = REPO_ROOT / "logs" / "agent_reports"


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

        try:
            output = run_fn()
        except Exception as e:
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
    return result
