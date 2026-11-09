"""
Master Agent (Milestone 5+).

Orchestrates and supervises the pipeline. Master Agent is the only module
that invokes intake_agent, research_agent, validation_agent (which now also
performs Parent Paper selection -- see validation_agent.rank_and_recommend),
data_agent and every later agent. It does not re-score, re-select, or re-search anything itself; that
logic stays inside the specialized agents. Master Agent's job is to act as
the boss: run each agent, independently check its output against the hard
academic requirements in CLAUDE.md, and hold it accountable when it fails.

For every stage:
  1. Run the agent.
  2. Write a saveable, human-readable report of what it produced
     (logs/agent_reports/<agent>_attempt<N>.txt).
  3. Check the output. No issues -> log an Approval entry, advance.
  4. Issues found (or an exception) -> diagnose the failure (one LLM call,
     _diagnose), log the diagnosis as a Correction, and re-run the agent
     with every diagnosis so far in its LLM prompts (llm.supervisor_notes).
  5. After MAX_ATTEMPTS (5) errors -> log a final Correction/rejection and
     halt the pipeline.

Per CLAUDE.md's "Autonomous Operation" policy, Master Agent never blocks on
a human prompt -- diagnose-and-retry-then-halt is itself the final decision, and
every step is logged (both the structured JSON log and the saveable text
reports) so a person can review the full trail after the fact.

Agents never run other agents. Each agent's module has a handoff()
that returns its request to Master Agent: which agent to run next, and
what to pass on (src/orchestration/handoff.py). Master Agent checks each
request and runs the requested agent (see _drive). After the first
evaluation the pipeline is not linear: the Evaluation Agent (validity
fixes) and the Experiment Agent (closing the gap to the Parent Paper's
score) hand off rerun requests for an earlier agent, with new knowledge.
These go through the same path: Master Agent checks the request, declines
a redundant one and tells the sender why, runs the requested agent, and
follows each agent's handoff until the new baseline is evaluated.

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
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.orchestration import run_store
from src.slack import slack_client, message_formatter

MIN_QUALIFYING = 5
# A stage halts after this many errors (failed attempts). Between attempts
# Master Agent diagnoses the failure and re-runs the agent with that
# diagnosis (see _supervise and _diagnose).
MAX_ATTEMPTS = 5
DIAGNOSIS_TEXT_CHARS = 4000  # of the failure and of the agent's report, per diagnosis prompt
FAILURE_NOTE_CHARS = 600     # of the raw failure, in each note passed to the retried agent
DATA_REQUEST_TARGET_INCREMENT = 5000  # fallback bump when a Slack request
                                       # doesn't state an explicit target=<int>

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = REPO_ROOT / "logs" / "master_agent_log.json"
ROUND_BACKUP_DIR = REPO_ROOT / "logs" / "round_backups"
REPORT_DIR = REPO_ROOT / "logs" / "agent_reports"
SLACK_SUMMARY_MAX_CHARS = 600

# The run database row for the pipeline run in progress (see run_pipeline).
# Module-level, like Slack, so every _supervise call -- including the ones
# check_agent_requests makes -- records its attempt without extra plumbing.
_current_run_id = None


def _log_entry(action: str, target: str, issue: str, decision: str, next_step: str) -> dict:
    return {
        "sender": "master_agent",
        "action": action,      # "Correction" | "Approval"
        "target": target,      # which agent's output is being acted on
        "issue": issue,
        "decision": decision,  # "Approved" | "Rejected" | "Retrying"
        "next": next_step,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    # UTF-8, not the platform default: reports quote paper text, and on
    # Windows (cp1252) a PDF ligature such as "ﬁ" (U+FB01) raised
    # UnicodeEncodeError and crashed a live run.
    path.write_text(text, encoding="utf-8")
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
        if result.get("dataset_warning"):
            lines.append(f"Warning: {result['dataset_warning']}")
    else:
        lines.append(f"No Parent Paper recommended. Reason: {result.get('degraded_reason', '')}")
    return "\n".join(lines)


def _format_data_report(result: dict) -> str:
    lines = [
        "[Data Agent] Dataset pool for Parent Paper replication",
        f"Status:        {result.get('status', '')}",
        f"Mode:          {result.get('mode', 'parent_only')}"
        + ("" if result.get("parent_dataset_found", True) else "  (Parent Paper's own dataset NOT found; substitute)"),
        f"Paper read:    {result.get('paper_text_notes') or result.get('paper_text_status') or 'unknown'}",
        f"Search stage:  {result.get('search_stage', '')}",
        f"Total entries: {result.get('total_entries', 0)} / {result.get('target_entries') or 'no target'} "
        f"(target met: {result.get('target_met')})",
        f"Datasets used: {len(result.get('selected_datasets', []))} "
        f"(considered {result.get('datasets_considered', 0)} papers)",
    ]
    plan = result.get("data_plan") or {}
    if plan:
        lines.append("Data plan (read from the Parent Paper; the suggested size is advice for the Evaluation Agent):")
        if plan.get("target_entries"):
            lines.append(f"  Suggested rows:       {plan['target_entries']}")
        if plan.get("paper_model"):
            lines.append(f"  Paper's model:        {plan['paper_model']}")
        if plan.get("dataset_urls"):
            lines.append(f"  Links in the paper:   {', '.join(plan['dataset_urls'])}")
        if plan.get("paper_dataset"):
            size = plan.get("paper_dataset_size")
            lines.append(f"  Paper's dataset:      {plan['paper_dataset']}"
                         + (f" ({size} rows, as stated in the paper)" if size else ""))
        lines.append(f"  Reasoning:            {plan.get('reasoning') or 'none given'}")
    for d in result.get("selected_datasets", []):
        label = d.get("display_name") or d.get("name", "")
        lines.append(f"  - {label} ({d.get('source', '')}): {d.get('entry_count', '?')} entries -> {d.get('name', '')}")
        if d.get("column_notes"):
            lines.append(f"      note: {d['column_notes']}")
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
        f"(target: {result.get('target_rows') or 'none'}, target met: {result.get('target_met')})",
        f"Master CSV:        {result.get('master_csv_path', '')}",
    ]
    split_rows = result.get("split_rows") or {}
    if result.get("train_csv_path"):
        lines.append(
            f"Train/Test/Val:    {split_rows.get('train', 0)}/{split_rows.get('test', 0)}/"
            f"{split_rows.get('validation', 0)} rows (seed={result.get('split_seed', '')}) -> "
            f"{result['train_csv_path']}, {result.get('test_csv_path', '')}, {result.get('validation_csv_path', '')}"
        )
    if result.get("conflicts"):
        lines.append("Conflicts:")
        for c in result["conflicts"]:
            lines.append(f"  - {c.get('dataset', '')} ({c.get('link', '')}): {c.get('reason', '')}")
    return "\n".join(lines)


def _format_code_discovery_report(result: dict) -> str:
    lines = [
        "[Code Discovery Agent] Implementation search for Parent Paper",
        f"Status:       {result.get('status', '')}",
        f"Source stage: {result.get('source_stage', '')}",
        f"Candidates considered: {len(result.get('candidates_considered', []))}",
    ]
    repo = result.get("repository")
    if repo:
        lines.append(f"Repository:  {repo.get('repo_url', '')} ({repo.get('source', '')})")
        lines.append(f"License:     {repo.get('license', '') or 'unknown'}")
        lines.append(f"Stars:       {repo.get('stars', 'n/a')}")
        lines.append(f"Dependency files: {', '.join(repo.get('dependency_files', [])) or 'none found'}")
    else:
        lines.append("No verified implementation found -- Replication Agent will reconstruct from the paper.")
    if result.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result["warnings"])
    return "\n".join(lines)


def _format_replication_report(result: dict) -> str:
    lines = [
        "[Replication Agent] Baseline reproduction",
        f"Status: {result.get('status', '')}",
        f"Source: {result.get('source', '')}",
    ]
    if result.get("repo_url"):
        lines.append(f"Repository: {result['repo_url']} (commit {result.get('commit_sha', '') or 'unknown'})")
    lines.append(f"Entry point: {result.get('entry_point', '')} (via {result.get('entry_point_source', '')})")
    lines.append(f"Exit code: {result.get('exit_code')}  Duration: {result.get('duration_seconds')}s")
    lines.append(f"Replication dir: {result.get('replication_dir', '')}")
    if result.get("inferred_assumptions"):
        lines.append("Inferred assumptions (reconstructed code, not the authors'):")
        lines.extend(f"  - {a}" for a in result["inferred_assumptions"])
    if result.get("modifications"):
        lines.append("Modifications made to run the original repo:")
        lines.extend(f"  - {m}" for m in result["modifications"])
    if result.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result["warnings"])
    return "\n".join(lines)


def _format_evaluation_report(result: dict) -> str:
    lines = [
        "[Evaluation Agent] Reproduced vs. published results",
        f"Overall: {result.get('status', '')}",
    ]
    for m in result.get("metrics", []):
        lines.append(
            f"  - {m['metric']}: published={m['published_value']}, reproduced={m['reproduced_value']} "
            f"-> {m['classification']}" + (f" ({m['notes']})" if m.get("notes") else "")
        )
    if not result.get("metrics"):
        lines.append("  No comparable metrics found.")
    if result.get("likely_causes"):
        lines.append("Likely causes:")
        lines.extend(f"  - {c}" for c in result["likely_causes"])
    review = result.get("review") or {}
    if review:
        lines.append(f"Senior analyst review (paper read: {review.get('paper_text_status') or 'unknown'}; "
                     f"reviewed by {review.get('reviewed_by') or 'unknown model'}):")
        if review.get("assessment"):
            lines.append(f"  {review['assessment']}")
        for f in review.get("static_findings", []):
            lines.append(f"  - [{f['severity']}] {f['issue']} (automated check, line {f['line']}: {f['evidence']})")
        for f in review.get("findings", []):
            mark = "" if f.get("verified") else " UNVERIFIED (evidence not found; not used for the decision)"
            lines.append(f"  - [{f['severity']}, {f['category']}] {f['issue']} Evidence: \"{f['evidence']}\"{mark}")
            if f.get("fix"):
                lines.append(f"      fix: {f['fix']}")
    decision = result.get("decision") or {}
    if decision:
        lines.append(f"Decision: {decision.get('action', '')}"
                     + (f" -> {decision['target_agent']}" if decision.get("target_agent") else ""))
        if decision.get("reason"):
            lines.append(f"  Reason: {decision['reason']}")
        if decision.get("target_entries"):
            lines.append(f"  Data goal: {decision['target_entries']} entries")
        for instruction in decision.get("instructions") or []:
            lines.append(f"  - {instruction}")
        for note in review.get("decision_notes") or []:
            lines.append(f"  Check: {note}")
    return "\n".join(lines)


def _format_experiment_report(result: dict) -> str:
    gap = result.get("gap_points")
    lines = [
        "[Experiment Agent] Gap analysis (expert data scientist)",
        f"Status: {result.get('status', '')}",
        f"Mean gap to the paper: {gap if gap is not None else 'not measurable'} percentage points",
        f"Paper read: {result.get('paper_text_status') or 'unknown'}; analysis by "
        f"{result.get('proposed_by') or 'unknown model'}",
    ]
    if result.get("analysis"):
        lines.append(f"Analysis: {result['analysis']}")
    for h in result.get("hypotheses") or []:
        mark = "" if h.get("verified") else " UNVERIFIED"
        lines.append(f"  - {h.get('cause', '')} Evidence: \"{h.get('evidence', '')}\"{mark}"
                     + (f" Fix: {h['fix']}" if h.get("fix") else ""))
    decision = result.get("decision") or {}
    if decision:
        lines.append(f"Decision: {decision.get('action', '')}"
                     + (f" -> {decision['target_agent']}" if decision.get("target_agent") else ""))
        lines.extend(f"  - {i}" for i in decision.get("instructions") or [])
        if decision.get("target_entries"):
            lines.append(f"  Data goal: {decision['target_entries']} rows")
        if decision.get("reason"):
            lines.append(f"  Reason: {decision['reason']}")
        if decision.get("expected_effect"):
            lines.append(f"  Expected effect: {decision['expected_effect']}")
    for e in result.get("evidence") or []:
        lines.append(f"  Evidence: \"{e['quote']}\"" + ("" if e.get("verified") else " UNVERIFIED (not found)"))
    for note in result.get("decision_notes") or []:
        lines.append(f"  Check: {note}")
    if result.get("feedback_received"):
        lines.append("Master Agent feedback it received:")
        lines.extend(f"  - {f}" for f in result["feedback_received"])
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
            "Diagnose the failure and re-run Intake Agent with the diagnosis.",
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

        # The Parent Paper must clearly identify its dataset. If a ranked
        # alternative names a specific dataset but the chosen paper doesn't,
        # the ranking broke that rule. If no candidate names one, there is
        # no better choice to make -- that case is reported by the
        # Validation Agent's dataset_warning instead of blocking every run.
        if not row.get("dataset_named", True):
            named_alternatives = [
                r["title"] for r in recommendation["comparison_table"][1:] if r.get("dataset_named")
            ]
            if named_alternatives:
                blockers.append(_log_entry(
                    "Correction", "Validation Agent",
                    f"Recommended paper '{chosen.get('title', '')}' does not name a specific dataset "
                    f"(stated: '{chosen.get('dataset', '') or 'none'}'), while {len(named_alternatives)} "
                    f"ranked alternative(s) do, e.g. '{named_alternatives[0]}'.",
                    "Rejected",
                    "Re-rank so that a candidate which names a specific dataset is recommended.",
                ))

        if chosen.get("openalex_retracted"):
            blockers.append(_log_entry(
                "Correction", "Validation Agent",
                f"Recommended paper '{chosen.get('title', '')}' is marked as retracted in OpenAlex.",
                "Rejected",
                "Recommend a paper that has not been retracted.",
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
            "the target-fit check) across the Parent Paper and its ranked alternates. "
            f"Warnings: {'; '.join(result.get('warnings', [])) or 'none reported'}.",
            "Rejected",
            "Diagnose the failure and re-run Data Agent with the diagnosis.",
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
            "Diagnose the failure and re-run Cleaning Agent with the diagnosis.",
        )]
    return []


def _check_code_discovery(result: dict) -> list:
    """
    Code Discovery Agent never blocks the pipeline, regardless of status.
    Unlike Data Agent/Cleaning Agent, finding zero usable implementations
    is an explicitly valid outcome per CLAUDE.md's Preferred Implementation
    Order: option 6 is to reconstruct from the paper itself when no usable
    code can be located. There is nothing here for Master Agent to reject;
    a NOT_FOUND status just gets passed through to the Replication Agent.
    A raw exception during the stage is still caught and retried by
    _supervise itself, independent of this checker.
    """
    return []


def _check_replication(result: dict) -> list:
    """
    Unlike Code Discovery, Replication Agent IS blocked on anything other
    than a clean "OK" -- CLAUDE.md's Baseline Replication Rule means the
    pipeline genuinely needs a working, executed baseline before
    advancing to evaluation/experimentation. BLOCKED (install/entry-point/
    execution failure) and TIMEOUT both trigger Master Agent's normal
    diagnose-and-retry, which matters especially for the reconstruction
    path, where a retry re-generates the script with the diagnosis (the
    earlier traceback and what to change) in its prompt.
    """
    if result.get("status") != "OK":
        return [_log_entry(
            "Correction", "Replication Agent",
            f"Replication Agent did not produce a working baseline (status={result.get('status', '')}, "
            f"source={result.get('source', '')}). Warnings: {'; '.join(result.get('warnings', [])) or 'none reported'}.",
            "Rejected",
            "Diagnose the failure and re-run Replication Agent with the diagnosis.",
        )]
    return []


def _check_evaluation(result: dict) -> list:
    """
    Evaluation Agent never blocks the pipeline, regardless of status. A
    FAILED or INCONCLUSIVE comparison is itself useful research
    information, not an agent defect -- CLAUDE.md's Results Integrity
    policy is explicit that "a failed or imperfect reproduction is still
    useful research information." A raw exception during the stage is
    still caught and retried by _supervise itself, independent of this
    checker.
    """
    return []


def _check_experiment(result: dict) -> list:
    """
    Experiment Agent never blocks the pipeline: its output is a request (or
    "stop"), which Master Agent checks itself in _review_rerun_request. A
    raw exception during the stage is still diagnosed and retried by
    _supervise; if every attempt fails, the rerun loop ends and the closest
    baseline so far is kept.
    """
    return []


# --- Diagnosing a failed attempt ---

# What each agent is for, so the diagnosis call knows the agent's job.
AGENT_ROLES = {
    "Intake Agent": "turns the user's topic into a structured research request (domain, ML task, keywords, "
                    "search queries, target terms).",
    "Research Agent": "searches academic sources for at least 5 peer-reviewed papers from 2022 to now on the topic.",
    "Validation Agent": "verifies each paper (real, peer reviewed, 2022+, credible venue), ranks the qualifying "
                        "papers and recommends one Parent Paper.",
    "Data Agent": "reads the Parent Paper, finds its dataset, downloads it and converts it to a local CSV.",
    "Cleaning Agent": "merges the downloaded CSVs into one labelled master CSV and splits it into train/test/validation.",
    "Code Discovery Agent": "searches for the Parent Paper's own implementation code.",
    "Replication Agent": "runs the authors' code, or reconstructs a baseline script from the paper, in an isolated "
                         "virtualenv on the master CSV.",
    "Evaluation Agent": "compares the reproduced metrics with the paper's published results and reviews the "
                        "baseline as a senior data analyst.",
    "Experiment Agent": "reads the gap to the paper's published score and requests one rerun of an earlier "
                        "agent with new instructions, or stops.",
}


def _diagnose(agent_name: str, attempt: int, failure: str, report_text: str, earlier: list) -> dict:
    """
    One LLM call: why did this attempt fail, and what should the agent do
    differently next time? Returns {"cause", "guidance": [...], "note"}.
    "note" is what the retried agent sees (llm.supervisor_notes): the raw
    failure itself plus the diagnosis, so the agent gets the facts even
    when the diagnosis call fails (then only the failure is passed on).
    """
    from src import llm

    prompt = (
        f"You are the Master Agent supervising a multi-agent research pipeline. The {agent_name} "
        f"({AGENT_ROLES.get(agent_name, 'a pipeline agent')}) failed attempt {attempt}.\n\n"
        f"FAILURE (exception or failed checks):\n{failure[-DIAGNOSIS_TEXT_CHARS:]}\n\n"
        f"THE AGENT'S REPORT OF WHAT IT PRODUCED:\n{(report_text or 'none')[:DIAGNOSIS_TEXT_CHARS]}\n\n"
        "EARLIER DIAGNOSES FOR THIS STAGE (did not fix it):\n"
        + ("\n".join(f"- {e}" for e in earlier) or "none") + "\n\n"
        "Find the most likely cause, using only the text above -- do not invent facts. Then give the agent "
        "concrete instructions for its next attempt that are different from what already failed. The agent "
        "receives your instructions in its own prompts. Never tell it to relax a hard requirement: at least 5 "
        "peer-reviewed papers published 2022 or later, no invented papers, datasets, URLs, results or metrics, "
        "and no blogs, Medium or social media as academic sources. Master Agent's checks still apply.\n\n"
        'Return only JSON: {"cause": "one or two sentences", "guidance": ["instruction", "..."]} '
        "with at most 5 short instructions."
    )
    failure_line = " ".join(failure.split())[-FAILURE_NOTE_CHARS:]
    try:
        answer = llm.parse_json(llm.chat(prompt, timeout=120, agent="master_agent"), default={}) or {}
    except Exception as e:  # a failed diagnosis must not stop the retry
        answer = {"cause": f"Master Agent could not diagnose this failure ({type(e).__name__}: {e})."}
    cause = str(answer.get("cause") or "").strip() or "No cause identified."
    guidance = [str(g).strip() for g in answer.get("guidance") or [] if str(g).strip()][:5]
    note = f"Attempt {attempt} failed: {failure_line} Diagnosis: {cause}"
    if guidance:
        note += " Do this next time: " + " ".join(f"({i}) {g}" for i, g in enumerate(guidance, 1))
    return {"cause": cause, "guidance": guidance, "note": note}


def _supervise(agent_name: str, run_fn, checker_fn, report_fn, max_attempts: int = MAX_ATTEMPTS,
               attempt_offset: int = 0, notes: list = None):
    """
    Invoke run_fn() until its output passes checker_fn, at most
    max_attempts times. After each failed attempt (failed checks, or an
    exception) Master Agent diagnoses the failure (_diagnose) and re-runs
    the agent with every diagnosis so far added to the agent's LLM prompts
    (llm.supervisor_notes) -- so each retry knows what failed and what to
    change. Writes a saveable report after every attempt (with the
    diagnosis appended) and logs Correction/Approval entries to LOG_FILE.

    Returns (output, blockers). blockers is empty when the stage passed;
    non-empty means it failed max_attempts times and the pipeline should
    halt.

    attempt_offset numbers the attempts of a later rerun of the same agent
    (see _rerun_loop) after its earlier ones, so its reports and run
    database rows do not overwrite or duplicate them. notes are given to
    every attempt's LLM prompts from the start: a rerun request's
    instructions, relayed by Master Agent.
    """
    from src import llm

    output = None
    blockers = []
    notes = list(notes or [])  # the relayed request, then one per failed attempt (failure + diagnosis)
    last = attempt_offset + max_attempts

    for attempt in range(attempt_offset + 1, last + 1):
        print(f"\n[Master Agent] Running {agent_name} (attempt {attempt}/{last})...")
        slack_client.post_message(message_formatter.format_started(
            agent_name, f"Stage starting (attempt {attempt}/{last})."
            + (f" Master Agent passed {len(notes)} note(s): relayed instructions and/or diagnoses." if notes else "")
        ))

        try:
            with llm.supervisor_notes(notes):
                output = run_fn()
        except Exception as e:
            slack_client.post_error_message(message_formatter.format_finished(
                agent_name, "Failed", f"Raised an exception: {e}",
            ))
            output = None
            failure = f"{agent_name} raised an exception: {e}\n{traceback.format_exc()}"
            report_text = f"{agent_name} raised an exception:\n{traceback.format_exc()}"
            blockers = [_log_entry(
                "Correction", agent_name,
                f"{agent_name} raised an exception on attempt {attempt}: {e}.",
                "Retrying" if attempt < last else "Rejected",
                "Diagnose the failure and re-run the agent." if attempt < last else "Halting pipeline.",
            )]
        else:
            report_text = report_fn(output)
            slack_client.post_message(message_formatter.format_finished(
                agent_name, "Completed", _summarize_for_slack(report_text),
            ))
            blockers = checker_fn(output)
            if not blockers:
                report_path = _write_report(agent_name, attempt, report_text)
                approval = _log_entry(
                    "Approval", agent_name,
                    f"{agent_name} output passed all checks on attempt {attempt}. Report saved to {report_path}.",
                    "Approved",
                    "Advance to next stage.",
                )
                _print_entry(approval)
                _append_log([approval])
                run_store.record_stage(_current_run_id, agent_name, attempt, "Approved", output, report_text)
                return output, []
            failure = "\n".join(b["issue"] for b in blockers)

        for b in blockers:
            _print_entry(b)
        _append_log(blockers)

        if attempt < last:
            diagnosis = _diagnose(agent_name, attempt, failure, report_text, notes)
            notes.append(diagnosis["note"])
            report_text += (f"\n\n--- Master Agent diagnosis ---\nCause: {diagnosis['cause']}\n"
                            + "\n".join(f"- {g}" for g in diagnosis["guidance"]))
            report_path = _write_report(agent_name, attempt, report_text)
            run_store.record_stage(_current_run_id, agent_name, attempt, "Retrying", output, report_text)
            retry_note = _log_entry(
                "Correction", agent_name,
                f"{agent_name} failed attempt {attempt}. Diagnosis: {diagnosis['cause']} "
                f"Report saved to {report_path}.",
                "Retrying",
                f"Re-running {agent_name} (attempt {attempt + 1}/{last}) with these instructions: "
                + ("; ".join(diagnosis["guidance"]) or "the failure details only (no diagnosis available)."),
            )
            _print_entry(retry_note)
            _append_log([retry_note])
        else:
            report_path = _write_report(agent_name, attempt, report_text)
            run_store.record_stage(_current_run_id, agent_name, attempt, "Rejected", output, report_text)
            final = _log_entry(
                "Correction", agent_name,
                f"{agent_name} still failing after {max_attempts} error(s), each diagnosed and retried. "
                f"See the issues above and the reports in {REPORT_DIR}.",
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

        previous = load_state().get("dataset") or {}
        new_target = (
            message_formatter.parse_target_from_detail(request["detail"])
            or max(previous.get("target_entries") or 0, previous.get("total_entries") or 0)
            + DATA_REQUEST_TARGET_INCREMENT
        )

        slack_client.post_message(message_formatter.format_started(
            "Master Agent",
            f"Relaying {request['sender']}'s request ({request['reason']}): "
            f"re-running Data Agent with target={new_target}.",
        ))

        new_dataset_result, blockers = _supervise(
            "Data Agent",
            lambda: data_agent.run(
                parent_paper, ranked_pool, target_entries=new_target,
                search_query=load_state().get("project_topic", ""),
                target_terms=load_state().get("target_terms", []),
                mode="expand", existing_datasets=previous.get("selected_datasets") or [],
            ),
            _check_data,
            _format_data_report,
        )
        if blockers:
            acted = {"dataset_result": None, "cleaning_result": None, "blockers": blockers}
            continue

        new_cleaning_result, blockers = _supervise(
            "Cleaning Agent",
            lambda: cleaning_agent.run(new_dataset_result["selected_datasets"],
                                       target_terms=load_state().get("target_terms", []),
                                       target_rows=new_dataset_result["target_entries"]),
            _check_cleaning,
            _format_cleaning_report,
        )
        update_state({
            "dataset": {
                "selected_datasets": new_dataset_result["selected_datasets"],
                "target_entries": new_dataset_result["target_entries"],
                "data_plan": new_dataset_result["data_plan"],
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
                "train_csv_path": new_cleaning_result["train_csv_path"],
                "test_csv_path": new_cleaning_result["test_csv_path"],
                "validation_csv_path": new_cleaning_result["validation_csv_path"],
                "split_rows": new_cleaning_result["split_rows"],
                "split_seed": new_cleaning_result["split_seed"],
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




# --- The driver: running agents from their handoff() requests ---
#
# No agent runs another agent. When an agent finishes, its module's
# handoff(result, context) returns its request to Master Agent: the agent
# to run next and what to pass on (src/orchestration/handoff.py). Master
# Agent checks the request (_review_handoff), runs the requested agent
# under _supervise, and asks that agent for its own handoff -- until an
# agent requests nothing more (the Experiment Agent's "stop").
#
# Reruns go through the same path. After the first evaluation, the
# Evaluation Agent (validity fixes such as data leakage, or too little
# data) and the Experiment Agent (closing the gap to the Parent Paper's
# published score) hand off a rerun request for an earlier agent, with new
# knowledge. Master Agent checks it (_review_rerun_request), including
# whether it is redundant, opens a rerun (the files it can overwrite are
# snapshotted), runs the requested agent with the request's instructions,
# and then follows each agent's own handoff until the Evaluation Agent has
# evaluated the new baseline, which closes the rerun. A declined request
# goes back to the Experiment Agent with the reason, so it can look for a
# different solution. A rerun that fails is undone from the snapshot, and
# an Experiment Agent rerun that moves the score further from the paper is
# undone too, so the current baseline is always the closest one so far.

MAX_PIPELINE_RERUNS = 10    # reruns acted on per pipeline run
REDUNDANT_STREAK = 3        # same agent rerun this many times in a row with no gain -> the next is redundant
NO_PROGRESS_LIMIT = 4       # this many reruns in a row with no gain -> Master Agent ends the loop
MAX_DECLINED_IN_A_ROW = 3   # this many declined requests in a row -> Master Agent ends the loop
GAP_TOLERANCE_POINTS = 0.01

# The results a rerun can replace; restored when the rerun is undone.
RERUN_RESULT_KEYS = ("dataset_result", "cleaning_result", "code_discovery_result", "replication_result",
                     "evaluation_result")

# Why a rerun is undone when one of its stages fails.
RERUN_FAILURE = {
    "Data Agent": "the Data Agent found no usable dataset",
    "Cleaning Agent": "the Cleaning Agent could not merge the data",
    "Code Discovery Agent": "the Code Discovery Agent failed",
    "Replication Agent": "the rerun baseline did not execute",
    "Evaluation Agent": "the new baseline could not be evaluated",
}

# Each agent's checks and report (see _supervise), by name: they are looked
# up when the stage runs.
STAGE_CHECKS = {
    "Intake Agent": ("_check_intake", "_format_intake_report"),
    "Research Agent": ("_check_research", "_format_research_report"),
    "Validation Agent": ("_check_validation_and_selection", "_format_validation_report"),
    "Data Agent": ("_check_data", "_format_data_report"),
    "Cleaning Agent": ("_check_cleaning", "_format_cleaning_report"),
    "Code Discovery Agent": ("_check_code_discovery", "_format_code_discovery_report"),
    "Replication Agent": ("_check_replication", "_format_replication_report"),
    "Evaluation Agent": ("_check_evaluation", "_format_evaluation_report"),
    "Experiment Agent": ("_check_experiment", "_format_experiment_report"),
}

# After these stages of the first pass, Master Agent reads Slack for a
# relayed Data Agent request (check_agent_requests).
SLACK_RELAY_AFTER = ("Validation Agent", "Data Agent", "Cleaning Agent", "Code Discovery Agent",
                     "Replication Agent")


def _agent_module(agent_name: str):
    """'Code Discovery Agent' -> the src.agents.code_discovery_agent module."""
    import importlib

    return importlib.import_module("src.agents." + agent_name.lower().replace(" ", "_"))


def _snapshot_files(rerun_no: int, paths: list) -> dict:
    """Copy the files a rerun overwrites; {original: copy}."""
    backup = ROUND_BACKUP_DIR / f"rerun_{rerun_no}"
    saved = {}
    for index, path in enumerate(p for p in paths if p):
        path = Path(path)
        if path.is_file():
            backup.mkdir(parents=True, exist_ok=True)
            copy = backup / f"{index}_{path.name}"
            shutil.copy2(path, copy)
            saved[str(path)] = str(copy)
    return saved


def _restore_files(saved: dict) -> None:
    for original, copy in saved.items():
        try:
            shutil.copy2(copy, original)
        except OSError as e:
            print(f"[Master Agent] Could not restore {original}: {e}")


def _baseline_files(cleaning_result: dict, replication_result: dict) -> list:
    paths = [cleaning_result.get(k) for k in ("master_csv_path", "train_csv_path", "test_csv_path",
                                              "validation_csv_path")]
    paths += [replication_result.get("stdout_path"), replication_result.get("stderr_path")]
    replication_dir = replication_result.get("replication_dir")
    if replication_dir:
        recon = Path(replication_dir) / "reconstructed"
        paths += [str(recon / "train.py"), str(recon / "RECONSTRUCTION_NOTES.md"),
                  str(Path(replication_dir) / "ENVIRONMENT.md")]
    return paths


def _metrics_summary(evaluation_result: dict) -> str:
    return ", ".join(f"{m['metric']}={m['reproduced_value']}" for m in evaluation_result.get("metrics", [])
                     if m.get("reproduced_value") is not None) or "no metrics"


def _is_rerun_request(request: dict) -> bool:
    """A forward handoff runs the next agent ("run_agent"); any other action
    (rerun_data_agent, revise_replication, rerun_agent) asks for an earlier
    agent to run again."""
    return request.get("action", "run_agent") != "run_agent"


def _review_handoff(request: dict, target: str, ctx: dict, drive: dict) -> str:
    """Master Agent's own check of a forward handoff. Returns "" when it is
    approved, else the reason it is declined: the target must be an agent
    of this pipeline, every agent it depends on must have an approved
    result (no skipped workflow states), and an agent that already ran runs
    again only inside an approved rerun (the Experiment Agent excepted: it
    is asked again after every rerun)."""
    from src.orchestration.handoff import PREREQUISITES, RESULT_KEY

    if target not in PREREQUISITES:
        return f"'{request.get('target_agent')}' is not an agent of this pipeline."
    missing = [p for p in PREREQUISITES[target] if ctx.get(RESULT_KEY[p]) is None]
    if missing:
        return f"the {target} cannot run before the {', '.join(missing)} has an approved result."
    if ctx.get(RESULT_KEY[target]) is not None and not drive["rerun"] and target != "Experiment Agent":
        return (f"the {target} already ran. An earlier agent runs again only through a rerun request from the "
                "Evaluation or Experiment Agent, checked by Master Agent.")
    return ""


def _review_rerun_request(request: dict, history: list, ctx: dict) -> str:
    """Master Agent's own check of a rerun request. Returns "" when it is
    approved, else the reason it is declined. Master Agent does not redo
    the sender's analysis; it checks that the request is actionable,
    supported by evidence, and not redundant."""
    from src.agents.experiment_agent import RERUNNABLE_AGENTS, canonical_agent

    target = canonical_agent(request.get("target_agent"))
    instructions = request.get("instructions") or []
    goal = request.get("goal") or {}
    if not target:
        return f"'{request.get('target_agent')}' is not an agent of this pipeline."
    if target not in RERUNNABLE_AGENTS:
        return (f"a rerun of the {target} could replace the Parent Paper, and a gap to a different paper is not a "
                "smaller gap. Ask for the Data, Cleaning, Code Discovery, Replication or Evaluation Agent instead.")
    if not instructions and not goal:
        return "it gives no instructions or goal, and a rerun without new knowledge repeats the last result."
    if not request.get("evidence"):
        return "it gives no verified evidence for the problem it would fix."
    for previous in history:
        if previous["target_agent"] == target and previous["instructions"] == instructions \
                and previous["goal"] == goal:
            return f"the same request was already run (rerun {previous['rerun']}: {previous.get('outcome', '')})."

    streak = []
    for previous in reversed(history):
        if previous["target_agent"] != target:
            break
        streak.append(previous)
    if len(streak) >= REDUNDANT_STREAK and not any(h.get("improved") for h in streak):
        gaps = ", ".join(str(h.get("gap_after")) for h in reversed(streak))
        return (f"redundant: the {target} was rerun {len(streak)} times in a row without bringing the score closer "
                f"to the paper (gap after each run: {gaps} points). Another run of it is not necessary. Look for a "
                "different solution: another agent, or a different kind of change.")

    if target == "Data Agent" and goal.get("target_entries") is not None:
        goal_rows = goal["target_entries"]
        rows_used = ctx["cleaning_result"].get("rows_total") or 0
        if not isinstance(goal_rows, int) or goal_rows <= rows_used:
            return f"the data goal ({goal_rows}) is not above the {rows_used} rows used."
        for previous in history:
            if previous.get("found_no_data") and goal_rows <= (previous["goal"].get("target_entries") or 0):
                return (f"an earlier request for {previous['goal']['target_entries']} rows (rerun "
                        f"{previous['rerun']}) found no usable new data.")
    if target == "Replication Agent" and ctx["replication_result"].get("source") != "reconstructed":
        return ("the baseline is the authors' cloned code, which is preserved, not revised. Change the data, or "
                "search for the code again, instead.")
    return ""


def _loop_end_reason(drive: dict) -> str:
    """Why Master Agent ends the rerun loop now, or "" to go on."""
    history = drive["history"]
    if len(history) >= MAX_PIPELINE_RERUNS:
        return f"the limit of {MAX_PIPELINE_RERUNS} reruns per pipeline run was reached"
    recent = history[-NO_PROGRESS_LIMIT:]
    if len(recent) == NO_PROGRESS_LIMIT and not any(h.get("improved") for h in recent):
        return (f"the last {NO_PROGRESS_LIMIT} reruns did not bring the score closer to the paper, so more reruns "
                "are not expected to help")
    return ""


def _ask_experiment_agent(reason: str) -> list:
    """Master Agent's own request for the Experiment Agent's next analysis,
    after a declined or undone rerun."""
    from src.orchestration.handoff import request

    return [request("master_agent", "Experiment Agent", {}, reason)]


# --- Project state for each stage ---

def _stage_state(agent_name: str, output: dict) -> dict:
    """The project-state fields an approved stage writes (logs/project_state.json)."""
    if agent_name == "Data Agent":
        return {"dataset": {key: output.get(key) for key in ("selected_datasets", "target_entries", "data_plan",
                                                         "total_entries", "target_met", "is_combined")}}
    if agent_name == "Cleaning Agent":
        return {"cleaning": {key: output.get(key) for key in (
            "master_csv_path", "datasets_merged", "rows_total", "target_met", "conflicts", "train_csv_path",
            "test_csv_path", "validation_csv_path", "split_rows", "split_seed")}}
    if agent_name == "Code Discovery Agent":
        return {"code_repository": {key: output.get(key) for key in ("status", "repository", "source_stage")}}
    if agent_name == "Replication Agent":
        return {"baseline_status": output.get("status"),
                "baseline_results": {key: output.get(key) for key in (
                    "source", "repo_url", "commit_sha", "entry_point", "exit_code", "duration_seconds",
                    "inferred_assumptions")}}
    if agent_name == "Evaluation Agent":
        return {"evaluation_result": {key: output.get(key) for key in ("status", "metrics", "likely_causes")}}
    return {}


def _after_stage(agent_name: str, output, ctx: dict, drive: dict) -> None:
    """Keep an approved first-pass result in ctx and project state. Inside a
    rerun, project state is written only when the rerun is kept
    (_close_rerun)."""
    from src.orchestration import dataset_cache
    from src.orchestration.handoff import AGENT_PHASE
    from src.orchestration.state_manager import update_state

    if agent_name == "Intake Agent":
        output["project_topic"] = ctx["topic"]
        ctx["target_terms"] = output.get("target_terms", [])
        update_state({"project_topic": ctx["topic"], "research_question": output.get("domain", ""),
                      "target_terms": ctx["target_terms"], "phase": AGENT_PHASE[agent_name]})
    elif agent_name == "Research Agent":
        update_state({"papers": output, "phase": AGENT_PHASE[agent_name]})
    elif agent_name == "Validation Agent":
        ctx["parent_paper"] = output["recommended_parent_paper"]
        ctx["ranked_pool"] = output["ranked_pool"]
        update_state({"papers": output["all"], "parent_paper": ctx["parent_paper"],
                      "parent_paper_approved": True, "phase": "MASTER_APPROVAL"})
        dataset_cache.cache_validated_papers(ctx["ranked_pool"])
    elif not drive["rerun"] and agent_name != "Experiment Agent":
        update_state({**_stage_state(agent_name, output), "phase": AGENT_PHASE[agent_name]})

    if agent_name in SLACK_RELAY_AFTER and not drive["evaluated"]:
        # A Data Agent request only becomes actionable once parent_paper/
        # ranked_pool exist -- i.e. from Validation on. Checking any earlier
        # would mark a not-yet-actionable request as "seen" (advancing the
        # cursor) before it ever had context to act on.
        _relay_and_apply(ctx, ctx["ranked_pool"], drive["floor_ts"])


def _halt(agent_name: str, output, blockers: list, ctx: dict, drive: dict) -> None:
    """A first-pass stage failed for good: record where the pipeline stopped."""
    from src.orchestration.handoff import AGENT_PHASE
    from src.orchestration.state_manager import update_state

    phase = AGENT_PHASE.get(agent_name, "")
    if agent_name == "Validation Agent":
        update_state({"papers": output["all"] if output else ctx.get("papers"), "parent_paper": None,
                      "parent_paper_approved": False, "phase": phase})
    elif agent_name == "Replication Agent":
        update_state({"baseline_status": output["status"] if output else "", "phase": phase})
    elif agent_name not in ("Intake Agent", "Research Agent"):
        update_state({"phase": phase})
    drive.update(phase=phase, blockers=blockers)


# --- Running one requested agent ---

def _stage_call(target: str, request: dict, ctx: dict, drive: dict):
    """The call that runs `target` for this request: its inputs come from the
    handoff (falling back to ctx), and inside a rerun the requested agent
    also gets the request's goal and instructions."""
    module = _agent_module(target)
    inputs = request.get("inputs") or {}
    rerun = drive["rerun"]
    first = bool(rerun) and rerun["target"] == target  # the agent the rerun was requested for
    goal = (rerun["request"].get("goal") or {}) if rerun else {}
    parent = ctx.get("parent_paper")

    if target == "Intake Agent":
        topic = inputs.get("topic") or ctx["topic"]
        return lambda: module.run(topic)
    if target == "Research Agent":
        structured = inputs.get("structured") or ctx["structured"]
        return lambda: module.run(structured)
    if target == "Validation Agent":
        papers = inputs.get("papers") or ctx["papers"]
        return lambda: module.run(papers)
    if target == "Data Agent":
        if first:
            expand = bool(goal.get("target_entries"))
            existing = ctx["dataset_result"]["selected_datasets"] if expand else None
            return lambda: module.run(parent, ctx["ranked_pool"], target_entries=goal.get("target_entries"),
                                      search_query=ctx["topic"], target_terms=ctx["target_terms"],
                                      mode="expand" if expand else "parent_only", existing_datasets=existing)
        parent = inputs.get("parent_paper") or parent
        pool = inputs.get("ranked_pool") or ctx["ranked_pool"]
        return lambda: module.run(parent, pool, search_query=ctx["topic"], target_terms=ctx["target_terms"])
    if target == "Cleaning Agent":
        datasets = inputs.get("selected_datasets") or ctx["dataset_result"]["selected_datasets"]
        target_rows = goal.get("target_entries") or inputs.get("target_rows",
                                                               ctx["dataset_result"].get("target_entries"))
        return lambda: module.run(datasets, target_terms=ctx["target_terms"], target_rows=target_rows)
    if target == "Code Discovery Agent":
        if first:  # a new search: the cached result is the one that did not help
            return lambda: module.run(parent, use_cache=False)
        return lambda: module.run(parent)
    if target == "Replication Agent":
        previous = ctx.get("replication_result") or {}
        revision = inputs.get("revision")
        if first:
            revision = {"round": rerun["record"]["rerun"], "instructions": rerun["record"]["instructions"],
                        "previous_inferred_assumptions": previous.get("inferred_assumptions") or []}
        elif rerun and revision and revision.get("reuse_script"):
            # New data: the SAME script runs on it, so only the data changes.
            revision = {**revision, "round": rerun["record"]["rerun"],
                        "previous_inferred_assumptions": previous.get("inferred_assumptions") or [],
                        "previous_generated_by": previous.get("generated_by") or []}
        kwargs = {"revision": revision} if rerun or revision else {}
        code, cleaning = ctx["code_discovery_result"], ctx["cleaning_result"]
        return lambda: module.run(parent, code, cleaning, **kwargs)
    if target == "Evaluation Agent":
        args = (parent, ctx["replication_result"], ctx["cleaning_result"], ctx["dataset_result"])
        if rerun:
            args += (drive["history"] + [dict(rerun["record"], outcome="(being evaluated)")],)
        return lambda: module.run(*args)
    if target == "Experiment Agent":
        args = (parent, ctx["replication_result"], ctx["evaluation_result"], ctx["cleaning_result"],
                ctx["dataset_result"], ctx["code_discovery_result"], list(drive["history"]),
                list(drive["feedback"]))
        return lambda: module.run(*args)
    raise ValueError(f"Master Agent has no call for the {target}.")


def _run_stage(target: str, request: dict, ctx: dict, drive: dict):
    """Run the requested agent under Master Agent's supervision. Returns
    (output, blockers), as _supervise does."""
    rerun = drive["rerun"]
    if target == "Experiment Agent":
        offset, notes = len(drive["experiment_results"]) * MAX_ATTEMPTS, None
    elif rerun:
        offset = rerun["record"]["rerun"] * MAX_ATTEMPTS
        notes = rerun["relay"] if rerun["target"] == target else None
    else:
        offset, notes = 0, None
    checker, report = (globals()[name] for name in STAGE_CHECKS[target])
    return _supervise(target, _stage_call(target, request, ctx, drive), checker, report,
                      attempt_offset=offset, notes=notes)


# --- Opening and closing a rerun ---

def _open_rerun(request: dict, target: str, sender: str, ctx: dict, drive: dict) -> None:
    """Snapshot the baseline and start a rerun record; the requested agent
    runs next, then each agent's handoff leads to the Evaluation Agent."""
    from src.agents.experiment_agent import score_gap

    rerun_no = len(drive["history"]) + 1
    goal = request.get("goal") or {}
    instructions = request.get("instructions") or []
    record = {"rerun": rerun_no, "requested_by": sender, "target_agent": target, "action": request.get("action", ""),
              "goal": goal, "instructions": instructions, "reason": request.get("summary", ""),
              "status_before": ctx["evaluation_result"].get("status"),
              "metrics_before": _metrics_summary(ctx["evaluation_result"]),
              "gap_before": score_gap(ctx["evaluation_result"]), "applied": False, "improved": False}
    relay = [f"Master Agent relays a request from the {sender} (rerun {rerun_no}): "
             + (" ".join(f"({i}) {text}" for i, text in enumerate(instructions, 1)) or "no instructions.")
             + (f" Goal: {goal}." if goal else "") + f" Reason: {request.get('summary', '')}"]
    drive["rerun"] = {
        "request": request, "target": target, "sender": sender, "record": record, "relay": relay,
        "saved": _snapshot_files(rerun_no, _baseline_files(ctx["cleaning_result"], ctx["replication_result"])),
        "before": {key: ctx.get(key) for key in RERUN_RESULT_KEYS},
    }

    evidence = "; ".join(e.get("evidence", "") or e.get("issue", "") for e in request.get("evidence") or [])
    entry = _log_entry("Correction", target,
                       f"Request from the {sender} (rerun {rerun_no}): {request.get('summary', '')} "
                       f"Evidence: {evidence or 'none listed'}.", "Approved request",
                       f"Rerunning the {target} with these instructions: "
                       + ("; ".join(instructions) or str(goal))
                       + ". Each agent then hands off to the next until the new baseline is evaluated.")
    _print_entry(entry)
    _append_log([entry])


def _end_rerun(drive: dict) -> None:
    """File the closed rerun's record (kept or undone)."""
    from src.orchestration.state_manager import update_state

    record = drive["rerun"]["record"]
    drive["history"].append(record)
    update_state({"reruns": drive["history"]})
    if not record["applied"]:
        drive["feedback"].append(f"Rerun {record['rerun']} of the {record['target_agent']} was not kept: "
                                 f"{record['outcome']}")
    drive["rerun"] = None


def _undo_rerun(reason: str, ctx: dict, drive: dict, decision: str = "Rejected") -> None:
    """Restore the files and results the rerun replaced; the previous
    baseline stays the current one."""
    rerun = drive["rerun"]
    _restore_files(rerun["saved"])
    ctx.update(rerun["before"])
    rerun["record"]["outcome"] = f"{reason}; the previous baseline was kept."
    entry = _log_entry("Correction", rerun["target"],
                       f"Rerun {rerun['record']['rerun']} (requested by the {rerun['sender']}): {reason}.",
                       decision, "Previous baseline restored; the requesting agent is told the result.")
    _print_entry(entry)
    _append_log([entry])
    _end_rerun(drive)


def _close_rerun(ctx: dict, drive: dict) -> bool:
    """The Evaluation Agent has evaluated the rerun's baseline. Keep it, or
    undo an Experiment Agent rerun that moved the score further from the
    paper. Returns True when it is kept."""
    from src.agents.experiment_agent import score_gap
    from src.orchestration.state_manager import update_state

    rerun = drive["rerun"]
    record = rerun["record"]
    gap_before, gap_after = record["gap_before"], score_gap(ctx["evaluation_result"])
    record.update(gap_after=gap_after, status_after=ctx["evaluation_result"].get("status"),
                  metrics_after=_metrics_summary(ctx["evaluation_result"]))
    change = (f"gap {gap_before} -> {gap_after} points; overall {record['status_before']} -> "
              f"{record['status_after']}; metrics {record['metrics_before']} -> {record['metrics_after']}")
    widened = gap_before is not None and (gap_after is None or gap_after > gap_before + GAP_TOLERANCE_POINTS)
    if widened and rerun["sender"] == "Experiment Agent":
        # Validity fixes (Evaluation Agent requests) are kept even when the
        # score moves away: a leaky score is not a real score.
        _undo_rerun(f"the score moved further from the paper ({change})", ctx, drive, decision="Reverted")
        return False

    record.update(applied=True, outcome=change,
                  improved=gap_after is not None and (gap_before is None
                                                      or gap_after < gap_before - GAP_TOLERANCE_POINTS))
    state = {}
    for agent_name, key in (("Data Agent", "dataset_result"), ("Cleaning Agent", "cleaning_result"),
                            ("Code Discovery Agent", "code_discovery_result"),
                            ("Replication Agent", "replication_result"), ("Evaluation Agent", "evaluation_result")):
        if ctx.get(key) is not rerun["before"].get(key):
            state.update(_stage_state(agent_name, ctx[key]))
    update_state(state)
    done = _log_entry("Approval", rerun["target"],
                      f"Rerun {record['rerun']} (requested by the {rerun['sender']}) kept: {change}.",
                      "Approved", "The new evaluation decides the next step.")
    _print_entry(done)
    _append_log([done])
    _end_rerun(drive)
    return True


def _decline(request: dict, target: str, sender: str, reason: str, ctx: dict, drive: dict) -> list | None:
    """Act on a declined request. Returns the next requests, or None when
    the drive ends. In the first pass a declined handoff is a wiring fault
    and halts the pipeline; inside a rerun it undoes the rerun; in the
    rerun loop the reason goes back to the Experiment Agent."""
    rerun_request = _is_rerun_request(request)
    asked = (f"request to rerun the {target}" if rerun_request
             else f"handoff to the {target or request.get('target_agent')}")
    if drive["rerun"]:
        _undo_rerun(f"the {sender}'s {asked} was declined: {reason}", ctx, drive)
        return _ask_experiment_agent("A rerun was undone; the closest baseline so far needs the next analysis.")
    if not drive["evaluated"]:
        entry = _log_entry("Correction", sender, f"The {sender}'s {asked} was declined: {reason}",
                           "Rejected", "Halting pipeline; manual review required.")
        _print_entry(entry)
        _append_log([entry])
        _halt(sender, None, [entry], ctx, drive)
        return None

    redundant = reason.startswith("redundant")
    entry = _log_entry("Correction", sender, f"{asked[0].upper()}{asked[1:]} declined: {reason}",
                       "Rejected (redundant)" if redundant else "Rejected",
                       f"The {sender} must look for a different solution or prompt."
                       + ("" if sender == "Experiment Agent" else " The Experiment Agent receives this reason."))
    _print_entry(entry)
    _append_log([entry])
    drive["feedback"].append(f"Master Agent declined the {sender}'s {asked} "
                             f"({'; '.join(request.get('instructions') or []) or 'no instructions'}): {reason}")
    drive["declined_in_a_row"] += 1
    if drive["declined_in_a_row"] >= MAX_DECLINED_IN_A_ROW:
        drive["end_reason"] = f"{MAX_DECLINED_IN_A_ROW} requests in a row were declined"
        return None
    return _ask_experiment_agent(f"Master Agent declined a request: {reason}")


def _new_drive(floor_ts: str = None, evaluated: bool = False) -> dict:
    return {"history": [], "experiment_results": [], "feedback": [], "declined_in_a_row": 0, "rerun": None,
            "evaluated": evaluated, "end_reason": "", "phase": None, "blockers": [], "floor_ts": floor_ts}


def _drive(ctx: dict, queue: list, drive: dict, sender: str = "Master Agent") -> None:
    """
    Run agents from handoff requests until no agent requests anything more,
    a first-pass stage fails for good (drive["blockers"]), or Master Agent
    ends the rerun loop (drive["end_reason"]). ctx holds the run's topic and
    every agent's latest approved result (handoff.RESULT_KEY), and is
    updated in place. sender is the agent whose requests `queue` holds.
    """
    from src.agents.experiment_agent import canonical_agent
    from src.orchestration.handoff import RESULT_KEY

    queue = list(queue)
    while True:
        if not queue:
            if drive["rerun"]:  # a rerun must end with the new baseline evaluated
                queue = _decline({"target_agent": "", "action": "run_agent"}, "", sender,
                                 f"the {sender} requested no next agent before the new baseline was evaluated",
                                 ctx, drive)
                if queue is None:
                    break
                sender = "Master Agent"
                continue
            if not drive["evaluated"]:
                _decline({"target_agent": "", "action": "run_agent"}, "", sender,
                         f"the {sender} requested no next agent before the baseline was evaluated", ctx, drive)
            break

        request = queue.pop(0)
        target = canonical_agent(request.get("target_agent")) or str(request.get("target_agent") or "")
        if drive["evaluated"] and not drive["rerun"]:
            drive["end_reason"] = _loop_end_reason(drive)
            if drive["end_reason"]:
                break

        if _is_rerun_request(request):
            reason = ("a rerun can be requested only after the baseline is evaluated, and not inside another rerun"
                      if drive["rerun"] or not drive["evaluated"]
                      else _review_rerun_request(request, drive["history"], ctx))
        else:
            reason = _review_handoff(request, target, ctx, drive)
        if reason:
            queue = _decline(request, target, sender, reason, ctx, drive)
            if queue is None:
                break
            sender = "Master Agent"
            continue

        if _is_rerun_request(request):
            drive["declined_in_a_row"] = 0
            _open_rerun(request, target, sender, ctx, drive)
        else:
            print(f"\n[Master Agent] Handoff approved: {sender} -> {target}. {request.get('summary', '')}")

        output, blockers = _run_stage(target, request, ctx, drive)
        if blockers:
            if drive["rerun"]:
                _undo_rerun(RERUN_FAILURE[target], ctx, drive)
                queue, sender = _ask_experiment_agent("A rerun failed; the closest baseline so far needs the next "
                                                      "analysis."), "Master Agent"
                continue
            if target == "Experiment Agent":
                drive["end_reason"] = "the Experiment Agent could not complete its analysis"
                break
            _halt(target, output, blockers, ctx, drive)
            break

        ctx[RESULT_KEY[target]] = output
        _after_stage(target, output, ctx, drive)
        sender = target

        rerun = drive["rerun"]
        if target == "Data Agent" and rerun and rerun["target"] == target \
                and (rerun["request"].get("goal") or {}).get("target_entries") \
                and output["total_entries"] <= rerun["before"]["dataset_result"]["total_entries"]:
            rerun["record"]["found_no_data"] = True
            _undo_rerun("the Data Agent found no usable new data", ctx, drive)
            queue, sender = _ask_experiment_agent("A rerun found no new data; the closest baseline so far needs "
                                                  "the next analysis."), "Master Agent"
            continue
        if target == "Evaluation Agent":
            if rerun and not _close_rerun(ctx, drive):
                queue, sender = _ask_experiment_agent("A rerun was undone; the closest baseline so far needs the "
                                                      "next analysis."), "Master Agent"
                continue
            drive["evaluated"] = True
        if target == "Experiment Agent":
            drive["experiment_results"].append(output)

        requests = _agent_module(target).handoff(output, ctx)
        if target == "Experiment Agent" and not requests:
            notes = " ".join(output.get("decision_notes") or [])
            reason = (output.get("decision") or {}).get("reason", "")
            drive["end_reason"] = "the Experiment Agent requested no further rerun" + (
                f": {reason or notes}" if reason or notes else "")
            break
        queue = list(requests) + queue

    if drive["evaluated"] and not drive["blockers"]:
        end = _log_entry("Approval", "Experiment Agent",
                         f"Master Agent ends the rerun loop after {len(drive['history'])} rerun(s): "
                         f"{drive['end_reason']}.", "Approved", "Keep the closest baseline so far and continue.")
        _print_entry(end)
        _append_log([end])


def _rerun_loop(ctx: dict) -> dict:
    """
    Drive the pipeline on from an evaluated baseline: start from the
    Evaluation Agent's own handoff (a validity rerun, or the Experiment
    Agent) and act on every request that follows (see _drive). ctx holds the
    run's parent_paper, ranked_pool, topic, target_terms and the current
    dataset/cleaning/code_discovery/replication/evaluation results; it is
    updated in place by every rerun that is kept.

    Returns {"reruns", "experiment_results", "end_reason"}.
    """
    from src.agents import evaluation_agent

    drive = _new_drive(evaluated=True)
    _drive(ctx, evaluation_agent.handoff(ctx["evaluation_result"], ctx), drive, sender="Evaluation Agent")
    return {"reruns": drive["history"], "experiment_results": drive["experiment_results"],
            "end_reason": drive["end_reason"]}


def run_pipeline(topic: str) -> dict:
    """
    Run the pipeline (see _run_stages) inside one run database record
    (src/orchestration/run_store.py): every agent attempt and the files it
    produced are stored as they happen, and the run is closed with its final
    status even if a stage raises.
    """
    global _current_run_id
    from src.config import LLM_PROVIDER

    _current_run_id = run_store.start_run(topic, llm_provider=LLM_PROVIDER)
    try:
        result = _run_stages(topic)
    except BaseException:
        run_store.finish_run(_current_run_id, "crashed")
        _current_run_id = None
        raise

    if _current_run_id:
        print(f"\n[Run Store] Run {_current_run_id} saved to {run_store.DB_FILE}")
    run_store.finish_run(
        _current_run_id, "success" if result["success"] else "halted", result["phase"] or "",
        parent_paper=result["parent_paper"], result=result,
    )
    result["run_id"] = _current_run_id
    _current_run_id = None
    return result


def _run_stages(topic: str) -> dict:
    """
    Drive the full pipeline from handoff requests (see _drive). Master Agent
    asks for the Intake Agent; from then on each agent's handoff() requests
    the next one -- Intake -> Research -> Validation & Selection -> Data
    Agent (Parent Paper's own dataset only) -> Cleaning Agent -> Code
    Discovery Agent -> Replication Agent -> Evaluation Agent -> the
    Evaluation and Experiment Agents' rerun requests -- and Master Agent
    checks every request before it runs the agent. Every stage is
    supervised: Master Agent checks its output, diagnoses each failure and
    retries with the diagnosis, and halts (with a logged rejection) after
    MAX_ATTEMPTS errors.

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
          "code_discovery_result": dict | None,
          "replication_result": dict | None,
          "evaluation_result": dict | None,     # the latest review: of the closest baseline so far
          "reruns": list,                       # rerun requests Master Agent acted on, with their effect
          "rerun_end_reason": str,              # why Master Agent ended the rerun loop
          "experiment_results": list,           # every Experiment Agent analysis, in order
          "experiment_result": dict | None,     # the last one
          "blockers": [...],   # non-empty only when success is False
        }
    """
    from src.orchestration.handoff import request
    from src.orchestration.state_manager import update_state

    # This run's own banner ts is the floor check_agent_requests uses
    # -- see check_agent_requests' floor_ts docstring for why this matters
    # once the pipeline is run repeatedly for different topics.
    run_floor_ts = slack_client.post_message(message_formatter.format_run_started(topic))

    ctx = {"topic": topic, "target_terms": [], "structured": None, "papers": None, "validation_result": None,
           "parent_paper": None, "ranked_pool": [], "dataset_result": None, "cleaning_result": None,
           "code_discovery_result": None, "replication_result": None, "evaluation_result": None,
           "experiment_result": None}
    drive = _new_drive(floor_ts=run_floor_ts)
    _drive(ctx, [request("master_agent", "Intake Agent", {"topic": topic}, f"New project topic: {topic}")], drive)

    result = {key: ctx[key] for key in ("structured", "papers", "validation_result", "parent_paper",
                                        "dataset_result", "cleaning_result", "code_discovery_result",
                                        "replication_result", "evaluation_result")}
    result.update(success=False, phase=drive["phase"], reruns=drive["history"],
                  rerun_end_reason=drive["end_reason"], experiment_results=drive["experiment_results"],
                  experiment_result=drive["experiment_results"][-1] if drive["experiment_results"] else None,
                  blockers=drive["blockers"])
    if drive["blockers"]:
        return result

    update_state({
        "evaluation_decision": ctx["evaluation_result"].get("decision"),
        "reruns": drive["history"],
        "experiments": drive["experiment_results"],
        "phase": "EXPERIMENTATION",
    })
    result.update(success=True, phase="EXPERIMENTATION")

    _relay_and_apply(result, ctx["ranked_pool"], run_floor_ts)

    return result
