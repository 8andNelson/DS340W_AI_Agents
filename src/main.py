import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents import master_agent
from src.orchestration.state_manager import reset_state


BANNER = """
============================================================
  AI Research & Replication Multi-Agent System
============================================================"""


def _print_papers(papers: list) -> None:
    print(f"\n--- Candidate Papers ({len(papers)} found) ---")
    for i, p in enumerate(papers, 1):
        score = p.get("parent_paper_score", 0)
        peer = "peer-reviewed" if p.get("peer_reviewed") else "NOT peer-reviewed"
        print(f"\n  [{i}] {p['title']}")
        print(f"       {p['year']} | {p['venue'] or 'Unknown venue'} | {peer}")
        print(f"       Score: {score}/10 | Citations: {p.get('citation_count', 0)}")
        if p.get("dataset"):
            print(f"       Dataset: {p['dataset']}")
        if p.get("methodology"):
            print(f"       Method: {p['methodology']}")
        if p.get("paper_url"):
            print(f"       URL: {p['paper_url']}")


def _print_validation_summary(result: dict) -> None:
    verified = result["verified"]
    unverified = result["unverified"]
    rejected = result["rejected"]

    print(f"\n--- Validation Results ---")
    print(f"  VERIFIED:   {len(verified)}")
    print(f"  UNVERIFIED: {len(unverified)}")
    print(f"  REJECTED:   {len(rejected)}")

    if rejected:
        print("\n  Rejected papers:")
        for p in rejected:
            print(f"    - {p['title'][:65]} | {p.get('rejection_reason', '')}")

    if unverified:
        print("\n  Unverified papers (flagged for human review):")
        for p in unverified:
            print(f"    - {p['title'][:65]} | {p.get('validation_notes', '')}")


def _print_parent_paper_recommendation(rec: dict) -> None:
    print(f"\n--- Parent Paper Comparison ({rec['candidate_pool_size']} candidates from '{rec['candidate_pool_source']}') ---")
    if rec.get("degraded"):
        print(f"  WARNING: {rec['degraded_reason']}")

    for row in rec["comparison_table"]:
        print(f"\n  {row['title'][:70]}")
        print(f"    Year: {row['publication_year']} | Peer-reviewed: {row['peer_reviewed']} | Composite score: {row['composite_score']:.2f}")
        print(f"    Methodology clarity: {row['methodology_clarity']} | Implementation clarity: {row['implementation_clarity']}")
        print(f"    Dataset: {row['dataset_availability']} | Code available: {row['code_availability']}")
        print(f"    Reproducibility: {row['reproducibility_difficulty']} | Compute: {row['computing_resources_estimate']}")

    audit = rec["code_availability_audit"]
    print(f"\n  Code availability audit: {audit['count_with_code']} of {rec['candidate_pool_size']} candidates have code.")
    if audit["policy_violation"]:
        print(f"    NOTE: {audit['note']}")

    chosen = rec["recommended_parent_paper"]
    print(f"\n>>> RECOMMENDED PARENT PAPER: {chosen['title']}")
    print(f"    {chosen['year']} | {chosen['venue'] or 'Unknown venue'}")
    print(f"    URL: {chosen.get('paper_url', '')}")
    print(f"    Expected difficulty: {rec['expected_difficulty']}")
    if rec.get("code_availability_deciding_factor"):
        print("    NOTE: code availability was a deciding factor in this ranking.")
    print(f"\n  Justification:\n  {rec['justification']}")


def _print_dataset_summary(result: dict) -> None:
    print(f"\n--- Dataset Pool ({result['status']}) ---")
    target = result.get("target_entries")
    print(f"  Total entries: {result['total_entries']} / {target if target is not None else 'no target'} "
          f"(target met: {result['target_met']})")
    plan = result.get("data_plan") or {}
    if plan.get("reasoning"):
        print(f"  Data plan:     {plan['reasoning']}")
    print(f"  Datasets used: {len(result['selected_datasets'])} "
          f"(considered {result['datasets_considered']} candidate papers)")
    print(f"  Search stage:  {result['search_stage']}")

    for d in result["selected_datasets"]:
        label = d.get("display_name") or d.get("name", "")
        print(f"    - {label} ({d.get('source', '')}): {d.get('entry_count', '?')} entries -> {d.get('name', '')}")

    if result.get("warnings"):
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")


def _print_cleaning_summary(result: dict) -> None:
    print(f"\n--- Master Dataset ({result['status']}) ---")
    print(f"  Datasets merged: {result['datasets_merged']} / {result['datasets_attempted']}")
    print(f"  Rows / columns:  {result['rows_total']} / {result['columns_total']} "
          f"(target: {result.get('target_rows') or 'none'}, target met: {result.get('target_met')})")
    if result.get("master_csv_path"):
        print(f"  Master CSV:      {result['master_csv_path']}")

    split_rows = result.get("split_rows") or {}
    if result.get("train_csv_path"):
        print(f"  Train/Test/Val:  {split_rows.get('train', 0)}/{split_rows.get('test', 0)}/"
              f"{split_rows.get('validation', 0)} rows (seed={result.get('split_seed', '')})")
        print(f"                   {result['train_csv_path']}")
        print(f"                   {result.get('test_csv_path', '')}")
        print(f"                   {result.get('validation_csv_path', '')}")

    if result.get("conflicts"):
        print("  Conflicts:")
        for c in result["conflicts"]:
            print(f"    - {c.get('dataset', '')} ({c.get('link', '')}): {c.get('reason', '')}")


def _print_code_discovery_summary(result: dict) -> None:
    print(f"\n--- Code Discovery ({result['status']}) ---")
    print(f"  Source stage: {result['source_stage']}")
    print(f"  Candidates considered: {len(result['candidates_considered'])}")

    repo = result.get("repository")
    if repo:
        print(f"  Repository: {repo.get('repo_url', '')} ({repo.get('source', '')})")
        print(f"  License:    {repo.get('license', '') or 'unknown'}")
        print(f"  Stars:      {repo.get('stars', 'n/a')}")
        print(f"  Dependency files: {', '.join(repo.get('dependency_files', [])) or 'none found'}")
    else:
        print("  No verified implementation found -- Replication Agent will reconstruct from the paper.")

    if result.get("warnings"):
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")


def _print_replication_summary(result: dict) -> None:
    print(f"\n--- Baseline Replication ({result['status']}) ---")
    print(f"  Source: {result['source']}")
    if result.get("repo_url"):
        print(f"  Repository:  {result['repo_url']} (commit {result.get('commit_sha', '') or 'unknown'})")
    print(f"  Entry point: {result.get('entry_point', '')} (via {result.get('entry_point_source', '')})")
    print(f"  Exit code:   {result.get('exit_code')}  Duration: {result.get('duration_seconds')}s")
    print(f"  Replication dir: {result.get('replication_dir', '')}")

    if result.get("inferred_assumptions"):
        print("  Inferred assumptions (reconstructed code, not the authors'):")
        for a in result["inferred_assumptions"]:
            print(f"    - {a}")

    if result.get("warnings"):
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")


def _print_evaluation_summary(result: dict) -> None:
    print(f"\n--- Evaluation ({result['status']}) ---")
    for m in result.get("metrics", []):
        print(f"  - {m['metric']}: published={m['published_value']}, reproduced={m['reproduced_value']} "
              f"-> {m['classification']}" + (f" ({m['notes']})" if m.get("notes") else ""))
    if not result.get("metrics"):
        print("  No comparable metrics found.")
    if result.get("likely_causes"):
        print("  Likely causes:")
        for c in result["likely_causes"]:
            print(f"    - {c}")
    review = result.get("review") or {}
    if review.get("assessment"):
        print(f"  Analyst review: {review['assessment']}")
    for f in review.get("static_findings", []) + [f for f in review.get("findings", []) if f.get("verified")]:
        print(f"    - [{f['severity']}] {f['issue']}")
    decision = result.get("decision") or {}
    if decision:
        print(f"  Decision: {decision.get('action', '')}"
              + (f" -> {decision['target_agent']}" if decision.get("target_agent") else ""))


def _print_reruns(reruns: list, end_reason: str) -> None:
    print("\n--- Reruns (requested by agents, run by Master Agent) ---")
    for r in reruns:
        kept = "kept" if r.get("applied") else "not kept"
        print(f"  Rerun {r['rerun']}: {r['requested_by']} -> {r['target_agent']} ({kept}): {r.get('outcome', '')}")
    if not reruns:
        print("  None.")
    if end_reason:
        print(f"  Master Agent ended the loop: {end_reason}.")


def _print_experiment_summary(result: dict) -> None:
    gap = result.get("gap_points")
    print(f"\n--- Experiment Agent's last analysis ({result.get('status', '')}) ---")
    print(f"  Mean gap to the paper: {gap if gap is not None else 'not measurable'} percentage points")
    if result.get("analysis"):
        print(f"  Analysis: {result['analysis']}")
    decision = result.get("decision") or {}
    if decision:
        print(f"  Decision: {decision.get('action', '')}"
              + (f" -> {decision['target_agent']}" if decision.get("target_agent") else ""))
    for note in result.get("decision_notes") or []:
        print(f"    - {note}")


def main():
    print(BANNER)

    if "--reset" in sys.argv:
        reset_state()
        print("\nProject state reset.\n")
        sys.argv.remove("--reset")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        topic = " ".join(args)
    else:
        print()
        topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Error: No research topic provided.")
        sys.exit(1)

    print(f"\nTopic: {topic}")
    print("\n[Master Agent] Starting supervised pipeline "
          "(Intake -> Research -> Validation & Selection -> Data Agent -> Cleaning Agent -> "
          "Code Discovery Agent -> Replication Agent -> Evaluation Agent, then reruns requested by the "
          "Evaluation and Experiment Agents)...")

    result = master_agent.run_pipeline(topic)

    if not result["success"]:
        print(f"\nPipeline halted at phase {result['phase']}. Master Agent could not get a passing "
              f"result after {master_agent.MAX_ATTEMPTS} diagnosed attempts. See the corrections above, "
              f"logs/master_agent_log.json and logs/agent_reports/.")
        sys.exit(1)

    if result["structured"]:
        s = result["structured"]
        print(f"\n  Domain:   {s.get('domain', 'N/A')}")
        print(f"  ML Task:  {s.get('ml_task', 'N/A')}")
        print(f"  Keywords: {', '.join(s.get('keywords', []))}")

    if result["papers"]:
        _print_papers(result["papers"])

    validation_result = result["validation_result"]
    _print_validation_summary(validation_result)
    _print_parent_paper_recommendation(validation_result)

    if result["dataset_result"]:
        _print_dataset_summary(result["dataset_result"])

    if result["cleaning_result"]:
        _print_cleaning_summary(result["cleaning_result"])

    if result["code_discovery_result"]:
        _print_code_discovery_summary(result["code_discovery_result"])

    if result["replication_result"]:
        _print_replication_summary(result["replication_result"])

    if result["evaluation_result"]:
        _print_evaluation_summary(result["evaluation_result"])
        _print_reruns(result.get("reruns") or [], result.get("rerun_end_reason", ""))

    if result["experiment_result"]:
        _print_experiment_summary(result["experiment_result"])

    print(f"\n[State] Phase -> {result['phase']}")
    print("[State] parent_paper_approved -> True")
    print("[State] Saved to logs/project_state.json")
    if result.get("run_id"):
        print(f"[Run Store] Run {result['run_id']} recorded -- view it with: "
              f"python -m src.orchestration.run_store show {result['run_id']}")
    print("\nParent Paper approved, dataset built, master CSV cleaned, code discovery attempted, "
          "a baseline reproduction established and evaluated against the paper, and earlier agents "
          "rerun on the Evaluation and Experiment Agents' requests to bring the score closer to the "
          "paper. Next pipeline stage: Report Agent (Milestone 13).")


if __name__ == "__main__":
    main()
