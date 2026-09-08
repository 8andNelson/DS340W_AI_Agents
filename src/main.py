import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents import intake_agent, research_agent, validation_agent, paper_selection_agent
from src.orchestration.state_manager import update_state, reset_state


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

    # --- Milestone 1: Intake Agent ---
    print(f"\nTopic: {topic}")
    print("\n[Intake Agent] Analyzing topic...")

    try:
        structured = intake_agent.run(topic)
    except Exception as e:
        print(f"\n[Intake Agent] ERROR: {e}")
        sys.exit(1)

    structured["project_topic"] = topic

    update_state({
        "project_topic": topic,
        "research_question": structured.get("domain", ""),
        "phase": "TOPIC_ANALYSIS",
    })

    print(f"  Domain:   {structured.get('domain', 'N/A')}")
    print(f"  ML Task:  {structured.get('ml_task', 'N/A')}")
    print(f"  Keywords: {', '.join(structured.get('keywords', []))}")

    # --- Milestone 2: Research Agent ---
    try:
        papers = research_agent.run(structured)
    except Exception as e:
        print(f"\n[Research Agent] ERROR: {e}")
        sys.exit(1)

    if not papers:
        print("\n[Research Agent] No qualifying papers found. Exiting.")
        sys.exit(1)

    update_state({
        "papers": papers,
        "phase": "LITERATURE_SEARCH",
    })

    _print_papers(papers)

    qualifying = [p for p in papers if p.get("peer_reviewed") and p.get("year", 0) >= 2022]
    print(f"\n[Research Agent] {len(qualifying)} papers meet peer-review + date requirements.")
    if len(qualifying) < 5:
        print(f"  WARNING: Need at least 5 qualifying papers. Found {len(qualifying)}.")
    else:
        print(f"  [OK] Minimum 5-paper requirement met.")

    # --- Milestone 3: Validation Agent ---
    try:
        validation_results = validation_agent.run(papers)
    except Exception as e:
        print(f"\n[Validation Agent] ERROR: {e}")
        sys.exit(1)

    verified = validation_results["verified"]
    unverified = validation_results["unverified"]
    rejected = validation_results["rejected"]

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

    print(f"\n  Top verified candidates:")
    for i, p in enumerate(verified[:5], 1):
        repro = p.get("validated_reproducibility", 0)
        score = p.get("parent_paper_score", 0)
        dataset = p.get("dataset") or p.get("validated_has_dataset") and "mentioned" or "unknown"
        print(f"\n  [{i}] {p['title']}")
        print(f"       {p['year']} | {p['venue'] or 'Unknown venue'}")
        print(f"       Reproducibility: {repro}/5 | Score: {score}/10")
        print(f"       Dataset: {dataset}")
        print(f"       {p.get('validation_notes', '')}")
        if p.get("paper_url"):
            print(f"       URL: {p['paper_url']}")

    update_state({
        "papers": validation_results["all"],
        "phase": "PAPER_VALIDATION",
    })

    if len(verified) < 5:
        print(f"\n  WARNING: Only {len(verified)} verified papers. Need at least 5.")
        print("  Consider running with a different topic or broader search terms.")
    else:
        print(f"\n  [OK] {len(verified)} verified papers ready for Parent Paper selection.")

    print("\n[State] Phase -> PAPER_VALIDATION")
    print("[State] Saved to logs/project_state.json")

    # --- Milestone 4: Parent Paper Selection Agent ---
    try:
        recommendation = paper_selection_agent.run(validation_results)
    except Exception as e:
        print(f"\n[Paper Selection Agent] ERROR: {e}")
        sys.exit(1)

    if recommendation["status"] == "NO_CANDIDATES":
        print("\n[Paper Selection Agent] No non-rejected papers available to select a Parent Paper from.")
        print("  Consider re-running Research/Validation with broader search terms.")
        update_state({"phase": "PARENT_PAPER_SELECTION", "parent_paper": None})
        sys.exit(1)

    _print_parent_paper_recommendation(recommendation)

    update_state({
        "papers": validation_results["all"],
        "parent_paper": recommendation["recommended_parent_paper"],
        "parent_paper_approved": False,
        "phase": "PARENT_PAPER_SELECTION",
    })

    print("\n[State] Phase -> PARENT_PAPER_SELECTION")
    print("[State] Saved to logs/project_state.json")
    print("\nMilestone 4 complete. Parent Paper recommended — awaiting Master Agent / human approval (Milestone 5).")


if __name__ == "__main__":
    main()
