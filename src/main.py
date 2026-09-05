import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents import intake_agent, research_agent
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

    print("\n[State] Phase -> LITERATURE_SEARCH")
    print("[State] Saved to logs/project_state.json")
    print("\nMilestone 2 complete. Ready for paper validation (Milestone 3).")


if __name__ == "__main__":
    main()
