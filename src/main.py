import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents import intake_agent
from src.orchestration.state_manager import update_state, reset_state


BANNER = """
============================================================
  AI Research & Replication Multi-Agent System
============================================================"""


def main():
    print(BANNER)

    # Allow --reset flag to clear previous session state
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
    print("\n[Intake Agent] Analyzing topic...")

    try:
        structured = intake_agent.run(topic)
    except Exception as e:
        print(f"\n[Intake Agent] ERROR: {e}")
        sys.exit(1)

    update_state({
        "project_topic": topic,
        "research_question": structured.get("domain", ""),
        "phase": "TOPIC_ANALYSIS",
    })

    print("\n--- Structured Research Request ---")
    print(f"  Domain:    {structured.get('domain', 'N/A')}")
    print(f"  ML Task:   {structured.get('ml_task', 'N/A')}")
    print(f"  Keywords:  {', '.join(structured.get('keywords', []))}")
    print("\n  Search Queries:")
    for q in structured.get("search_queries", []):
        print(f"    - {q}")

    print("\n[State] Phase -> TOPIC_ANALYSIS")
    print("[State] Saved to logs/project_state.json")
    print("\nMilestone 1 complete. Ready for literature search (Milestone 2).")


if __name__ == "__main__":
    main()
