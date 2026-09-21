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
    print("\n[Master Agent] Starting supervised pipeline (Intake -> Research -> Validation & Selection)...")

    result = master_agent.run_pipeline(topic)

    if not result["success"]:
        print(f"\nPipeline halted at phase {result['phase']}. Master Agent could not get "
              f"a passing result after retrying. See corrections above and logs/master_agent_log.json.")
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

    print("\n[State] Phase -> MASTER_APPROVAL")
    print("[State] parent_paper_approved -> True")
    print("[State] Saved to logs/project_state.json")
    print("\nParent Paper approved by Master Agent. Ready for Code Discovery (Milestone 7).")


if __name__ == "__main__":
    main()
