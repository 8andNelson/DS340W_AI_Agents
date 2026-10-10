import json
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent.parent / "logs" / "project_state.json"

DEFAULT_STATE = {
    "project_topic": "",
    "research_question": "",
    "phase": "USER_INPUT",
    "papers": [],
    "parent_paper": None,
    "parent_paper_approved": False,
    "code_repository": None,  # {"status", "repository", "source_stage"} once Code Discovery Agent runs
    "dataset": None,
    "cleaning": None,
    "baseline_status": "",  # "OK" | "BLOCKED" | "TIMEOUT" once Replication Agent runs
    "baseline_results": {},  # {"source", "repo_url", "commit_sha", "entry_point", "exit_code",
                              #  "duration_seconds", "inferred_assumptions"} once Replication Agent runs
    "experiments": [],
    "report_status": "",
    "presentation_status": "",
    "slack_last_checked_ts": None,  # cursor for master_agent.check_agent_requests
}


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def update_state(updates: dict) -> dict:
    state = load_state()
    state.update(updates)
    save_state(state)
    return state


def reset_state() -> dict:
    state = DEFAULT_STATE.copy()
    save_state(state)
    return state
