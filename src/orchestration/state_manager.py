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
    "code_repository": "",
    "dataset": "",
    "baseline_status": "",
    "baseline_results": {},
    "experiments": [],
    "report_status": "",
    "presentation_status": ""
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
