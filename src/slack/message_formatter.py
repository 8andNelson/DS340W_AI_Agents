"""
Builds the structured Slack message text blocks used across the pipeline
(the `[Agent Name]\\nSTATUS: ...` format from CLAUDE.md's "Slack Message
Format" section), and parses the one machine-readable shape -- REQUEST --
back out of raw message text.

Keeping format and parse in the same module keeps them from drifting apart:
whatever format_request emits is exactly what parse_request must be able to
read back, since a REQUEST message is how one agent (today, only Master
Agent acting on its own retries; in the future, any agent) asks Master
Agent to take an action on another agent's behalf -- e.g. a future Modeling
Agent asking for Data Agent to be rerun with a larger target.
"""
import re

REQUEST_TARGET_RE = re.compile(r"target\s*=\s*(\d+)", re.IGNORECASE)


def format_started(agent_name: str, task: str) -> str:
    """Every agent's own 'I am running' message. Always begins with the
    agent's name, per CLAUDE.md's Slack Message Format."""
    return f"[{agent_name}]\nSTATUS: Working\nTASK: {task}"


def format_finished(agent_name: str, status: str, summary: str, next_step: str = "") -> str:
    """Every agent's own 'I am done' message. `status` is a short label
    (e.g. 'Completed', 'Failed') -- not the Master Agent's judgment of the
    output, which gets its own, separate [Master Agent] Approval/Correction
    post via format_master_entry."""
    lines = [f"[{agent_name}]", f"STATUS: {status}", f"SUMMARY: {summary}"]
    if next_step:
        lines.append(f"NEXT: {next_step}")
    return "\n".join(lines)


def format_run_started(topic: str) -> str:
    """Posted once at the very start of each master_agent.run_pipeline()
    call. Two purposes: gives a human scrolling the shared channel a clear
    boundary between separate topic runs, and its Slack ts is used as a
    floor so a run never acts on a request left over from a previous,
    different topic's run (see master_agent.check_agent_requests)."""
    return f"[Master Agent]\nSTATUS: New pipeline run started\nTOPIC: {topic}"


def format_master_entry(entry: dict) -> str:
    """Renders one of master_agent._log_entry's dicts (action/target/issue/
    decision/next) in the exact ACTION/TARGET/ISSUE/DECISION/NEXT shape
    CLAUDE.md's Master Agent correction examples use."""
    return (
        "[Master Agent]\n"
        f"ACTION: {entry.get('action', '')}\n"
        f"TARGET: {entry.get('target', '')}\n"
        f"ISSUE: {entry.get('issue', '')}\n"
        f"DECISION: {entry.get('decision', '')}\n"
        f"NEXT: {entry.get('next', '')}"
    )


def format_request(sender_agent: str, target_agent: str, reason: str, detail: str = "") -> str:
    """The one structured shape an agent uses to ask Master Agent to act on
    another agent's behalf. `detail` may include `target=<int>` to state a
    specific numeric ask (e.g. a dataset entry-count target) -- see
    parse_request / REQUEST_TARGET_RE."""
    lines = [f"[{sender_agent}]", f"REQUEST: {target_agent}", f"REASON: {reason}"]
    if detail:
        lines.append(f"DETAIL: {detail}")
    return "\n".join(lines)


def parse_request(text: str) -> dict | None:
    """
    Inverse of format_request. Returns None if text doesn't match the
    REQUEST shape -- an ordinary status update (STATUS:/TASK:/SUMMARY:), a
    human's plain-text message, or anything else posted in the channel
    must never be misread as a request. Deterministic line-based parsing
    (no LLM judgment call), per this project's preference for a hard,
    reproducible gate wherever one is possible.
    """
    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not (lines[0].startswith("[") and lines[0].endswith("]")):
        return None
    sender = lines[0][1:-1].strip()

    fields: dict[str, str] = {}
    current_key = None
    for line in lines[1:]:
        match = re.match(r"^([A-Z]+):\s*(.*)$", line)
        if match:
            current_key = match.group(1)
            fields[current_key] = match.group(2)
        elif current_key:
            fields[current_key] += " " + line

    if "REQUEST" not in fields or "REASON" not in fields:
        return None

    return {
        "sender": sender,
        "target": fields["REQUEST"].strip(),
        "reason": fields["REASON"].strip(),
        "detail": fields.get("DETAIL", "").strip(),
    }


def parse_target_from_detail(detail: str) -> int | None:
    """Best-effort extraction of an explicit `target=<int>` from a
    REQUEST's DETAIL line. Returns None (never guesses) if no such value
    is stated, so the caller can fall back to its own default increment."""
    if not detail:
        return None
    match = REQUEST_TARGET_RE.search(detail)
    return int(match.group(1)) if match else None
