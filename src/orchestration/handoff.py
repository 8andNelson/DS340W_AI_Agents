"""
Agent-to-agent handoffs, through Master Agent.

No agent runs another agent. When an agent finishes, its module's
handoff(result, context) returns a list of requests to Master Agent: which
agents to run next and the information to pass on. Master Agent checks
each request and runs independent first-pass stages concurrently, up to
its worker limit. Requests wait for approved prerequisites; identical
requests joining two branches execute once, while conflicting inputs are
rejected. An unresolved dependency halts rather than skipping a stage.
Reruns remain serial to preserve baseline snapshot/restore semantics.

A person may turn off the optional agents for a run (OPTIONAL_AGENTS);
Master Agent then skips their requests and logs the skip.

This module holds the shared parts: the request builder, the workflow
order each request is checked against, and the phase each agent's stage
belongs to.
"""
import re

# Which agents must have produced an approved result before an agent may
# run ("Agents should not skip workflow states", CLAUDE.md).
PREREQUISITES = {
    "Intake Agent": (),
    "Research Agent": ("Intake Agent",),
    "Validation Agent": ("Research Agent",),
    "Data Agent": ("Validation Agent",),
    "Cleaning Agent": ("Data Agent",),
    "Code Discovery Agent": ("Validation Agent",),
    "Replication Agent": ("Cleaning Agent", "Code Discovery Agent"),
    "Evaluation Agent": ("Replication Agent",),
    "Experiment Agent": ("Evaluation Agent",),
    "Report Agent": ("Evaluation Agent",),
    "Presentation Agent": ("Report Agent",),
}

# The workflow state an agent's stage belongs to (project state's "phase").
AGENT_PHASE = {
    "Intake Agent": "TOPIC_ANALYSIS",
    "Research Agent": "LITERATURE_SEARCH",
    "Validation Agent": "PARENT_PAPER_SELECTION",
    "Data Agent": "DATA_DISCOVERY",
    "Cleaning Agent": "DATA_CLEANING",
    "Code Discovery Agent": "CODE_DISCOVERY",
    "Replication Agent": "BASELINE_REPLICATION",
    "Evaluation Agent": "BASELINE_EVALUATION",
    "Experiment Agent": "EXPERIMENTATION",
    "Report Agent": "REPORT_GENERATION",
    "Presentation Agent": "PRESENTATION_GENERATION",
}

# The result key each agent's output is kept under in Master Agent's context.
RESULT_KEY = {
    "Intake Agent": "structured",
    "Research Agent": "papers",
    "Validation Agent": "validation_result",
    "Data Agent": "dataset_result",
    "Cleaning Agent": "cleaning_result",
    "Code Discovery Agent": "code_discovery_result",
    "Replication Agent": "replication_result",
    "Evaluation Agent": "evaluation_result",
    "Experiment Agent": "experiment_result",
    "Report Agent": "report_result",
    "Presentation Agent": "presentation_result",
}


# Agents a person may turn off for a run (run.py --agents-off, the
# dashboard's switches). No other agent depends on their results, so the
# pipeline still reaches a valid, evaluated baseline without them. The
# Presentation Agent needs the Report Agent, so turning the report off turns
# the slides off too.
OPTIONAL_AGENTS = ("Experiment Agent", "Report Agent", "Presentation Agent")


def _agent_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name or "").lower()).removesuffix("agent")


def parse_agents_off(names) -> set:
    """'report, presentation' or ['Report Agent'] -> {'Report Agent', ...}.
    Raises ValueError for a name that is not an optional agent: turning off
    a required agent would stop the run from reaching an evaluated baseline."""
    if isinstance(names, str):
        names = names.split(",")
    chosen, refused = set(), []
    for name in names or []:
        if not str(name).strip():
            continue
        match = next((a for a in OPTIONAL_AGENTS if _agent_key(a) == _agent_key(name)), None)
        if match:
            chosen.add(match)
        else:
            refused.append(str(name).strip())
    if refused:
        raise ValueError(f"These agents cannot be turned off: {', '.join(refused)}. Only "
                         f"{', '.join(OPTIONAL_AGENTS)} can; every other agent is needed for the baseline.")
    if "Report Agent" in chosen:
        chosen.add("Presentation Agent")  # the slides are built from the report
    return chosen


def request(sender: str, target_agent: str, inputs: dict = None, summary: str = "",
            action: str = "run_agent") -> dict:
    """One handoff request to Master Agent (CLAUDE.md's message model)."""
    return {
        "sender": sender,
        "recipient": "master_agent",
        "message_type": "request",
        "status": "needs_review",
        "requires_review": True,
        "action": action,
        "target_agent": target_agent,
        "inputs": inputs or {},
        "summary": summary,
    }
