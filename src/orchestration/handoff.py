"""
Agent-to-agent handoffs, through Master Agent.

No agent runs another agent. When an agent finishes, its module's
handoff(result, context) returns its request to Master Agent: which agent
to run next and the information to pass on (CLAUDE.md's agent message
model). Master Agent checks the request (master_agent._review_handoff),
runs the requested agent, and asks that agent for its own handoff -- until
an agent requests nothing more.

This module holds the shared parts: the request builder, the workflow
order each request is checked against, and the phase each agent's stage
belongs to.
"""

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
}


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
