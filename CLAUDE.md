# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a semester-long AI Agents project.

The goal is to build a multi-agent system that accepts a user-provided machine learning or data-related research topic and manages the full research workflow from academic-paper discovery through code replication, evaluation, reporting, and presentation generation.

The system should use multiple specialized AI agents coordinated by a central Master Agent.

Agents must communicate their progress through Slack so that the user can observe what each agent is doing, what decisions are being made, and when the Master Agent intervenes.

---

# Core Project Goal

The system must perform the following pipeline:

1. Accept a machine learning or data-related topic from the user.
2. Research relevant peer-reviewed academic literature.
3. Find at least 5 qualifying academic papers.
4. Select one qualifying paper as the Parent Paper.
5. Locate or reconstruct the implementation used by the Parent Paper.
6. Successfully execute and reproduce the Parent Paper's implementation.
7. Track development and changes using GitHub.
8. Extend or analyze the implementation later in the semester.
9. Generate a final research report.
10. Generate presentation slides.
11. Use specialized AI agents for each major stage.
12. Use a Master Agent to supervise and correct the specialized agents.
13. Post meaningful agent activity and Master Agent interventions to Slack.

---

# Non-Negotiable Academic Requirements

These requirements come from the course project and must be treated as hard constraints.

## Academic Paper Requirements

The Research Agent must locate a minimum of **5 academic papers** related to the user's topic.

All selected papers must:

* Be legitimate academic papers or journal/conference publications.
* Be peer reviewed.
* Be published between **2022 and the present year**.
* Be relevant to the user's machine learning or data-related topic.
* Come from credible academic sources.

Papers must NOT come from:

* Medium.com
* Social media
* Blog posts
* Kaggle articles/notebooks presented as academic literature
* Random websites
* Non-peer-reviewed opinion pieces

Kaggle, GitHub, Google, and other sources may be used later to locate implementation code, but they do not count as academic papers.

## Code Availability Restriction

Of the minimum 5 academic papers, only **one paper may be selected based on or showcase an available code implementation**.

Do not choose the literature set simply because every paper has convenient GitHub code.

The research process must prioritize academic relevance and quality.

---

# Parent Paper

Exactly one paper must ultimately be designated as the **Parent Paper**.

The Parent Paper becomes the foundation of the implementation and final research project.

The Parent Paper must:

* Be peer reviewed.
* Be published from 2022 through the present.
* Clearly describe the problem being studied.
* Clearly identify the dataset or data source.
* Describe its methodology.
* Explain the experimental procedure.
* Report results.
* Contain conclusions.
* Include useful visual material such as:

  * Graphs
  * Tables
  * Figures
  * Experimental results
  * Architecture diagrams
  * Flowcharts when applicable
* Provide enough implementation detail that the work can realistically be replicated.

When comparing possible Parent Papers, strongly prefer papers that clearly explain:

* Data collection
* Data preprocessing
* Features
* Model architecture
* Hyperparameters
* Training procedure
* Evaluation metrics
* Experimental setup
* Baselines
* Results

A paper with vague implementation details is a poor Parent Paper candidate even if its topic is interesting.

---

# Parent Paper Selection Process

The Research Agent must not immediately choose the first acceptable paper.

It should create a comparison of candidate Parent Papers.

At minimum, compare:

* Publication year
* Peer-reviewed status
* Research topic
* Dataset availability
* Dataset source
* Methodology clarity
* Implementation clarity
* Code availability
* Required computing resources
* Reproducibility difficulty
* Tables/graphs/figures available
* Evaluation metrics
* Expected difficulty of reproducing results

The Master Agent must review the recommendation before the Parent Paper is finalized.

---

# Source Verification

Never invent academic papers, citations, authors, DOIs, publication venues, datasets, repositories, results, or URLs.

Before treating a paper as valid, verify as much of the following as possible:

* Paper title
* Authors
* Publication year
* Journal or conference
* Peer-reviewed status
* DOI or official publication page
* Dataset
* Code repository if one exists

If something cannot be verified, clearly mark it as:

`UNVERIFIED`

Do not silently assume that a source is legitimate.

---

# Implementation / Coding Phase

The first implementation milestone is **replication**, not improvement.

The objective is to reproduce what the Parent Paper's authors did as closely as reasonably possible.

## Preferred Implementation Order

Attempt implementation in this order:

1. Official code repository supplied by the authors.
2. GitHub repository associated with the paper.
3. Public implementation referenced by the authors.
4. Kaggle implementation of the paper or method.
5. Other credible online implementation.
6. Reconstruct the implementation from the paper only if usable code cannot be located.

The project should avoid unnecessary rewriting during the initial replication phase.

The purpose of Phase 1 is to get the selected implementation running successfully.

---

# Baseline Replication Rule

Do NOT immediately improve, optimize, refactor, or redesign the implementation.

First establish a working **baseline reproduction**.

The baseline should answer:

* Can the code execute?
* Can the dataset be obtained?
* Can preprocessing run?
* Can the model train or execute?
* Can evaluation run?
* Can expected outputs be generated?
* Can graphs or tables from the paper be approximately reproduced?
* Are the reproduced results reasonably close to the published results?

Only after a working baseline has been established should experimental changes be introduced.

---

# Existing Code vs New Code

If implementation code is available online:

* Prefer using that code for the initial reproduction.
* Preserve the original behavior as much as possible.
* Document where the code came from.
* Preserve applicable licenses and attribution.
* Record the repository URL.
* Record the commit/version used if possible.

If no implementation can be found:

* Implementation may be reconstructed from the methodology described in the Parent Paper.
* Clearly document which parts are inferred.
* Do not claim reconstructed code is the authors' original code.

---

# Environment Reproducibility

The Coding Agent must document the environment required to run the project.

Where applicable, record:

* Python version
* Operating system assumptions
* Dependencies
* Package versions
* CUDA version
* PyTorch/TensorFlow version
* Hardware requirements
* Dataset paths
* Environment variables
* Random seeds

Use reproducible environment files when appropriate, such as:

* `requirements.txt`
* `environment.yml`
* `pyproject.toml`

Never commit secrets, credentials, tokens, or private keys.

---

# GitHub Requirements

The repository will be public.

All meaningful project development must be tracked through Git.

Important milestones should be committed separately.

Examples:

* Initial project structure
* Research-agent implementation
* Paper-selection logic
* Slack integration
* Parent Paper selected
* Dataset loader working
* Baseline implementation added
* Baseline successfully executed
* Evaluation implemented
* First reproducible result
* Experimental modification
* Final report generation
* Presentation generation

Commit messages should explain what changed.

Avoid meaningless messages such as:

`update`

Prefer messages such as:

`Add Parent Paper validation pipeline`

or:

`Reproduce baseline Random Forest experiment from Parent Paper`

---

# Baseline Preservation

Once the Parent Paper implementation is working, preserve the working baseline.

Future experimental changes should not destroy the known-working implementation.

Prefer one of the following strategies:

* Git branches
* Tagged commits
* Separate experiment configuration files
* Separate experiment directories

A baseline result should always remain reproducible.

---

# Proposed Multi-Agent Architecture

The system should use specialized agents rather than one AI attempting to perform every task.

The current recommended architecture is:

## 1. Master Agent

The Master Agent is the orchestrator.

Responsibilities:

* Receive the user's project topic.
* Break the request into tasks.
* Delegate tasks to specialized agents.
* Track agent progress.
* Validate agent outputs.
* Reject incorrect work.
* Ask agents to retry when necessary.
* Resolve disagreements between agents.
* Ensure academic requirements are followed.
* Ensure the Parent Paper satisfies all constraints.
* Prevent hallucinated sources.
* Prevent implementation work from starting before paper validation.
* Track overall project state.
* Determine when the workflow may advance to the next phase.

The Master Agent should not blindly trust another agent.

The Master Agent should review important outputs before approving them.

---

# 2. User Intake Agent

Responsibilities:

* Receive the user's machine learning or data-related topic.
* Clarify the research question when necessary.
* Extract important keywords.
* Identify likely ML/data-science subfields.
* Generate academic search terms.
* Pass a structured research request to the Research Agent.

Example input:

`I want to investigate machine learning for detecting credit card fraud.`

Possible structured output:

* Domain: financial fraud detection
* ML task: classification / anomaly detection
* Keywords:

  * credit card fraud detection
  * machine learning fraud detection
  * imbalanced classification
  * anomaly detection
  * transaction classification

---

# 3. Research Agent

Responsibilities:

* Search academic literature.
* Find at least 5 qualifying papers.
* Keep publication dates within 2022-present.
* Check academic credibility.
* Collect paper metadata.
* Collect links/DOIs.
* Identify datasets.
* Identify methodologies.
* Identify results.
* Identify implementation information.
* Identify code availability.
* Produce candidate Parent Papers.

The Research Agent must produce structured results rather than only prose.

Suggested paper record:

```python
{
    "title": "",
    "authors": [],
    "year": 0,
    "venue": "",
    "peer_reviewed": False,
    "doi": "",
    "paper_url": "",
    "dataset": "",
    "dataset_source": "",
    "methodology": "",
    "results_summary": "",
    "has_tables": False,
    "has_figures": False,
    "has_graphs": False,
    "has_code": False,
    "code_url": "",
    "parent_paper_score": 0,
    "notes": ""
}
```

---

# 4. Paper Validation Agent

The Validation Agent independently checks the Research Agent's findings.

Responsibilities:

* Verify publication dates.
* Verify peer-reviewed status.
* Confirm that papers are real.
* Confirm academic venue.
* Confirm the source is not a prohibited source.
* Verify DOI/publication links where possible.
* Confirm the Parent Paper contains required sections.
* Confirm dataset information exists.
* Check reproducibility potential.
* Flag questionable or unverifiable claims.

The Validation Agent should operate independently from the Research Agent whenever possible.

---

# 5. Parent Paper Selection Agent

Responsibilities:

* Rank qualifying papers.
* Compare reproducibility.
* Compare implementation clarity.
* Compare dataset accessibility.
* Compare compute requirements.
* Compare methodology clarity.
* Determine likely difficulty.
* Recommend the strongest Parent Paper.

It must explain *why* the recommended Parent Paper is better than alternatives.

The Master Agent makes the final approval.

---

# 6. Code Discovery Agent

Once a Parent Paper is approved, search for implementation resources.

Responsibilities:

* Locate the authors' official repository.
* Search GitHub.
* Search paper supplemental materials.
* Search academic repositories.
* Search Kaggle if appropriate.
* Search Google or other code indexes.
* Determine whether implementations actually correspond to the Parent Paper.
* Identify licensing information.
* Identify setup instructions.
* Identify required datasets.
* Identify dependencies.

Do not assume a repository implements the paper simply because it has a similar name.

---

# 7. Replication / Coding Agent

Responsibilities:

* Set up the Python environment.
* Download or prepare the dataset.
* Install dependencies.
* Run the original implementation.
* Debug environment problems.
* Record commands used.
* Record changes required to make the implementation run.
* Reproduce experiments.
* Generate output artifacts.
* Compare reproduced results with the Parent Paper.

During baseline replication, minimize modifications to the research implementation.

Any necessary modification should be documented.

---

# 8. Evaluation Agent

Responsibilities:

* Compare reproduced results to published results.
* Compare metrics.
* Compare tables.
* Compare figures when applicable.
* Detect major discrepancies.
* Determine possible causes for differences.

Possible causes may include:

* Dataset version differences
* Random seeds
* Hardware
* Library versions
* Missing preprocessing details
* Hyperparameter differences
* Undocumented author implementation details

The Evaluation Agent must distinguish between:

* Exact reproduction
* Approximate reproduction
* Failed reproduction

---

# 9. Experiment Agent

This agent should become active only after baseline reproduction succeeds.

Responsibilities:

* Propose meaningful changes or experiments.
* Change one major variable at a time when possible.
* Maintain experiment configurations.
* Preserve baseline results.
* Evaluate improvements or regressions.
* Record hypotheses before running experiments.
* Track experiment results.

Example experimental changes:

* Alternative model
* Hyperparameter changes
* Different preprocessing
* Different feature selection
* Alternative loss function
* Alternative sampling technique
* Additional evaluation metric
* Different dataset split

The experiment stage should produce original analysis rather than simply rerunning the Parent Paper.

---

# 10. Report Agent

After implementation and experimentation are complete, the Report Agent generates the final research paper/report.

The report should be grounded in actual project artifacts and results.

Do not fabricate results.

Possible report sections:

1. Abstract
2. Introduction
3. Research Question
4. Literature Review
5. Parent Paper
6. Dataset
7. Methodology
8. Baseline Replication
9. Proposed Changes / Experiments
10. Experimental Setup
11. Results
12. Discussion
13. Limitations
14. Conclusion
15. Future Work
16. References

Graphs, tables, and metrics should come from actual experiments whenever possible.

---

# 11. Presentation Agent

The Presentation Agent produces slides summarizing the completed project.

Possible presentation structure:

1. Title
2. Research Problem
3. Motivation
4. Literature Review
5. Parent Paper
6. Dataset
7. Methodology
8. Original Architecture / Workflow
9. Baseline Reproduction
10. Proposed Changes
11. Experimental Results
12. Comparison
13. Conclusions
14. Limitations
15. Future Work
16. References

Slides should favor visual communication over dense paragraphs.

---

# Slack Integration

Slack will act as the human-observable communication layer for the AI agents.

Agents should post meaningful status updates so that the user can follow the project.

Do not post every internal operation.

Post updates when meaningful events occur.

Examples:

* Agent begins a major task.
* Research Agent finds candidate papers.
* Validation Agent rejects a paper.
* Parent Paper is recommended.
* Master Agent approves or rejects the recommendation.
* Code repository is found.
* Dataset download starts/completes.
* Baseline execution succeeds.
* Baseline execution fails.
* Agent requests assistance.
* Master Agent corrects another agent.
* New experiment starts.
* Experiment completes.
* Final report is generated.

---

# Master Agent Slack Requirement

Whenever the Master Agent corrects, overrides, redirects, or provides significant guidance to another agent, it **must create a Slack post**.

Example:

`MASTER -> RESEARCH AGENT: Paper #3 cannot be used because it was published in 2021. Replace it with a peer-reviewed paper published between 2022 and the present.`

Another example:

`MASTER -> CODING AGENT: Do not modify the model architecture yet. The project is still in baseline replication. Restore the Parent Paper configuration and establish a reproducible baseline first.`

These messages should create an observable audit trail of agent supervision.

---

# Slack Message Format

Prefer structured Slack messages.

Example:

```text
[Research Agent]
STATUS: Working
TASK: Academic paper search
TOPIC: Credit card fraud detection
PROGRESS: 4/5 qualifying papers verified
NEXT: Validate peer-reviewed status for remaining candidates
```

Master Agent correction:

```text
[Master Agent]
ACTION: Correction
TARGET: Research Agent
ISSUE: Candidate paper published before 2022
DECISION: Rejected
NEXT: Find replacement published between 2022-present
```

Coding update:

```text
[Replication Agent]
STATUS: Baseline execution failed
CAUSE: Repository requires Python 3.10 but environment uses Python 3.13
ACTION: Creating Python 3.10 virtual environment
```

---

# Agent Communication

Agents should communicate through structured messages.

Avoid passing large unstructured text between agents when a structured representation would work.

Suggested message model:

```python
{
    "sender": "research_agent",
    "recipient": "master_agent",
    "task_id": "",
    "status": "in_progress",
    "message_type": "result",
    "summary": "",
    "artifacts": [],
    "warnings": [],
    "requires_review": True
}
```

Possible statuses:

* `queued`
* `in_progress`
* `blocked`
* `needs_review`
* `approved`
* `rejected`
* `completed`
* `failed`

---

# Human Oversight

The user remains the final human authority.

The system should pause for human approval at major irreversible or academically important decisions when appropriate.

Important approval checkpoints include:

* Final Parent Paper selection
* Major change in project research question
* Abandoning the selected Parent Paper
* Major experimental direction changes
* Final report submission version

The system may automate routine steps without requiring approval for every action.

---

# Project State

Maintain persistent project state so agents know what has already happened.

Suggested state structure:

```python
{
    "project_topic": "",
    "research_question": "",
    "phase": "",
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
```

Do not rely solely on conversational memory for important project state.

Persist important state to files or a database.

---

# Suggested Workflow States

Use an explicit workflow instead of allowing agents to operate randomly.

```text
USER_INPUT
    ↓
TOPIC_ANALYSIS
    ↓
LITERATURE_SEARCH
    ↓
PAPER_VALIDATION
    ↓
PARENT_PAPER_SELECTION
    ↓
HUMAN / MASTER APPROVAL
    ↓
CODE_DISCOVERY
    ↓
ENVIRONMENT_SETUP
    ↓
BASELINE_REPLICATION
    ↓
BASELINE_EVALUATION
    ↓
EXPERIMENT_DESIGN
    ↓
EXPERIMENTATION
    ↓
RESULT_ANALYSIS
    ↓
REPORT_GENERATION
    ↓
PRESENTATION_GENERATION
    ↓
COMPLETE
```

Agents should not skip workflow states unless the Master Agent explicitly approves the change.

---

# Error Handling

Agents must report failures instead of hiding them.

When something fails, record:

* What was attempted
* What failed
* Error message
* Likely cause
* What was tried
* Recommended next action

Example:

```text
STATUS: BLOCKED

Task:
Run Parent Paper training script.

Failure:
CUDA out-of-memory error.

Environment:
NVIDIA GPU with 8 GB VRAM.

Paper configuration:
Batch size 64.

Recommended next step:
Determine whether reducing batch size would invalidate reproduction requirements before modifying configuration.
```

---

# No Hallucination Policy

This project deals with academic research.

Hallucinated information can invalidate the entire project.

Never fabricate:

* Papers
* Authors
* Journals
* Conferences
* Publication dates
* DOI numbers
* Datasets
* Code repositories
* Experimental results
* Metrics
* Quotes
* Citations

When uncertain, state uncertainty.

Prefer:

`I could not verify this information.`

over guessing.

---

# Results Integrity

Never invent experimental results to make replication appear successful.

If the reproduced result differs from the Parent Paper, report the actual result.

For example:

```text
Published accuracy: 94.2%
Reproduced accuracy: 91.8%
Difference: -2.4 percentage points
```

Then investigate possible explanations.

A failed or imperfect reproduction is still useful research information.

---

# Repository Organization

```text
project-root/
│
├── CLAUDE.md
├── README.md
├── .gitignore
├── requirements.txt
├── run.py                  ← entry point: python run.py "topic"
│
├── src/
│   ├── config.py           ← env vars, model constants
│   ├── models.py           ← model registry (all OpenRouter model IDs)
│   ├── agents/
│   │   ├── master_agent.py
│   │   ├── intake_agent.py
│   │   ├── research_agent.py
│   │   ├── validation_agent.py
│   │   ├── paper_selection_agent.py
│   │   ├── code_discovery_agent.py
│   │   ├── replication_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── experiment_agent.py
│   │   ├── report_agent.py
│   │   └── presentation_agent.py
│   │
│   ├── slack/
│   │   ├── slack_client.py
│   │   └── message_formatter.py
│   │
│   ├── orchestration/
│   │   ├── workflow.py
│   │   └── state_manager.py
│   │
│   └── main.py
│
├── data/raw, data/processed    ← gitignored
├── papers/candidates, papers/parent  ← gitignored
├── replication/                ← gitignored
├── experiments/                ← gitignored
├── reports/                    ← gitignored
├── presentations/              ← gitignored
├── logs/                       ← gitignored (includes project_state.json)
└── tests/
```

---

# Run Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Run the system interactively
python run.py

# Run with topic as argument
python run.py "machine learning for detecting credit card fraud"

# Reset project state and start fresh
python run.py --reset "new topic"

# Run tests
pytest

# Run a single test
pytest tests/test_file.py::test_function_name
```

---

# Model Usage

Models are defined in `src/models.py` and configured in `src/config.py`.

```python
DEFAULT_MODEL  # Free model — use during development (no credits required)
FAST_MODEL     # anthropic/claude-haiku-4.5 — lightweight tasks
CAPABLE_MODEL  # anthropic/claude-sonnet-4.5 — research, validation, reports
```

Agents import `DEFAULT_MODEL` from `src/config`. Switch `DEFAULT_MODEL` in `config.py` once OpenRouter credits are loaded.

## Web Search Providers

Agents that search the web use one of two providers:

1. **SerpentAPI** (`SERPENT_API_KEY`) — primary. Use unless rate-limited.
2. **BraveSearchAPI** (`BRAVE_API_KEY`) — fallback. Limited to **1,000 requests/month**, resets on the **1st of each month**.

Always attempt SerpentAPI first. On rate-limit error, fall back to Brave and log the switch.

---

# Secrets

All credentials are stored in `.env` (gitignored). See `.env.example` for required variable names:

```text
OPENROUTER_API_KEY=
SERPENT_API_KEY=
BRAVE_API_KEY=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
```

---

# Development Philosophy

Build incrementally. Do not attempt to build the entire multi-agent platform at once.

## Milestones

1. CLI entry point that accepts a research topic ✅
2. Research Agent — returns structured paper candidates
3. Paper validation
4. Parent Paper selection
5. Master Agent
6. Slack status reporting
7. Code discovery
8. Replication workflow
9. Experiment management
10. Report generation
11. Presentation generation

---

# Instructions to Claude Code

When working in this repository:

1. Read this entire `CLAUDE.md` before making architectural decisions.
2. Inspect the existing repository before creating new files.
3. Do not rewrite working systems unnecessarily.
4. Keep modules focused.
5. Prefer readable Python over clever abstractions.
6. Add type hints when useful.
7. Preserve baseline research implementations.
8. Never fabricate research information.
9. Do not silently alter experimental results.
10. Maintain reproducibility.
11. Run relevant tests before declaring work complete.
12. Do not claim something works unless it has actually been tested.
13. When a command fails, inspect the actual error before modifying unrelated code.
14. Avoid large unrelated refactors while implementing a specific feature.
15. Keep Git commits logically scoped.
16. Protect the academic integrity of the project above convenience.

---

# Current Priority

Unless another task is explicitly provided, focus on the next incomplete milestone.

Milestone 1 (CLI + Intake Agent) is complete.

Next: **Milestone 2** — Research Agent that searches for and returns structured academic paper candidates.

---

# Definition of Success

The completed project should allow a user to provide a machine learning or data-related research topic and have a coordinated AI-agent system:

* Research current peer-reviewed literature.
* Find at least 5 qualifying papers.
* Select a strong Parent Paper.
* Verify academic requirements.
* Locate the Parent Paper's implementation.
* Execute and reproduce the implementation.
* Preserve the baseline.
* Conduct later experiments.
* Track progress in GitHub.
* Communicate agent activity through Slack.
* Allow a Master Agent to supervise other agents.
* Preserve an observable record of Master Agent corrections.
* Analyze real experimental results.
* Produce a final academic report.
* Produce presentation slides.

The system should prioritize **research integrity, reproducibility, transparency, and observable agent collaboration** over autonomous behavior for its own sake.
