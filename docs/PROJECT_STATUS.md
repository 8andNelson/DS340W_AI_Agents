# Project Status and Next Steps

Date: 2026-11-09

This document records the state of the project, the work that is started
but not finished, and the plan for the next phase. The next phase moves the
system from a fixed, linear pipeline to an autonomous agent swarm that
learns from its past runs.

---

## 1. Summary

* The pipeline runs end to end, from the user's topic to an evaluated
  baseline and an Experiment Agent analysis. Milestones 1-12 are complete.
* The Master Agent no longer uses a fixed list of stages. Each agent now
  requests the next agent through `handoff()`, and Master Agent checks and
  runs each request. This is the first step to the swarm model. It is
  tested with mocked agents but has **not been run live**.
* Two agents are not built yet: the **Report Agent** and the
  **Presentation Agent**. Their files exist but are empty.
* The system does not learn from earlier runs yet. Every run starts with
  no memory of the others. The plan is to add **Case-Based Reasoning (CBR)**
  from the DS-Agent paper (Guo et al., ICML 2024), together with a **ranking
  of agent outputs** and a **backend** that stores old runs as cases.
* A lot of work is not committed (see section 4).

---

## 2. Current state

### 2.1 Agents

| # | Agent | Status | Notes |
|---|-------|--------|-------|
| 1 | Master Agent | Done, changed | Handoff driver (`_drive`) replaces the fixed stage list and the separate rerun loop. Not run live yet. |
| 2 | Intake Agent | Done | Produces `target_terms` for the target-fit checks. |
| 3 | Research Agent | Done | Uses `web_search.py` (Brave, then SerpApi), OpenAlex, Semantic Scholar. |
| 4 | Validation & Parent Paper Selection | Done, has a defect | See 4.2: it can select a literature review as the Parent Paper. |
| 5 | Data Agent | Done | Reads the Parent Paper itself. `parent_only` / `expand` modes. |
| 6 | Cleaning Agent | Done | Label-aware merge, verified column mapping. |
| 7 | Code Discovery Agent | Done | Finding nothing is a valid result. |
| 8 | Replication Agent | Done | Clones the repository, or reconstructs the code from the paper. Isolated venv. |
| 9 | Evaluation Agent | Done | Comparison against the paper, plus a senior data analyst review and leakage checks. |
| 10 | Experiment Agent | Done | Expert data scientist. It requests reruns, and it does not run code. |
| 11 | **Report Agent** | **Not started** | `src/agents/report_agent.py` is empty. |
| 12 | **Presentation Agent** | **Not started** | `src/agents/presentation_agent.py` is empty. |

### 2.2 Infrastructure

| Part | Status |
|------|--------|
| Run database (`database/project_runs.db`, SQLite, `run_store.py`) | Done. It keeps every run, every agent attempt and a copy of every artifact. Local only. 9 runs recorded. |
| Project state (`logs/project_state.json`) | Done. The next run overwrites it. |
| Slack reporting | Done. Rate limits (`ratelimited`) were seen in the last live run. Posts fail soft. |
| Dashboard (`python -m src.dashboard`) | Done. It reads only what the pipeline writes. |
| LLM providers | OpenRouter, FreeLLMAPI (used live), Anthropic (implemented, never run live). |
| Web search | Brave, then SerpApi. Bright Data opens a link only when the site blocks it. |
| Tests | The last full run passed 600 tests (`.venv\Scripts\python.exe -m pytest -q`). One encoding test was added after that and passes. All network and LLM calls are mocked. |

### 2.3 Recent runs

* `20261109T224256Z-92d4c899`: **crashed** after the Experiment Agent
  finished. Cause: the agent report was written in the Windows default
  encoding (cp1252) and could not encode the PDF ligature "ﬁ" (U+FB01).
  **Fixed**: `_write_report` now writes UTF-8. A test covers it. The
  dashboard ran older code; restart it to get the fix.
* `20261109T210501Z-908ef5b8`: success, phase EXPERIMENTATION.

---

## 3. From a linear pipeline to an agent swarm

### 3.1 What is done

* `src/orchestration/handoff.py`: the request format, the prerequisites of
  each agent (`PREREQUISITES`), the phase of each agent and the result key
  of each agent.
* Each agent module has `handoff(result, context)`. It returns the request
  for the next agent, with the inputs to pass on.
* Master Agent's driver (`_drive`) starts from its own request for the
  Intake Agent. For each request it:
  1. checks the request (`_review_handoff`): known agent, no skipped
     prerequisite, and no second run outside an approved rerun;
  2. runs the agent under `_supervise` (check, diagnose, retry);
  3. asks that agent for its handoff.
* Reruns use the same path. The Evaluation Agent or the Experiment Agent
  requests a rerun. Master Agent checks the request, saves a snapshot,
  runs the requested agent and follows the handoffs until the Evaluation
  Agent has evaluated the new baseline. Then Master Agent keeps the result
  or undoes it.

### 3.2 What is not done yet

The driver can run a swarm, but the agents still act like a line:

1. **Each handoff gives one fixed next agent.** No agent selects among
   several possible next agents from its own result, except the
   Evaluation Agent and the Experiment Agent.
2. **No parallel work.** The Data Agent and the Code Discovery Agent need
   only an approved Parent Paper, so they could run at the same time.
   The driver runs one request at a time.
3. **Agents cannot ask for help.** For example, the Replication Agent
   cannot request a different dataset from the Data Agent. Only the
   Evaluation Agent and the Experiment Agent can request reruns.
4. **The Slack request relay (`check_agent_requests`) is a separate,
   older path.** It must be moved into the handoff driver so that all
   requests go through one check.
5. **No memory between runs.** This is the CBR work in section 6.

Recommended order: do item 4 first (one request path), then CBR (section 6).
Then let more agents send requests (item 3), and add parallel work (item 2)
last, because it changes the file snapshots and the run database records.

---

## 4. Started, not finished

### 4.1 Work that is not committed

The working tree has many changes that are not committed. They include
the handoff driver, the `handoff()` functions in every agent, the new search
modules (`web_search.py`, `serpapi_search.py`, `page_fetch.py`,
`paper_access.py`, `pdf_links.py`, `content_validator.py`), and these new
tests: `test_master_handoffs.py`, `test_master_reruns.py`,
`test_page_fetch.py`, `test_paper_access.py`, `test_pdf_links.py`,
`test_web_search.py`.

**Action:** review the changes and commit them in logical parts, for
example: (1) web search providers and page fetch, (2) handoffs and the
driver, (3) the UTF-8 report fix. `src/models.py` stays local and is not
committed.

### 4.2 Known defects and gaps

| Item | Detail | Action |
|------|--------|--------|
| The Parent Paper can be a literature review | The crashed run selected *"Financial fraud detection through the application of machine learning techniques: a literature review"*. A review has no experiment of its own to reproduce, so the evaluation was INCONCLUSIVE and the Experiment Agent stopped. | Add a check to the Validation Agent and to Master Agent: the Parent Paper must report its own experiment (dataset, method, results). Reject reviews and surveys as Parent Paper candidates. They can stay in the literature set. |
| Handoff driver not run live | Tested with mocked agents only. | Do one full live run after the dashboard restart. |
| `.env.example` is not complete | It has no `BRIGHTDATA_API_KEY` or `BRIGHTDATA_UNLOCKER_ZONE`, but CLAUDE.md says it is the authoritative list. | Add the two variables. |
| Console output encoding | Python output on Windows is cp1252. Printing paper text can raise the same error as the crash. | Set `PYTHONIOENCODING=utf-8` for dashboard runs, or reconfigure `sys.stdout` to UTF-8 in `run.py`. |
| Slack rate limits | Many `ratelimited` errors in the last run. | Post fewer messages per stage, or retry after the `Retry-After` time. |
| Anthropic provider | Implemented, never run live. | One short live call when API access is available. |

---

## 5. The two new agents

Both agents use only stored artifacts. They never invent results (see the
Results Integrity policy in CLAUDE.md).

### 5.1 Report Agent (Milestone 13)

* **Input:** the run's records in the run database: papers, Parent Paper,
  dataset pool, cleaning result, code search result, replication output,
  evaluation comparison, every rerun with its effect on the gap, every
  Experiment Agent analysis, and the models that wrote each script.
* **Output:** `reports/<run_id>/report.md` (and a PDF or DOCX if needed),
  with the sections in CLAUDE.md: abstract to references. Tables and
  graphs come from the stored metrics.
* **Master Agent checks:** every number in the report occurs in a stored
  artifact; every reference is a validated paper; the report names the
  models that generated code; reconstructed code is not called the
  authors' code; a sample is not called the full dataset.
* **Handoff:** the Experiment Agent's "stop" requests the Report Agent
  (today it requests nothing). The Report Agent requests the Presentation
  Agent.

### 5.2 Presentation Agent (Milestone 14)

* **Input:** the approved report and its figures.
* **Output:** `presentations/<run_id>/slides.pptx`, with the 16-part
  structure in CLAUDE.md. Use more visuals and less text.
* **Master Agent checks:** each slide's numbers match the report; the
  slide count and sections are complete.
* **Handoff:** requests nothing. The run ends here (phase COMPLETE).

---

## 6. Case-Based Reasoning (DS-Agent, Guo et al. 2024)

Source: Siyuan Guo, Cheng Deng, Ying Wen, Hechang Chen, Yi Chang, Jun
Wang. *DS-Agent: Automated Data Science by Empowering Large Language Models
with Case-Based Reasoning.* ICML 2024, PMLR 235. Code:
https://github.com/guosyjlu/DS-Agent. (Local copy: `guo24b.pdf`.)

### 6.1 The idea in the paper

CBR solves a new task with the solutions of similar past tasks. It does not
update model weights. DS-Agent has two stages:

**Development stage** (an iteration loop):

1. **Retrieve:** compute the cosine similarity between an embedding of
   the task description and each case in the case bank. Take the top-k
   cases.
2. **ReviseRank:** an LLM reorders the k cases by their estimated
   usefulness, using the feedback from the last execution. The output is
   a ranking in the form `[2] > [1] > [3]`.
3. **Reuse:** a Planner adapts the top case into an experiment plan for
   the current task.
4. **Execute:** write and run the code, and record the result.
5. Repeat from step 2 with the new feedback (the revise loop).
6. **Retain:** store the solution with the best result in the agent case
   bank for later reuse.

**Deployment stage** (low resources): retrieve a similar *past successful
solution* and adapt it in a single pass, with no revise loop.

The paper reports that this makes code generation more reliable, also for
weaker LLMs. This is important for us because FreeLLMAPI sends calls to
many different free models.

### 6.2 How CBR fits this project

Our system already has the loop. The Evaluation Agent gives the feedback,
the Experiment Agent requests the revision, and Master Agent decides what
to keep. Retrieve, ReviseRank and Retain are the parts that are missing.

| DS-Agent step | Our system |
|---------------|------------|
| Task description τ | Topic, `target_terms`, Parent Paper metadata, dataset profile (rows, columns, class balance), agent name. |
| Case bank | A new `cases` store, filled from old runs (section 7) and from each new run. |
| Retrieve | New `src/orchestration/case_bank.py`: `retrieve(agent, task, k)`. Similarity × quality score (section 8). |
| ReviseRank | One LLM call that reorders the k cases with the latest Evaluation feedback. Skip it on the first attempt, as the paper does (empty feedback). |
| Reuse | The top case goes into the agent's prompt as context, in the same way as `llm.supervisor_notes` (for example `llm.case_context`). No agent signature changes. |
| Execute | No change: `_supervise` and the handoff driver. |
| Retain | At the end of each run, Master Agent writes one case for each approved stage output, with its score. |
| Deployment stage | A "fast mode": when a very similar, high-scored case exists, the agent adapts it in one pass. |

**Agents that get the most from cases:**

* **Replication Agent:** reconstructed scripts that ran and scored close to
  the paper. This is the nearest match to DS-Agent.
* **Experiment Agent:** rerun requests and their effect on the gap (kept,
  reverted, improved). It must learn which changes helped.
* **Data Agent and Code Discovery Agent:** which links worked for a
  Parent Paper or dataset name (this extends the current caches).
* **Validation Agent:** earlier Parent Paper choices and their result
  (for example, "a literature review gave INCONCLUSIVE").
* **Evaluation Agent:** findings that were verified and fixed the problem.

### 6.3 Integrity rules for cases

* A case is a **hint, never a source**. A paper, dataset, URL or code
  repository from a case must be verified again in the current run.
* **Never reuse a metric or result from a case.** Results come only from
  the current run's execution.
* A failed case can be retrieved only as a **warning** ("this did not
  work"). It is never a solution to copy.
* The report must name each case that an agent reused (run id and stage).

### 6.4 Embeddings: decision needed

The project has no embedding model yet. Options:

1. A local model (for example `sentence-transformers`). Free and
   reproducible, but it adds a large dependency.
2. An embedding API (OpenRouter or another provider). It adds a cost and
   another network call.
3. TF-IDF from scikit-learn as the first version. No new dependency, and
   it is sufficient for short, structured task descriptions.

Recommendation: start with option 3 behind a `similarity()` interface, and
change it later if retrieval quality is low.

---

## 7. Plan for old runs

The run database holds 9 runs. They were made with different code
versions, so they are not all equally useful.

1. **Record the version of every case.** `runs.project_git_commit`
   exists already. Add `schema_version` and `code_version` to the `cases`
   records, so that cases from old code can be filtered or weighted lower.
2. **Classify each old run:**
   * `success`: all approved stages become cases.
   * `halted`: the stages before the halt become cases. The stage that
     failed becomes a *negative* case (a warning), with its rejection
     reason from the Master Agent log.
   * `crashed`: approved stages become cases. The crashed stage has no
     record and is skipped (for example, the Experiment Agent in the
     crash of 2026-11-09).
   * Runs from before the Data Agent's target-fit check, or with a
     literature review as the Parent Paper, are marked `low_trust`.
3. **Backfill once.** A script (`python -m src.orchestration.case_bank
   backfill`) reads `runs`, `stages` and `artifacts` and writes the cases.
   It must not change the run records: the run database stays the
   permanent record, and the case bank is built from it.
4. **Retention:** keep all runs. The case bank can drop or weight down
   cases that are old, low-scored or `low_trust`, but the raw run record
   stays.
5. **Duplicates:** many runs use the same topic. When several cases have
   the same task and the same solution, keep the one with the best score
   and count how many times it was seen.

---

## 8. Ranking agent outputs

Each stored output gets a **quality score**, so that retrieval favours
good cases and every agent learns from what worked. Use only signals that
the system already records, so the score is deterministic and not an LLM
opinion.

### 8.1 Signals

| Signal | Where it comes from | Meaning |
|--------|---------------------|---------|
| First-attempt approval | `stages.attempt`, `stages.decision` | Approved on attempt 1 is better than after retries. Rejected is negative. |
| Checks passed | Master Agent log | Number and type of issues found. |
| Downstream success | the run's handoff chain | Did the run reach evaluation with this output? |
| Gap to the paper | `score_gap(evaluation_result)` | The final gap to the published score. Smaller is better. |
| Rerun effect | `reruns`: `applied`, `improved`, `Reverted` | For Experiment Agent and rerun cases. |
| Validity | Evaluation Agent leakage and verified findings | A leaky result is not a good case, even if the score is close. |
| Human rating (optional) | dashboard button, later | A person marks a case as good or bad. |

### 8.2 Credit along the handoff chain

The final gap depends on many agents. Give credit back along the chain:
the agent that produced the final evaluated baseline gets the full credit,
and each earlier agent gets less (for example, multiply by 0.7 for each
step back). A rerun gives its credit to the requested agent and to the
agent that requested it.

### 8.3 Proposed first formula

```
quality = 0.4 * outcome      # 1 - min(gap, 20) / 20; 0 when no gap is measurable
        + 0.2 * approval     # 1 on attempt 1, 0.5 after retries, 0 if rejected
        + 0.2 * validity     # 0 if verified leakage, else 1
        + 0.2 * rerun_effect # 1 improved, 0.5 kept, 0 reverted or failed; 0.5 when no rerun
```

The weights are a first proposal. Test them on the backfilled cases, and
check that the ranking looks correct to a person before any agent uses it.

### 8.4 How the score is used

* Retrieval ranks cases by `similarity × quality` (step 1 of CBR).
* ReviseRank receives the score and the reason for it.
* Negative cases (low score) go to the agent as warnings.
* The dashboard shows a table of the best cases for each agent, so the
  team can see what the system learned.

---

## 9. Backend: decision needed

Today all data is local: the SQLite run database, the JSON logs and the
files. A case bank that all agents and team members share needs a backend.
The team must decide which one. Options:

| Option | For | Against |
|--------|-----|---------|
| Keep SQLite, add a `cases` table | No new service. Simple. | One machine only. Team members cannot share cases. |
| Hosted Postgres (for example Supabase or Neon) | Shared by the team. Can store vectors (pgvector) for retrieval. | Needs an account, credentials in `.env`, and network access in runs. |
| A small API service in front of the database | Agents do not need database credentials. Dashboard and pipeline use one interface. | More code to build and run. |

Recommendation: write the case bank behind one interface
(`case_bank.py`: `retain`, `retrieve`, `backfill`), first with SQLite.
Then add a hosted Postgres adapter when the team needs to share cases.
The pipeline must still run, and the tests must pass, with no backend
connected. Use the same fail-soft rule as Slack. Credentials go only in
`.env`.

---

## 10. Next steps, in order

1. **Commit the current work** in logical parts (section 4.1).
2. **Restart the dashboard and do one live run** with the handoff driver
   and the UTF-8 fix.
3. **Stop literature reviews from being selected as Parent Paper** (4.2).
4. **Fix the small gaps:** `.env.example` Bright Data variables, UTF-8
   console output, Slack rate limits.
5. **Move the Slack request relay into the handoff driver** (3.2, item 4).
6. **Report Agent** (5.1), with the Experiment Agent's "stop" handing off
   to it.
7. **Presentation Agent** (5.2).
8. **Decide the backend and the embedding method** (sections 9 and 6.4).
9. **Build `case_bank.py`:** the `cases` schema, `retain`, `retrieve`
   (TF-IDF first) and the quality score (section 8).
10. **Backfill the old runs** (section 7) and check the ranking by hand.
11. **Connect CBR to the agents:** first the Replication Agent and the
    Experiment Agent, then the others. Add ReviseRank. Measure whether
    runs with cases get closer to the paper's score than runs without
    them. This comparison is also a result for the report.
12. **Extend the swarm:** more agents send requests (3.2, item 3), then
    parallel work (item 2).

## 11. Open questions for the team

* Which backend: SQLite only, or hosted Postgres now?
* Which embedding method?
* Do we want a human rating button in the dashboard for the ranking?
* Report format for submission: Markdown, PDF or DOCX?
* Must the CBR comparison (with and without cases) be part of the final
  report as an experiment?
