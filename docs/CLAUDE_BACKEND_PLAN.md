# Agent Memory Backend: Plan

Date: 2026-11-09 (updated 2026-11-13: run metrics, section 11; then
points taken from `docs/CODEX_BACKEND_PLAN.md`: validity counts as
success (9.1), baseline lineage (8.5), events saved as they happen (8.1),
memory states (8.4), "used" is a self-report (9.3), assigned vs. actual
memory exposure (11.3), frozen memory snapshot (11.8), retrieval test set
(9.6))
Author: Claude (planning only; no code is written in this document's step)

This plan describes a shared memory for all agents. With it, each agent can
look at its past work across every run of every team member, and learn
from its successes and failures. It replaces sections 6-9 of
`docs/PROJECT_STATUS.md` where they disagree.

Decisions already made by the user (2026-11-09):

* The memory is **shared by the team**.
* A **free hosted account** (Postgres) is permitted. Credentials go in `.env`.
* Memory can do more than give hints:
  * agents can **reuse** a good past solution as a starting point, and
  * memory can **tune pipeline choices**.
* Agents must be able to **call a memory function themselves**, and
  their prompts must tell them to use it when possible. The example: the
  coding agent must write an XGBoost script. It calls memory and gets
  good and bad past XGBoost scripts, ordered by the ranking system.
* DS-Agent (Guo et al., ICML 2024, `guo24b.pdf`) is the inspiration, not
  the specification.

---

## 1. Goals and non-goals

### 1.1 Goals

1. Every agent can **recall** relevant past work: solutions, failures and
   lessons. It can do this before it acts and while it acts.
2. Every run **retains** what it learned, automatically, at the end of the
   run (also after a halt or a crash).
3. Each memory has a **quality rank** from recorded evidence, so agents
   see the best examples first and the worst ones as warnings.
4. The rank **learns from use**: when a memory is used and the result is
   good, its rank goes up. When the result is bad, its rank goes down.
5. The memory is **shared**: one team member's run helps the next run of
   any team member.
6. The pipeline **still runs with no memory backend** (offline, tests,
   no credentials), in the same fail-soft way as Slack.
7. Every run **records a full set of run metrics** (result quality,
   reliability, cost and time, memory use) and the conditions it ran
   under, with or without memory (section 11).
8. We can **compare runs** with each other, and **measure** whether memory
   helps (memory on vs. off). This is also a result for the final report.

### 1.2 Non-goals

* Memory never replaces verification. A paper, dataset, URL, repository
  or metric from memory must be verified again in the current run.
* Memory never supplies a result. Metrics in a report come only from the
  current run's execution.
* No model fine-tuning. Learning is by retrieval and ranking only (the
  same approach as DS-Agent).
* Datasets, CSVs and PDFs are **not** uploaded to the shared memory (size,
  licenses, privacy). They stay in the local run database.

---

## 2. Facts about the current system that shape the design

| Fact | Effect on the design |
|------|----------------------|
| The local run database (`database/project_runs.db`) is about 300 MB. Almost all of it is CSV copies (about 1.5 GB of CSV before compression). Scripts total about 50 KB. | Upload only text: scripts, plans, summaries and decisions. A free hosted tier (about 0.5 GB) is sufficient for thousands of runs. |
| 10 runs are recorded: 3 success, 4 halted, 2 crashed, 1 still marked running. They cover only **2 distinct topics**. | The memory starts almost empty ("cold start"). The design must give good results with few cases, and should support seed cases (section 12). |
| Stage records already have `agent`, `attempt`, `decision` (Approved / Retrying / Rejected), `output_json` and `report_text`. Master Agent's diagnoses are in the stage report text and the Master Agent log. | Most memories can be built from records that exist already. Backfill is possible. |
| No agent uses native LLM tool calling. Each agent calls `llm.chat()` once per decision and parses JSON or code from the answer. FreeLLMAPI sends calls to many free models. Their tool-calling support is not consistent. | The memory "tool" must work **without** native tool calling: a text protocol that every model can follow, plus a deterministic recall that the agent code does itself. |
| `llm.supervisor_notes` adds text to the system prompt of every call during a retry, and it is process-wide. | Memory context must be **per call** (an argument), not process-wide, because the Research and Validation Agents call `chat()` from worker threads and each call needs different memories. |
| `tests/conftest.py` blocks all network access. | Tests use a local backend. No test reaches the hosted database. |
| The repository is public. | No credentials in the code. Memory content must be scrubbed of secrets before upload. |

---

## 3. Architecture

```text
            ┌──────────────────────────── each agent ────────────────────────────┐
            │  1. prefetch: memory.recall(agent, task)    (deterministic, in code)│
            │  2. llm.chat(prompt, memory=pack)  ── model may answer              │
            │        {"memory_request": {...}}  → llm runs recall → asks again    │
            │  3. agent output includes "memory_used": [ids]                      │
            └─────────────────────────────────┬───────────────────────────────────┘
                                              │ recall / record_usage
                                    ┌─────────▼─────────┐
                                    │  src/memory/ API  │  recall · retain · record_usage
                                    │  (one interface)  │  rate · backfill · stats
                                    └───┬───────────┬───┘
                              online    │           │  offline / tests
                     ┌──────────────────▼──┐   ┌────▼──────────────────────┐
                     │ Hosted Postgres     │   │ Local SQLite              │
                     │ (Supabase)          │◄──┤ database/memory_local.db  │
                     │ shared by the team  │   │ read cache + write outbox │
                     └─────────────────────┘   └───────────────────────────┘
                                              ▲
         run end: Master Agent ── memory.retain(run) ── curator (lessons) ── ranking update
```

Five parts:

1. **Memory store**: hosted Postgres, with a local SQLite mirror.
2. **Memory API** (`src/memory/`): the only module that touches the store.
   Agents call it, and it never calls agents.
3. **Recall in agents**: a deterministic prefetch, and a memory request
   that the model itself can send.
4. **Retain at run end**: Master Agent converts the run's records into
   memories, and a curator step writes short lessons.
5. **Ranking**: a deterministic quality score, changed by use.

---

## 4. Backend choice

### 4.1 Recommendation: Supabase (hosted Postgres)

| Need | Supabase | Neon (alternative) |
|------|----------|--------------------|
| Plain Postgres connection string | Yes | Yes |
| Full-text search (`tsvector`) | Yes | Yes |
| Vector search (`pgvector`) for later | Yes | Yes |
| Web table editor for the team to look at memories | Yes (good) | Basic |
| Row-level security and roles | Yes | Postgres roles only |
| Free-tier limits | About 500 MB database. A free project pauses after about 1 week with no activity. | About 0.5 GB. It scales to zero and wakes on connection. |

Supabase is recommended for the table editor and the access rules. The
code uses **only a standard Postgres connection** (`psycopg`), not the
Supabase SDK. Thus Neon or any other Postgres works with no code change.
**Check the current free-tier limits when the account is made**: they
change, and the values above are from memory and are not verified.

Pause risk: if the Supabase project pauses, recall fails. Because the API
is fail-soft and has a local cache (4.3), the run continues. The dashboard
shows "memory offline".

### 4.2 Access and credentials

* `.env` (never committed): `MEMORY_DATABASE_URL`, `MEMORY_BACKEND`,
  `MEMORY_MODE`, `MEMORY_CONTRIBUTOR` (section 13). Add them to `.env.example`.
* Make two database roles:
  * `memory_agent`: `SELECT` and `INSERT` on all memory tables, and
    `UPDATE` only on the counters and score columns. **No `DELETE`.**
    The pipeline uses this role.
  * `memory_admin`: full rights. It is for migrations and clean-up only,
    and it is never in a pipeline run's `.env`.
* With no `DELETE`, a defect or a bad model output cannot erase the team's
  memory. Bad memories are quarantined (`state = quarantined`, section
  8.4), not deleted.
* `MEMORY_CONTRIBUTOR` is a short name that each team member selects (for
  example `aiden`), not an email address.

### 4.3 Local mirror (`database/memory_local.db`, gitignored)

* **Read cache:** each successful recall stores its results locally. When
  the hosted store is offline, recall uses the cache. A quarantined
  memory must not stay usable through the cache: each `sync()` gets the
  current state of every cached memory and removes the ones that are not
  `active`. A cache that has not synced for more than 7 days is not used,
  and the run continues with no cross-run memory (the result records
  this as `memory_exposure = none`, section 11.3).
* **Write outbox:** `retain`, `record_usage` and `rate` write to the local
  outbox first. A sync step sends them to the hosted store (at run end,
  and at the start of the next run). A write is never lost if the network
  fails.
* **Tests and offline use:** `MEMORY_BACKEND=sqlite` uses only the local
  file, with the same schema and API. `tests/conftest.py` sets this, with a
  temp file.
* Each record has a UUID made on the client, so sync is idempotent: the
  same row sent twice is stored once.

---

## 5. What a memory is

### 5.1 Memory kinds

| Kind | What it holds | Example |
|------|---------------|---------|
| `solution` | A concrete output that was used: a script, a plan, a link, a query, a decision. | The Replication Agent's reconstructed XGBoost script, with its run output and its gap to the paper. |
| `failure` | An output that failed a check, raised an error, or was reverted, with the cause. | A script that applied SMOTE before the split; the Evaluation Agent's leakage finding; the fix that worked. |
| `lesson` | A short, general rule distilled from one or more runs, with links to the evidence. | "When the paper reports AUC on a highly imbalanced set, use stratified splits. A random split changed AUC by 4 points in run X." |
| `decision` | A Master Agent action: approval, rejection, diagnosis, rerun accepted, declined or reverted. | "Declined: Data Agent rerun redundant after 3 reruns with no gain." |

`solution` and `failure` come directly from records. `lesson` is written
by the curator (section 8.3) and must cite its evidence.

### 5.2 Task features (how memories are matched)

Each memory stores the **task** it belongs to as structured features, made
deterministically (no LLM) wherever possible:

| Feature | Source | Method |
|---------|--------|--------|
| `agent` | stage record | direct |
| `topic`, `target_terms` | Intake Agent result | direct |
| `task_type` | Intake Agent result, label column | binary / multiclass / regression / anomaly |
| `domain_tags` | Intake keywords | normalized lowercase words |
| `model_family` | the script (Python `ast` imports and calls, in the same way as `code_checks.py`), else the paper's method text | a fixed vocabulary: `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic_regression`, `svm`, `knn`, `mlp`, `cnn`, `lstm`, `transformer`, `autoencoder`, `isolation_forest`, ... |
| `techniques` | the script | `smote`, `undersampling`, `class_weight`, `standard_scaler`, `pca`, `grid_search`, `cross_validation`, `early_stopping`, ... |
| `dataset_traits` | Cleaning Agent result | row bucket, column count, minority-class ratio bucket, anonymized columns (yes/no) |
| `metrics` | Evaluation Agent result | metric names (accuracy, f1, auc, recall, ...) |
| `parent_paper` | Validation result | DOI and normalized title |
| `dataset_identity` | Data Agent result | resolved download link |
| `failure_signature` | traceback or check text | exception type plus a normalized message (numbers, paths and names removed) |
| `source` | code discovery | `cloned_repo` / `reconstructed` |

Fixed vocabularies make the features reliable across many free models. A
value outside a vocabulary is stored as `other:<word>`.

### 5.3 Content limits

* `content` (script, plan, JSON): at most 64 KB. Longer content is cut,
  and a note says so.
* `summary`: at most 1,000 characters. It is what the agent sees first.
* No dataset rows. A data sample is not a memory. Only the dataset
  profile (traits) is stored.
* A **secret scrubber** runs before every write. It removes any value of
  a variable in `.env`, and anything that looks like an API key, token,
  connection string, or email address.

---

## 6. Database schema (Postgres; SQLite mirror is the same)

Types are shown for Postgres. The SQLite mirror stores JSON as text and
arrays as JSON text.

### 6.1 `memory_runs`: one row per pipeline run

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | text PK | the same id as `runs.run_id` in the local run database |
| `contributor` | text | `MEMORY_CONTRIBUTOR` |
| `topic` | text | |
| `status` | text | success / halted / crashed |
| `final_phase` | text | |
| `git_commit` | text | `runs.project_git_commit` |
| `memory_mode` | text | off / read / read_write / pre_memory (for the A/B comparison) |
| `memories_recalled` | int | |
| `memories_used` | int | |
| `trust` | text | normal / low_trust (section 10) |
| `started_at`, `finished_at` | timestamptz | |

The run's metrics and conditions (gap, cost, attempts, ...) are in
`run_metrics` and `run_conditions` (section 11.5), not in this table.

### 6.2 `memories`: the case bank

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | made on the client |
| `kind` | text | solution / failure / lesson / decision |
| `agent` | text | canonical agent name |
| `run_id` | text FK | null for seed memories |
| `stage_attempt` | int null | |
| `polarity` | text | positive / negative / neutral |
| `title` | text | one line |
| `summary` | text | at most 1,000 characters |
| `content` | text | at most 64 KB |
| `content_sha` | text | for duplicates |
| `task` | jsonb | the features in 5.2 |
| `tags` | text[] | flat list of feature values for fast filters |
| `outcome` | jsonb | decision, gap before/after, leakage found, attempts needed, error, rerun effect |
| `evidence` | jsonb | references: run id, stage id, artifact sha, log entry time, exact quotes |
| `quality` | real | 0-1, section 9 |
| `quality_reasons` | jsonb | each component and its value, so the score can be explained |
| `trust` | text | normal / low_trust / seed |
| `schema_version` | int | memory schema version |
| `code_version` | text | git commit of the pipeline that produced it |
| `seen_count` | int | duplicates folded in |
| `times_recalled`, `times_used`, `times_helped`, `times_hurt` | int | section 9.3 |
| `state` | text | candidate / active / quarantined / superseded / retired (section 8.4); never deleted |
| `superseded_by` | uuid null | the memory that replaces this one |
| `invocation_id` | uuid null | the exact agent attempt that produced it (section 8.5) |
| `lineage` | text | kept / superseded / reverted / rejected / unknown (section 8.5) |
| `fts` | tsvector | generated from title, summary and tags |
| `embedding` | vector(384) null | milestone M8 only, if needed (section 9.5) |
| `created_at`, `updated_at` | timestamptz | |

Indexes: `(agent, kind)`, GIN on `tags`, GIN on `fts`, `(content_sha, agent)`.

### 6.3 `memory_usage`: every recall, and its result

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `run_id`, `agent`, `attempt` | | who recalled |
| `memory_id` | uuid FK | |
| `rank_shown` | int | position in the pack |
| `trigger` | text | prefetch / model_request |
| `query` | jsonb | the request as sent |
| `used` | bool | the agent named it in `memory_used`. This is the agent's **self-report**, not proof that the memory caused the result (section 9.3) |
| `supplied_digest` | text | hash of the exact text put in the prompt, so the context can be rebuilt even if the memory changes later |
| `stage_decision` | text | Approved / Retrying / Rejected for that attempt |
| `gap_delta` | real null | change in gap after the rerun or stage, when measurable |
| `helped` | bool null | set at run end (section 9.3) |

This table is the "learning signal". It records which memories were
shown, which ones were used, and what happened after.

### 6.4 `memory_ratings`: optional human ratings

| Column | Type | Notes |
|--------|------|-------|
| `memory_id`, `contributor` | PK | one rating per person |
| `rating` | int | -1 / +1 |
| `note` | text | |
| `created_at` | timestamptz | |

### 6.5 `memory_meta`

One row: `schema_version`, the vocabularies' version, and the active
ranking weights, so every machine uses the same values.

### 6.6 Migrations

Number the SQL files (`src/memory/migrations/001_init.sql`, ...). A
command applies them with the admin role. The API refuses to write when
the store's `schema_version` is newer than the code's version, and asks
the user to pull.

---

## 7. Recall: how agents use memory

### 7.1 Public API (`src/memory/__init__.py`)

| Function | Purpose |
|----------|---------|
| `recall(agent, task, query=None, kinds=None, k=5, include_failures=True) -> MemoryPack` | Find and rank memories. Never raises; returns an empty pack on any error. |
| `MemoryPack.as_prompt() -> str` | Text block for the prompt: good examples, then warnings, each with its id, score and reasons. |
| `record_usage(run_id, agent, attempt, pack, used_ids, decision)` | Store which memories were shown and used. |
| `retain_run(run_id)` | Make memories from a finished run (section 8). |
| `rate(memory_id, rating, note)` | Human rating. |
| `stats(agent=None)` | Best and worst memories for the dashboard. |
| `sync()` | Send the outbox and refresh the cache. |

### 7.2 Two ways to recall

**A. Prefetch (deterministic, always on).** At each decision point (table
7.4), the agent's code makes the task features and calls `recall()` before
its LLM call. This does not depend on the model following instructions,
so every agent gets memory even with weak free models.

**B. Model request (the "memory tool").** The prompt tells the model it can
ask for more memory. Because native tool calling is not reliable across
FreeLLMAPI models, the protocol is text:

* The model may reply with **only** a JSON object:
  `{"memory_request": {"agent": "...", "query": "...", "filters": {"model_family": "xgboost"}, "kinds": ["solution","failure"]}}`.
* `llm.chat` sees this form, runs `recall()`, adds the new pack to the
  conversation, and sends the call again.
* At most `MEMORY_MAX_REQUESTS_PER_CALL` (default 2) requests per call.
  After that, the prompt says "answer now", and a further request is
  treated as a bad answer (the agent's existing parse and retry handle it).
* With the Anthropic provider, the same request can use native tool use
  later, with the same `recall()` behind it. This is optional.

`llm.chat` gets one new optional argument, `memory=` (a pack or a recall
specification). It does not use process-wide state, because worker
threads call `chat()` at the same time.

### 7.3 Prompt rules (added to every agent prompt that has memory)

The block that `MemoryPack.as_prompt()` writes starts with these rules:

1. "These are past cases from earlier runs. They are ranked by measured
   quality. Use the good cases as a starting point when they fit the
   current task. Avoid the patterns in the warnings."
2. "A case is a hint, not a source. Do not copy a paper, dataset, URL,
   repository or metric from a case into your answer unless you verify it
   again in this run's inputs."
3. "Never report a number from a case as a result."
4. "If you need more cases, reply with only a memory_request object."
5. "In your answer, list the ids of the cases you used in `memory_used`
   (an empty list if none)."

Each case in the pack shows: id, kind, polarity, score (with its main
reasons, for example "approved on attempt 1; gap 0.8 points; no
leakage"), the task features, the summary and, for a solution, the
content (cut to a size budget per pack, default 12,000 characters).

### 7.4 Recall points per agent

| Agent | When | Prefetch features | What it gets | Reuse ("fast mode", 7.5) |
|-------|------|-------------------|--------------|--------------------------|
| Intake | before keyword extraction | topic words | past intake outputs for similar topics that led to approved Parent Papers; target terms that failed the target-fit check | no |
| Research | before query generation | domain tags, target terms | search queries that found qualifying papers; prohibited-source rejections | no |
| Validation | before ranking | candidate DOIs and titles | past outcome of each candidate as a Parent Paper (reproduced, inconclusive, review paper); past rejection reasons | **pipeline tuning** (7.6) |
| Data | before planning and searching | Parent Paper DOI, dataset name, target terms | links that downloaded and passed the target-fit check; links that failed and why | try the remembered working link first (still downloaded and checked) |
| Cleaning | before column mapping | dataset identities | verified column mappings for the same dataset pair; rejected mappings | reuse a verified mapping proposal (verification still runs) |
| Code Discovery | before the search | Parent Paper DOI and title | repositories verified or rejected for this paper | check the remembered repository first (still verified) |
| Replication | before generation, before repair, before revision | model family (from the paper's method text), task type, dataset traits, metrics, failure signature (for repair) | **good and bad past scripts for the same model family** (the user's XGBoost example); repairs that fixed the same failure signature | adapt the best script in one pass (DS-Agent's deployment stage) |
| Evaluation | before the analyst review | model family, techniques, dataset traits | verified findings that were real and were fixed; findings that were wrong | no |
| Experiment | before each decision | target agent, gap size, metrics, model family | past rerun requests with their effect on the gap (improved, kept, reverted, declined) | no |
| Master (diagnosis) | in `_diagnose` | agent, failure signature | past diagnoses for the same signature, and whether the retry then passed | no |
| Report / Presentation (when built) | before writing | topic, parent paper | section layouts that passed Master Agent's checks | no |

### 7.5 Reuse (fast mode)

When the best solution memory for the task has `quality ≥ 0.8`, the same
`model_family` and `task_type`, and similar dataset traits, the agent is
told to **adapt** it rather than write from nothing. For the Replication
Agent, this is DS-Agent's deployment stage: one generation call with the
past script as the template.

* The adapted output goes through every existing check (execution,
  leakage checks, evaluation). Fast mode skips nothing.
* If the adapted script fails, the normal repair loop runs, and the
  memory's rank goes down (section 9.3).
* A cloned repository is never replaced by a memory script.
  Preferred-implementation order in CLAUDE.md stays: authors' code first.
* `MEMORY_REUSE=false` turns fast mode off.

### 7.6 Pipeline tuning (behind `MEMORY_TUNING=true`, off by default)

Memory can change pipeline choices only in these bounded ways. Each one
is logged and posted to Slack as a Master Agent note:

| Choice | How memory changes it | Limit |
|--------|-----------------------|-------|
| Parent Paper ranking | Add a `past_reproducibility` term to the Validation Agent's composite score: higher for a paper that reproduced within 5 points, lower for one that was INCONCLUSIVE or a review. | At most ±10% of the composite score. It never makes a paper qualify or fail. The hard requirements are unchanged. |
| Data source order | Try remembered working links first; skip a link that failed the target-fit check in 2 or more runs. | A skipped link is logged. It is still allowed if nothing else is found. |
| Experiment requests | Master Agent declines a rerun request that is the same as one with `times_hurt ≥ 2` and no `times_helped`, as "known not to help". | The reason goes back to the Experiment Agent as feedback, as other declines do. |

Master Agent's rule "do not re-rank" stays true: the term goes into the
Validation Agent's own scoring, and Master Agent only checks it.

---

## 8. Retain: how runs become memories

### 8.1 When

* **As it happens:** after each agent attempt, `_supervise` writes the
  attempt's raw record (an "event": invocation id, agent, attempt,
  decision, failed checks, error, memory supplied) to the local outbox at
  once. Rerun opened, kept, reverted or declined, and baseline accepted,
  are events too. A crash or a killed process therefore loses nothing
  that happened before it. The events do not become memories yet.
* **At run end:** at the end of `run_pipeline` (success, halt, or crash
  in the `finally` block), Master Agent first calls
  `run_metrics.compute(run_id)` (section 11.6), then
  `memory.retain_run(run_id)`. Retain turns the events and the run's
  records into memories. Only then is the lineage (8.5) known, and the
  final gap for the base quality (section 9.1).
* **After a crash with no `finally`** (for example the process was
  killed): the next run's start, or `python -m src.memory sync`, finds
  runs with events but no `retain_run`, and retains them with
  `lineage = unknown` where the end of a rerun was not recorded.
* Retain reads the **local run database** (stages, artifacts, Master
  Agent log entries in the run's time span, `result["reruns"]`), so it
  works the same for new runs and for backfill.
* It must never stop or slow a run in a way the user notices. Retain runs
  after `finish_run`, and if it fails, it is logged and tried again at the
  next sync.

### 8.2 Rules: records → memories

| Record | Memory |
|--------|--------|
| Approved stage output | `solution`, positive |
| Retrying / Rejected stage output, plus Master Agent's diagnosis and failed checks | `failure`, negative, with the diagnosis in the summary |
| A stage that raised, with the traceback | `failure`, with `failure_signature` |
| A retry that passed after a diagnosis | `decision` (diagnosis → fixed), positive; linked to the failure |
| Replication script (each version, each repair) | `solution` or `failure`, with the run output, the leakage findings and the gap |
| Evaluation finding (verified) | `solution` if the fix was kept and the gap or validity improved; else neutral |
| Rerun (requested, declined, kept, reverted) | `decision`, polarity from the effect on the gap |
| Data link downloaded and accepted | `solution` for the Data Agent; rejected link (with reason) → `failure` |
| Verified / rejected column mapping | `solution` / `failure` for the Cleaning Agent |
| Code repository verified / rejected | `solution` / `failure` for Code Discovery |

Duplicates: if a memory with the same `agent` and `content_sha` exists,
increase its `seen_count` and add the new evidence. Do not add a row.
A duplicate from the same run, or from a copy of the same imported run,
is not an independent confirmation (9.1).

An approved stage output becomes a positive `solution` only when its
lineage is `kept` (8.5). An approved output from a reverted rerun is
stored as a neutral observation, never as an accepted solution.

Memories made by these rules start as `active` (8.4): each one is a direct
record of what happened, with its evidence.

### 8.3 Curator: lessons (one LLM call per run)

After the rules in 8.2, one LLM call ("curator") reads the run's new
memories and writes at most 5 `lesson` memories. Each lesson must:

* be a general rule that can help a later run;
* cite the ids of the memories it comes from, with an exact quote from
  each;
* pass a deterministic check: each quote must occur in the cited memory.
  A lesson that fails the check is **not stored** (the same rule as the
  Evaluation Agent's `_verify`).

Lessons start with a medium quality (0.5). They go up only through use
(section 9.3). The curator can be turned off (`MEMORY_CURATOR=false`)
when the LLM budget is low.

A lesson is a general claim made by an LLM, so it starts as `candidate`,
not `active` (8.4). A candidate lesson is not given to agents.

### 8.4 Memory states

| State | Meaning | Given to agents |
|-------|---------|-----------------|
| `candidate` | Not confirmed yet (curator lessons) | No |
| `active` | Usable | Yes |
| `quarantined` | Wrong, misleading or unsafe; stopped at once | No |
| `superseded` | Replaced by a newer memory (`superseded_by`) | No |
| `retired` | No longer applies (for example the code it describes was removed) | No |

* A candidate lesson becomes `active` when a person approves it
  (`python -m src.memory approve <id>`), or automatically when evidence
  from **2 independent runs** supports it (two different runs, not two
  attempts of one run and not two copies of one imported run).
* Anyone in the team can quarantine a memory
  (`python -m src.memory quarantine <id> "<reason>"`). Recall excludes it
  at once. A check can also quarantine a memory, for example when a
  memory's content fails the secret scrubber on a later scan.
* Each state change is an append-only record (who, when, why). A memory
  row is never deleted.
* When two active memories disagree under the same task features, both
  are linked (`evidence.contradicts`), their quality is lowered by 0.2,
  and the pack shows them together with the label "conflicting cases".

### 8.5 Baseline lineage

An approved stage is not always part of the result that was kept: a
rerun can approve a stage and then be reverted (CLAUDE.md, rerun loop).
Memory must know the difference.

* **Invocation id:** each agent attempt gets a UUID in `_supervise`. It
  goes into the event (8.1), the memory and `memory_usage`. Today
  `record_stage` returns a stage row id that the callers do not keep, and
  `get_stage_output` returns the latest attempt of an agent, which can be
  a rejected or reverted one. Neither is enough to find the exact attempt.
* **Lineage per attempt**, set at run end from the run's events (rerun
  opened/closed events carry what `result["reruns"]` holds):
  * `kept`: the attempt is part of the final kept baseline (every approved
    Experiment Agent analysis is kept: it is a decision, not a baseline part);
  * `superseded` (added in M3): it was accepted, then replaced by a later
    kept attempt of the same agent (for example a kept Replication rerun);
  * `reverted`: it was approved, but its rerun was undone (reverted, failed
    or declined);
  * `rejected`: it failed its checks;
  * `unknown`: the record is not complete (a crash or a killed process
    before the baseline was final, an attempt inside a rerun that never
    closed, or a backfilled run with no rerun history). `unknown` is never
    treated as `kept`.
* **Use:** only `kept` attempts become positive solutions and may be
  reused in fast mode (7.5). A reverted attempt can still give a useful
  warning or neutral observation (for example: "this change ran
  correctly but moved the score away from the paper").
* Evaluation Agent validity fixes that are kept although the gap gets
  larger (CLAUDE.md) are `kept`, and count as successful work (9.1).

---

## 9. Ranking

### 9.1 Base quality (deterministic, at retain time)

The base quality is the same for every agent, but each agent gives its own
`outcome` signal:

```
quality_base = 0.35 * outcome
             + 0.20 * approval      # 1 on attempt 1; 0.6 on attempt 2; 0.3 later; 0 if rejected
             + 0.20 * validity      # 0 if verified leakage or an UNVERIFIED claim was used; else 1
             + 0.15 * downstream    # 1 if the run reached a kept evaluation with this output; 0.5 halted later; 0 halted here
             + 0.10 * trust         # 1 normal; 0.5 low_trust; 0.7 seed
```

`outcome` per agent:

| Agent | `outcome` |
|-------|-----------|
| Replication, Experiment | `1 - min(gap, 20) / 20` for the evaluation that used it, **only for a valid baseline** (no verified leakage) and only when the metric definitions match (same metric, unit and split); else 0.5 (unknown). For a rerun: 1 if it improved the gap, 0.5 kept with no change, 0 reverted |
| Evaluation | **Not the gap.** 1 if its verified findings were real and their fix was kept (for example leakage found and removed), even if the gap then got larger; 1 for a correct `accept` with no verified defect later found; 0 if a finding was UNVERIFIED or later shown wrong |
| Data | 1 if the dataset passed target-fit and was merged; 0 if rejected |
| Cleaning | share of datasets merged with no conflict |
| Code Discovery | 1 if the repository ran; 0.5 found but not run; 0 rejected |
| Validation | 1 if the Parent Paper reached a measurable gap; 0.3 INCONCLUSIVE; 0 if a review paper or halted |
| Intake, Research | the same as `downstream` |
| Master (diagnosis) | 1 if the next attempt passed; 0 if not |

Rules for the gap (applied to all rows above):

* **Validity is success.** A change that makes a result valid (removes
  leakage, fixes a wrong split, uses the correct test set) is good work,
  also when the score moves further from the paper. A leaky score that is
  close to the paper is not a good result, and gives `validity = 0`.
* A small gap is not proof of a good method. The gap is used only when
  the reproduced and published metrics have the same definition, unit
  and split. Gaps of different metrics are never combined, and changing
  the metric is not an improvement.
* **Independent confirmation** means different runs, not several attempts
  in one run, and not copies of one imported run. Repeated recall, the
  agent's own praise, or self-citation never counts as evidence.

A **negative** memory (a failure) is not ranked by low quality alone. It
is ranked by **how useful the warning is**: a failure with a clear cause and
a known fix ranks higher as a warning than one with no cause.

### 9.2 Retrieval score

```
retrieval = relevance * (0.5 + 0.5 * quality_effective) * freshness
```

* `relevance` (0-1): exact feature matches first (agent, model family,
  task type, failure signature), weighted; then Postgres full-text rank
  over title, summary and tags for the free-text query. No embeddings are
  needed in the first version.
* `quality_effective`: section 9.3.
* `freshness`: 1.0 for the current `code_version`; 0.8 for an older
  version of the same schema; 0.5 for older schema versions.
* Only `active` memories are candidates (8.4).
* **The current run is excluded:** cross-run recall never returns
  memories made from the current run. Retries inside the run already get
  their feedback through `supervisor_notes` and the rerun history, and
  this prevents a run from confirming itself.
* **Snapshot:** recall returns only memories created before the run's
  `memory_bank_snapshot` (11.3). In a controlled test this is a fixed
  snapshot (11.8). Quarantines are applied at once, also inside a
  snapshot.
* The pack has two lists: **good** (positive, best first) and **warnings**
  (negative, most useful first). The default is 3 good and 2 warnings, so
  the agent always sees both, as the user asked.

### 9.3 Learning from use

At run end, for each `memory_usage` row with `used = true`:

* `helped = true` if the stage was approved on that attempt, its lineage
  is `kept` (8.5), and the result is valid; for the gap, the rules in 9.1
  apply (a validity fix helps even when the gap gets larger);
* `helped = false` if the stage was rejected, the script failed, the
  rerun was reverted, or the result had verified leakage;
* `helped = null` if the lineage is `unknown`;
* `times_used`, `times_helped` and `times_hurt` are increased.

Then:

```
use_score         = (times_helped + 1) / (times_used + 2)     # starts at 0.5; Laplace smoothing
quality_effective = w * quality_base + (1 - w) * use_score,   w = 1 / (1 + times_used / 3)
human             = +0.1 for each +1 rating, -0.1 for each -1 rating, capped at ±0.2
```

A new memory is ranked mostly by its base quality. After a few uses, it
is ranked mostly by the outcomes of the stages that used it. This is the
persistent version of DS-Agent's ReviseRank.

Limit of this signal: `used` is the agent's self-report, and `helped`
only says what happened after the memory was supplied. Neither proves
that the memory caused the result. The system can prove which text it
supplied (`supplied_digest`), not that the model relied on it. Reports
and the dashboard therefore say "used (self-reported)" and "outcome
after use", never "memory caused". A causal claim needs the controlled
comparison in section 11.8.

### 9.4 ReviseRank inside a run

On a retry or a rerun, the agent's recall includes the **latest feedback**
(failed checks, Evaluation Agent findings). The pack is re-ordered by one
short LLM call (DS-Agent's `[2] > [1] > [3]` format), as in the paper. The
call is skipped on the first attempt (no feedback yet). The answer is
checked: it must be a permutation of the shown ids, else the original
order is kept.

### 9.5 Embeddings (milestone M8, only if needed)

Start with feature match plus full-text search. Add `pgvector` with a
small local embedding model only if the measured recall quality is low
(section 11). Before that, check that the embedding library supports the
project's Python version (3.14). Every machine must use the same model
name and dimension (stored in `memory_meta`).

### 9.6 Retrieval test set

A fixed, hand-labelled test set measures whether recall returns the
right memories. It is made before ranking weights are tuned, so the
weights are not fitted to the answers.

* At least **30 queries** (a task description for one agent), across the
  agents that have memories. They include these cases: no matching
  memory; exact paper or dataset match; same topic but a different
  target; environment mismatch (other OS or package version); a failure
  warning that must be shown; two conflicting memories.
* For each query, a person labels which memories are acceptable, and
  which are known to be harmful (they must not be shown).
* Stored in `tests/memory_retrieval_cases.json` with the snapshot of the
  memory bank it was labelled on (memory ids only, no content), and run
  with `python -m src.memory benchmark`.
* Release gate (proposed targets, to be checked, not measured results):
  for at least 80% of the queries that have an acceptable memory, one is
  in the top 3; no known harmful memory is shown; every labelled critical
  warning is shown when its conditions match. The report gives the
  number of queries and each miss.
* Rerun after each change to ranking, features or vocabularies. A change
  that fails the gate is not used.

---

## 10. Old runs (backfill)

1. `python -m src.memory backfill` reads `database/project_runs.db`
   (read-only, `mode=ro`) and runs the same `retain_run` for each run that
   is not yet in `memory_runs`.
2. **Trust labels:**
   * `crashed` and `halted` runs are used. Failures are the most useful
     warnings.
   * The run still marked `running` with no recent activity is treated as
     `crashed`.
   * A run whose Parent Paper is a literature review or survey is
     `low_trust`.
   * Runs before the target-fit check (by `git_commit`) are `low_trust`
     for the Data Agent and the Cleaning Agent.
   * The low trust applies to the affected claims only, not to every
     stage of the run. For example, a review-paper run lowers trust in its
     reproduction claims, but a correct import-error fix from the same run
     keeps normal trust.
   * Lineage (8.5) is taken from `result["reruns"]` when it is recorded
     and is clear; otherwise it is `unknown`. A backfilled attempt with
     `unknown` lineage never becomes a positive solution.
   * A copy of the same run database imported twice (for example from
     two team members) is detected by run id and content hash, and does
     not add independent confirmations.
3. The backfill writes a report: the memories made per run and the score
   of each. A person checks the top and bottom 10 per agent before the
   agents use memory (the gate at the end of milestone M4).
4. The local run database is never changed. It stays the permanent
   record, and the memory is built from it.

---

## 11. Run metrics: record every run, compare runs

Added 2026-11-13. Every run records a full, fixed set of metrics, so that
runs can be compared with each other. The main comparison is runs **with**
memory against runs **without** memory, but the same records also compare
providers, code versions, topics and Parent Papers.

### 11.1 Rules

1. **Every run is recorded:** success, halted and crashed, and every
   memory mode, also `MEMORY_BACKEND=off`. A run with no memory is the
   control group, so it must be recorded in the same way as a run with
   memory. Thus run metrics do **not** depend on the memory backend.
2. **Deterministic only:** each metric is calculated in code from the run's
   own records (run database, logs, result). No LLM calculates or
   estimates a metric.
3. **Missing is not zero:** a metric that cannot be measured is stored as
   `null`, with a reason (for example `no published metric in the paper`).
   It is never stored as 0, and it is never guessed.
4. **The conditions are recorded too:** a comparison is valid only when
   the runs had the same conditions, or when the differences are known.
   Each run stores its conditions (11.3), next to its metrics.
5. **Written once:** the metrics of a finished run are not changed. If a
   calculation changes, the new values are written with a new
   `metrics_version`, and the old values stay.
6. **Fail-soft:** if the metrics step fails, the run still finishes, the
   error is logged, and `python -m src.orchestration.run_metrics compute
   <run_id>` can make the metrics later from the same records.

### 11.2 Where the metrics are stored

* **Local (always):** two new tables in the run database
  (`database/project_runs.db`): `run_conditions` and `run_metrics`
  (11.5). The run database is the permanent record, so its metrics are
  kept with its stages and artifacts. This changes the statement in
  section 14 "Run database: nothing changes".
* **Shared (when a Postgres backend is configured):** the same two tables
  in the hosted store. They are sent with the memory outbox (4.3) by
  `sync()`. They contain only numbers, names and ids, no content, so they
  are small. A run recorded with no backend is sent at the next `sync()`
  after a backend is configured.
* Note: `MEMORY_BACKEND` is **where** records are stored, and `MEMORY_MODE`
  is **whether agents use memory**. For a team A/B test, a run with no
  memory must use `MEMORY_MODE=off` with `MEMORY_BACKEND=postgres`. Then
  its metrics are shared, but its agents do not see memory.

### 11.3 Run conditions (one row per run)

| Condition | Source | Why it matters |
|-----------|--------|----------------|
| `run_id`, `contributor`, `started_at`, `finished_at` | run database, `.env` | identity |
| `topic`, `topic_key` | Intake input; `topic_key` is the normalized topic (lowercase, no punctuation, sorted words) | runs on the same topic are grouped |
| `parent_paper_doi`, `parent_paper_title` | Validation result | a gap is comparable only for the same Parent Paper |
| `dataset_identities` | Data Agent result (resolved links) | same data or not |
| `rows_used`, `sampled` | Cleaning Agent result | the row budget changes results |
| `code_source` | Code Discovery | `cloned_repo` / `reconstructed` |
| `model_family` | features (5.2) | |
| `git_commit`, `git_dirty` | run database; `git status` | pipeline version |
| `llm_provider` | run database | |
| `served_models` | `logs/llm_calls.jsonl` | share of calls per model that answered; FreeLLMAPI changes models from call to call, which is a confounder |
| `memory_mode`, `memory_backend`, `memory_reuse`, `memory_tuning`, `memory_curator` | `.env` | the **assigned** condition, set before the run starts |
| `memory_exposure` | `memory_usage`, recall errors | the **actual** condition: `none` (memory off, store offline, or no matching memory), `partial` (memory supplied to only some agent calls), `full` (supplied at every recall point that had a match). With a count per agent and the reason for each `none` |
| `memory_bank_size` | memory store at run start | number of visible `active` memories per agent; a run with memory on and an empty bank is not a real memory run |
| `memory_bank_snapshot` | memory store | the newest `created_at` that the run could see (fixed for a controlled test, 11.8), so the memory state can be rebuilt |
| `comparison_group` | `MEMORY_COMPARISON_GROUP` (.env) or null | the controlled test the run belongs to (11.8); null for an ordinary run |
| `seeds` | Replication result, script | random seeds used |
| `python_version`, `os` | runtime | |
| `metrics_version` | code constant | version of the metric calculations |

### 11.4 Metric catalog

Each metric has a name, a unit, a scope (`run` or one agent) and a
direction (`lower_better`, `higher_better` or `neutral`). The catalog is a
constant in code (`METRICS`), so all machines use the same definitions.
Its version is `metrics_version`.

**Result quality**

| Metric | Unit | Direction | Source |
|--------|------|-----------|--------|
| `run_status` | success / halted / crashed | | run database |
| `final_phase_reached` | index in the workflow states | higher | run database |
| `reached_measurable_gap` | 0/1 | higher | Evaluation Agent result |
| `final_gap_points` | percentage points (mean over matched metrics) | lower | last kept evaluation, `score_gap` |
| `final_gap_<metric>` | points, one per matched metric (accuracy, f1, auc, ...) | lower | `score_gap` |
| `baseline_gap_points` | points, first evaluation before any rerun | lower | first evaluation |
| `gap_closed_points` | `baseline_gap_points - final_gap_points` | higher | calculated |
| `reproduced_<metric>`, `published_<metric>` | value as stated | neutral | Evaluation Agent result |
| `evaluation_status` | EXACT / APPROXIMATE / FAILED / INCONCLUSIVE | | Evaluation Agent result |
| `verified_leakage_findings` | count, first and final baseline | lower | `code_checks.py` findings |
| `parent_dataset_found` | 0/1 | higher | Data Agent result |

**Reliability (per agent and for the run)**

| Metric | Unit | Direction | Source |
|--------|------|-----------|--------|
| `attempts` | count | lower | stages |
| `first_attempt_approved` | 0/1 | higher | stages |
| `retries`, `diagnoses` | count | lower | stages, Master Agent log |
| `exceptions` | count | lower | stage reports |
| `halted_at_agent` | agent name or null | | run database |
| `repair_attempts` | count, Replication only | lower | Replication result |
| `reruns_requested`, `reruns_declined`, `reruns_kept`, `reruns_reverted` | count | | `result["reruns"]` |
| `rerun_end_reason` | text | | result |

**Cost and time**

| Metric | Unit | Direction | Source |
|--------|------|-----------|--------|
| `run_seconds` | seconds | lower | run database |
| `agent_seconds` | seconds, per agent | lower | time between the agent's stage rows |
| `llm_calls`, `llm_failed_calls` | count, per agent and run | lower | `logs/llm_calls.jsonl` |
| `llm_input_tokens`, `llm_output_tokens` | tokens, per agent and run | lower | `logs/llm_calls.jsonl` |
| `llm_seconds` | seconds | lower | `logs/llm_calls.jsonl` |
| `search_calls` | count, per provider | lower | `logs/search_calls.jsonl` |
| `page_fetches`, `browser_fetches` | count | lower | `logs/page_fetch_calls.jsonl` |

**Memory use** (0 or null when memory is off, never left out)

| Metric | Unit | Direction | Source |
|--------|------|-----------|--------|
| `memory_exposure` | none / partial / full | | 11.3 |
| `recall_errors`, `recall_empty` | count | | memory API |
| `memories_recalled` | count, per agent | | `memory_usage` |
| `memories_used` | count, per agent (self-reported, 9.3) | | `memory_usage` |
| `recall_precision` | used / shown | higher | `memory_usage` |
| `memory_requests` | count of model `memory_request`s | | `llm.chat` |
| `memories_helped`, `memories_hurt` | count | | `memory_usage.helped` (9.3) |
| `fast_mode_used` | 0/1, per agent | | agent result |
| `tuning_actions` | count | | Master Agent log |
| `memory_prompt_chars` | characters added to prompts | lower | `llm.chat` |
| `memories_retained` | count of new memories from this run | | `retain_run` |

### 11.5 Schema (run database and hosted store)

`run_conditions`: one row per run, with the columns in 11.3 (`run_id` is
the primary key; lists and maps are JSON).

`run_metrics`: one row per metric value, in long form, so a new metric
needs no migration.

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | text FK | |
| `metrics_version` | int | |
| `scope` | text | `run` or the canonical agent name |
| `metric` | text | a name from the catalog |
| `value_num` | real null | numeric value |
| `value_text` | text null | for status values |
| `unit` | text | |
| `missing_reason` | text null | set when the value is null |
| `computed_at` | timestamptz | |

Primary key: `(run_id, metrics_version, scope, metric)`. Index on
`(metric, scope)`.

`memory_runs` (6.1) keeps only the memory links of a run (contributor,
mode, counts). Its `final_gap` column is removed, because `run_metrics`
holds it.

### 11.6 How the metrics are collected

* **Run id in every log line:** `run_store.start_run` sets the current run
  id for the process (one run per process, so a process-wide value is
  correct here, unlike the memory pack in 7.2). `llm._record_call`, the
  search log and the page-fetch log add `run_id` to each line. Today these
  logs have no run id, and lines can be matched to a run only by time
  span.
* **Per-agent cost:** the call logs also add the agent name where it is
  known (`llm.chat(..., agent=)` already has it). A call with no agent name
  is counted for the run only.
* **At run end:** in the same `finally` block as `retain_run` (8.1), and
  before it, Master Agent calls `run_metrics.compute(run_id)`. It reads the
  run database, the logs (lines with this `run_id`) and the result, and
  writes `run_conditions` and `run_metrics`. Then `sync()` sends them.
* **Backfill:** `python -m src.orchestration.run_metrics backfill` makes
  the metrics of the 10 recorded runs. Their log lines have no run id, so
  lines are matched by the run's time span. Such a run's conditions have
  `log_matching = time_span` (new runs have `run_id`). These runs are
  `memory_mode = pre_memory`: they are a valid "no memory" group only for
  the same code version, so the comparison tool shows them separately by
  default.

### 11.7 Comparing runs

CLI: `python -m src.orchestration.run_metrics compare`, with filters
(`--topic`, `--parent-paper`, `--since`, `--contributor`) and a grouping
(`--group-by memory_mode`, the default; or `llm_provider`, `git_commit`,
`code_source`, ...). Output: a table and a CSV/JSON file in `reports/`.

Rules of the comparison:

1. **Same task only:** by default, runs are compared only when they have
   the same `topic_key` and the same Parent Paper. Runs with a different
   Parent Paper are listed apart, not mixed into the same mean.
2. **Paired by topic:** for each topic, the groups are compared, and then
   the per-topic differences are combined. One easy topic must not hide
   the result of the others.
3. **Show n and spread:** each group shows the number of runs, the median,
   the mean, the minimum and maximum, and the interquartile range. A group
   with fewer than 3 runs is marked "too few runs".
4. **A test only when there are enough runs:** with at least 3 runs per
   group, a permutation test (or Mann-Whitney U) and an effect size are
   shown. The tool never says "memory helps" by itself. It shows the
   numbers and the test, and states the limits.
5. **List the confounders:** the tool prints each condition that differs
   between the groups (for example a different `git_commit`, a different
   mix of `served_models`, a different `rows_used`, an empty memory bank
   for a memory run).
6. **Halted and crashed runs stay in:** they count in reliability
   metrics, and they are not removed to make a group look better. Gap
   metrics use only runs that reached a measurable gap, and the tool says
   how many runs that is.
7. **Group by assigned condition:** runs are grouped by the
   `memory_mode` they were assigned before they started, never moved to
   another group after their results are known. Each run also shows its
   actual `memory_exposure`. A memory-on run that got no memory (outage,
   empty bank) stays in the memory-on group, labelled `exposure: none`. A
   second table grouped by actual exposure can be shown, with a note that
   it is a selected subset and can be biased.
8. **Unknown is not a control:** a backfilled run with unknown memory
   settings is not counted as a memory-off run (`pre_memory` is shown
   apart, see 11.6).
9. **Ordinary runs vs. controlled tests:** any runs can be compared to
   describe differences, but only a controlled test (11.8) can support a
   claim that memory caused a change. The tool prints which kind each
   comparison is.

### 11.8 A/B protocol for memory on vs. off

**Before the test starts**, write down (in
`reports/comparisons/<group>.json`): the test topics, the conditions, the
number of runs per condition, the primary metrics, and the exclusion
rules. They are not changed after results are seen.

* **Conditions:** `MEMORY_MODE=off` (no cross-run memory) and
  `MEMORY_MODE=read` (memory supplied, bank frozen). Optional third
  condition: memory with good cases only, no warnings
  (`MEMORY_WARNINGS=false`), to measure the value of the warnings.
* **Frozen snapshot:** all memory runs of the test use the same
  `memory_bank_snapshot`, taken before the first test run. The test runs
  do not add to that snapshot: a test run's memories are retained, but
  are not visible to the other runs of the same test. Thus the order of
  the runs does not change what memory they see.
* **Held-out topics:** the test topics must not already be in the bank.
  Memories made from earlier runs on the same topic are excluded from the
  snapshot for this test, so that a run cannot get the answer from an
  earlier run of the same task.
* **Same conditions:** the same code commit, provider and settings in all
  conditions. Current-run retry feedback (`supervisor_notes`) stays on in
  all conditions. Record any difference (11.3).
* **Repeats:** at least 3 runs per topic and condition, because
  FreeLLMAPI changes models from call to call. Alternate the order (off,
  on, off, on, ...) so that a change in time (for example a free model
  that stops) does not affect only one group.
* **Learning curve (a separate, descriptive test):** a series of
  `read_write` runs where the bank grows, with results shown against
  `memory_bank_size`. It shows a trend, but it is not a controlled test,
  because the order of the runs and the bank change together.
* **What to report:** per-run results first, then per-topic differences
  and group summaries with n, spread, completion and validity rates and
  missing values. Measure valid-execution rate, repeated known failures,
  leakage rate, verified corrections, final gap where comparable, time,
  LLM calls and tokens. Learning is never summarized by one gap number.
* The comparison table is an experiment for the final report. The Report
  Agent reads it only from `run_metrics` and never writes its own numbers.

### 11.9 Dashboard

A read-only **Compare runs** view: choose runs or a grouping, see the
table from 11.7, and see a chart of one metric over runs (for example
`final_gap_points` against `memory_bank_size`). It only reads the run
database, so it follows the dashboard rule "write only to
`dashboard_files/`".

---

## 12. Cold start (seed memories)

With only 2 topics recorded, early recall finds little. Options, in order
of preference:

1. **Run more topics** with `read_write` (the simplest and most honest
   method).
2. **Seed lessons written by the team**, marked `trust = seed`, each with
   a real source (for example a scikit-learn documentation page). Seeds
   never count as results.
3. **DS-Agent's public case bank** (github.com/guosyjlu/DS-Agent).
   Before using it: check its license and its content, and record the
   commit used. This is `UNVERIFIED` until someone checks it.

---

## 13. Configuration (`.env`, add to `.env.example`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MEMORY_BACKEND` | `sqlite` | `postgres` / `sqlite` / `off` |
| `MEMORY_DATABASE_URL` | (empty) | Postgres connection string for the `memory_agent` role |
| `MEMORY_MODE` | `read_write` | `off` / `read` / `read_write` |
| `MEMORY_CONTRIBUTOR` | (empty) | short name of the team member |
| `MEMORY_REUSE` | `true` | fast mode (7.5) |
| `MEMORY_TUNING` | `false` | pipeline tuning (7.6) |
| `MEMORY_CURATOR` | `true` | lessons (8.3) |
| `MEMORY_MAX_REQUESTS_PER_CALL` | `2` | model requests per LLM call |
| `MEMORY_PACK_CHARS` | `12000` | size budget of one pack in a prompt |
| `MEMORY_WARNINGS` | `true` | include warnings in packs (off only for the test condition in 11.8) |
| `MEMORY_SNAPSHOT` | (empty) | fixed snapshot time for a controlled test (11.8); empty = now |
| `MEMORY_COMPARISON_GROUP` | (empty) | name of the controlled test this run belongs to |

Run metrics (section 11) have no switch: they are always recorded.

New dependency: `psycopg[binary]` (Postgres client). Nothing else in the
first version.

---

## 14. Module layout (proposed)

```text
src/memory/
├── __init__.py        public API (7.1); fail-soft wrapper around the backend
├── __main__.py        CLI: migrate | backfill | sync | stats | benchmark | show <id> | approve <id> | quarantine <id> <reason>
├── backends/
│   ├── base.py        the interface each backend implements
│   ├── postgres.py    psycopg; hosted store
│   └── sqlite.py      local mirror, cache, outbox; also the test backend
├── migrations/        001_init.sql, ... (Postgres) and the SQLite version
├── features.py        task features (5.2): ast-based model family and techniques, vocabularies, failure signature
├── events.py          per-attempt events to the outbox as they happen (8.1)
├── retain.py          run records and events → memories (8.2); lineage (8.5)
├── curator.py         lessons (8.3), with quote verification
├── ranking.py         base quality, use score, retrieval score (9)
├── prompt.py          MemoryPack.as_prompt, memory_request parsing, ReviseRank call
└── scrub.py           secret scrubber (5.3)
```

Run metrics are outside `src/memory/`, because they are recorded also
with no memory backend:

```text
src/orchestration/
└── run_metrics.py     catalog (METRICS), compute, backfill, compare; CLI (11)
```

Changes outside `src/memory/` (each is small):

* `src/orchestration/run_store.py`: the `run_conditions` and `run_metrics`
  tables (11.5); set the current run id at `start_run` (11.6).
* `src/llm.py`, `src/search/web_search.py`, `src/search/page_fetch.py`:
  add `run_id` (and the agent, where known) to each log line.
* `src/llm.py`: optional `memory=` argument; the memory-request loop.
* Each agent: build task features, prefetch at its recall points (7.4),
  pass `memory=`, read `memory_used` from the answer, call `record_usage`.
* `master_agent.py`: an invocation UUID per attempt and an event after
  each attempt and rerun decision in `_supervise` / the rerun loop (8.1,
  8.5); `run_metrics.compute`, `retain_run` and `sync` at run end; memory in
  `_diagnose`; known-not-to-help declines (7.6); a `memory_used` check (the
  ids must be ids that were shown).
* Slack: one line per stage when memory was used ("used 2 past cases:
  ..."); a Master Agent note for each tuning decision.
* Run database: two new tables for run metrics (11.5). Nothing else
  changes. `memory_usage` holds the links to memories.
* Dashboard: a read-only **Compare runs** view (11.9), and a read-only **Memory** view (best and worst per agent,
  search, one memory with its evidence and score reasons). The rating
  button **writes** to the memory store. This is a change to the rule
  "the dashboard writes only to `dashboard_files/`", so it needs the
  team's approval (question 3 in section 17).
* Report Agent (when built): a "Memory used" section listing each memory
  that a stage used, with its run id. A reused script is described as
  adapted from run X, not as new work.

---

## 15. Milestones

The work is divided into 9 milestones. Each milestone has one **key
goal**, gives something that can be used or measured by itself, and has
a "done when" test. A milestone is complete only when its tests pass
(`.venv\Scripts\python.exe -m pytest -q`) and its "done when" items were
actually checked, not assumed (CLAUDE.md, Instructions 11-12).

### 15.1 Overview

| # | Milestone | Key goal | Needs | Account needed |
|---|-----------|----------|-------|----------------|
| M1 | Run metrics | Measure every run, so runs can be compared | none | No |
| M2 | Memory store | A safe, shared place to keep memories | M1 | Yes |
| M3 | Run recording | Never lose what happened in a run, and know which attempts were kept | M2 | No (SQLite is enough) |
| M4 | Memories from runs | Turn run records into ranked memories | M3 | No |
| M5 | First recall (Replication + Master) | The coding agent sees good and bad past scripts (the XGBoost example) | M4 | No |
| M6 | Learning loop | Memory ranks change with the outcomes of their use | M5 | No |
| M7 | All agents | Every agent can use memory at its decision points | M6 | No |
| M8 | Controlled comparison | Measure whether memory helps (memory on vs. off) | M1, M7 | Yes (shared runs) |
| M9 | Dashboard and report | People can see, check and report the memory and the comparison | M8 | No |

M1 does not depend on memory and can start now. M1-M5 give the user's
main example (the coding agent looks at past scripts, and every run is
measured). M6-M9 make memory learn, spread to all agents, and prove its
effect.

```text
M1 ──► M2 ──► M3 ──► M4 ──► M5 ──► M6 ──► M7 ──► M8 ──► M9
 └──────────────────────────────────────────────►┘ (run metrics feed the comparison)
```

### 15.2 M1: Run metrics

**Key goal:** every run, with or without memory, records its conditions
and a fixed set of metrics, and runs can be compared. Runs made before
memory exists become the measured baseline.

Deliverables (status 2026-11-13):
- [x] `src/orchestration/run_metrics.py`: metric catalog (`METRICS`,
      `METRICS_VERSION`), `compute`, `backfill`, `show`, `compare` CLI (11.4-11.7).
- [x] `run_conditions` and `run_metrics` tables in the run database (11.5).
- [x] Current run id set at `start_run` (`src/orchestration/run_context.py`);
      `run_id` in `search_calls.jsonl` and `page_fetch_calls.jsonl`.
- [x] `run_id` in `llm_calls.jsonl` (`llm._record_call`; added with M3),
      and a per-attempt start/end time for exact `agent_seconds` (done by
      Codex: `run_store.stage_timer` and the `stage_timings` table, with
      `METRICS_VERSION` 2; version-1 rows are relabelled on read, never
      rewritten). Runs before these changes keep time-span log matching and
      `timing_method = legacy_completion_gaps`.
- [x] Metrics computed at run end, fail-soft. Changed from the plan: the hook
      is in `run_store.finish_run` (called on success, halt and crash), not
      in `master_agent.py`, so no edit of that file was needed.
- [x] Backfill of the 12 recorded runs (`pre_memory`; time-span log matching;
      a stale `running` run measured as crashed).
- [x] Tests (`tests/test_run_metrics.py`): missing values are `null` with a
      reason, never 0; halted and crashed runs are recorded; a metric failure
      does not stop a run; written once; backfill; compare and its test.

Done when:
- A new run (success, halted and crashed) has all its conditions and
  catalog metrics, each with a value or `null` with a reason.
- `compare` prints the 10 backfilled runs, grouped by topic, with n,
  spread and the list of differing conditions.

### 15.3 M2: Memory store

**Key goal:** a shared store that the team can use, that cannot be
erased by mistake, and that the pipeline does not need in order to run.

Deliverables (status 2026-11-13):
- [ ] Supabase project (or other Postgres) for the team. **Waiting for the
      team** (open question 1): the code is ready; see the setup steps below.
- [x] `memory_agent` role with no `DELETE` (`migrations/postgres/002_roles.sql`);
      the admin connection is used only by `python -m src.memory migrate`
      (`MEMORY_ADMIN_DATABASE_URL`, set in the shell, never in `.env`);
      `MEMORY_BACKEND` and `MEMORY_DATABASE_URL` in `.env.example`.
- [x] Schema and numbered migrations for both backends (`src/memory/schema.py`,
      `src/memory/migrations/{sqlite,postgres}/`). The Postgres tables are in
      their own schema `agent_memory`, not `public`, so Supabase's web API
      cannot reach them.
- [x] `src/memory/` fail-soft API (`put_memory`, `get_memory`,
      `list_memories`, `record_usage`, `increment`, `sync`, `status`);
      SQLite and Postgres backends; local outbox and read cache; `sync()`
      also sends run metrics. Every op is applied exactly once (`applied_ops`).
      `sync()` runs at the start and end of every run when
      `MEMORY_BACKEND=postgres` (`run_store`).
- [x] Secret scrubber before every write (`src/memory/scrub.py`).
- [x] Tests with SQLite only (`tests/test_memory_store.py`; a second SQLite
      file stands in for the hosted store); `tests/conftest.py` forces a temp
      SQLite store. New dependency: `psycopg[binary]==3.3.9`.
- [x] Live check against a throwaway local Postgres 16 (Docker), 2026-11-13:
      migrations applied twice (second time: nothing); `check-role`: reads
      work, `DELETE` refused; also refused: `UPDATE` of a non-counter column,
      `INSERT` into `memory_meta`, `TRUNCATE`; offline writes kept in the
      outbox and sent once when the store was reachable; the 12 measured
      runs uploaded (1,508 metric rows); a resent op applied once. This check
      found one defect, now fixed: the stored schema version was the number
      of migration files, not the code's `SCHEMA_VERSION`.

Setup steps for the hosted store (one person, once):
1. Create the Supabase project (or Neon / any Postgres 14+).
2. In a shell (not in `.env`): set `MEMORY_ADMIN_DATABASE_URL` to the admin
   connection string, and run `python -m src.memory migrate`.
3. In the SQL editor: `ALTER ROLE memory_agent WITH LOGIN PASSWORD '<long random>';`
4. Give each team member the `memory_agent` connection string privately
   (not in Slack or the repository). On Supabase's pooler the user name is
   `memory_agent.<project-ref>`.
5. Each member sets in `.env`: `MEMORY_BACKEND=postgres`,
   `MEMORY_DATABASE_URL=<that string>`, `MEMORY_CONTRIBUTOR=<short name>`,
   then runs `python -m src.memory check-role` and `python -m src.memory sync`.

Done when:
- A `memory_agent` connection can `SELECT 1`, and `DELETE` is refused.
- A manual test writes while offline, then syncs to Postgres, with no
  duplicate rows.
- With `MEMORY_BACKEND=off` the pipeline runs as it does today (plus the
  M1 run metrics).

### 15.4 M3: Run recording

**Key goal:** nothing that happened in a run is lost, also when the run
crashes, and each attempt is known to be kept, superseded, reverted,
rejected or unknown.

Deliverables (status 2026-11-13):
- [x] Invocation UUID for each agent attempt: the one Codex's
      `run_store.stage_timer` already makes in `_supervise` (same id in
      `stage_timings`, the events and the lineage), not a second id.
- [x] `src/memory/events.py`: an event after each attempt (decision, failed
      checks, error), each rerun opened / closed (kept, reverted, undone)
      and each declined request, written to the store and outbox at once;
      a `run_finished` event from `run_store.finish_run`. Schema version 2
      (`memory_events`, `invocation_lineage`; migrations sqlite/002,
      postgres/003). Applied to the team's Neon store on 2026-11-13.
- [x] Lineage at run end (`events.lineage_for`, rules in 8.5), written once
      per run to `invocation_lineage`.
- [x] Recovery: a run with events but no lineage (process killed) is
      resolved as crashed when the next run starts (`recover_unfinished`).
      (`retain_run` itself is M4.)
- [x] Tests (`tests/test_run_events.py`): the lineage rules; real
      `_supervise` attempts and the real rerun loop (reverted, kept,
      declined); a killed run resolved at the next start; written once;
      events sent to the hosted store; fail-soft with memory off; the LLM
      log's run id.

Done when:
- A run killed in the middle still has the events of all its finished
  attempts.
- For a run with a reverted rerun, each attempt has the correct lineage.

### 15.5 M4: Memories from runs

**Key goal:** each run (new or old) becomes a set of memories (good
solutions, failures, decisions) with a quality score that can be
explained.

Deliverables:
- [ ] `src/memory/features.py`: task features with fixed vocabularies,
      model family and techniques from the script (`ast`), failure
      signature (5.2).
- [ ] `src/memory/retain.py`: the record → memory rules (8.2), duplicates,
      lineage rule (only `kept` gives a positive solution).
- [ ] Memory states (8.4) and the `approve` / `quarantine` CLI.
- [ ] `src/memory/ranking.py`: base quality with the validity and gap
      rules (9.1).
- [ ] `retain_run` in Master Agent's `finally` block, after `run_metrics.compute`.
- [ ] Backfill of the 10 runs, with trust labels and a report (10).

Done when:
- The backfill report lists the memories of each run with their score
  reasons.
- No reverted or `unknown`-lineage attempt is a positive solution.
- An Evaluation Agent memory that found real leakage is not ranked down
  because the gap got larger.
- A person checks the top 10 and bottom 10 memories per agent and agrees
  with them (gate before M5).

### 15.6 M5: First recall (Replication + Master)

**Key goal:** the coding agent gets good and bad past scripts for the
same model family before it writes, repairs or revises a script, and can
ask for more memory itself. Master Agent's diagnosis gets past diagnoses
of the same failure.

Deliverables:
- [ ] Retrieval test set of at least 30 labelled queries, made **before**
      any ranking tuning, and `python -m src.memory benchmark` (9.6).
- [ ] Recall with relevance, quality and freshness; current run
      excluded; snapshot applied; only `active` memories (9.2).
- [ ] `MemoryPack.as_prompt()` with the prompt rules (7.3).
- [ ] `llm.chat(..., memory=)` and the `memory_request` loop, at most 2
      per call (7.2).
- [ ] Prefetch in the Replication Agent (generation, repair, revision)
      and in `_diagnose` (7.4).
- [ ] `memory_used` in answers; Master Agent checks the ids were shown;
      `record_usage` with `supplied_digest`; `memory_exposure` (6.3, 11.3).

Done when:
- The retrieval test passes its gate (9.6).
- A live run shows good and bad past scripts in the Replication Agent's
  generation prompt, and `memory_usage` rows are written.
- A test shows that the model's `memory_request` is answered and that a
  third request is refused.
- The current run's own memories are never recalled.

### 15.7 M6: Learning loop

**Key goal:** memories that are followed by good outcomes rise, and
memories followed by bad outcomes fall, across runs. Good past solutions
can be reused safely.

Deliverables:
- [ ] Use score and `quality_effective` at run end (9.3).
- [ ] ReviseRank on retries and reruns, with the permutation check (9.4).
- [ ] Curator lessons, quote-checked, starting as `candidate` (8.3, 8.4).
- [ ] Fast mode for the Replication Agent, only from `kept` solutions,
      with all checks still run (7.5).
- [ ] Contradiction linking (8.4).

Done when:
- After two runs on the same topic, the memory followed by a good
  outcome has a higher `quality_effective` than the one followed by a bad
  outcome.
- A curator lesson stays `candidate` until a person approves it or 2
  independent runs support it.
- A fast-mode script that fails goes through the normal repair loop, and
  its memory's rank goes down.

### 15.8 M7: All agents

**Key goal:** every agent uses memory at its own decision points, not
only the coding agent.

Deliverables:
- [ ] Recall points for Intake, Research, Validation, Data, Cleaning,
      Code Discovery, Evaluation and Experiment (7.4). Deterministic
      agents get structured hints only; no LLM call is added only to use
      memory.
- [ ] Slack line per stage when memory was used, and a Master Agent note
      for each tuning action.
- [ ] Hooks for the Report and Presentation Agents when they are built.

Done when:
- Each agent's prompt has a memory pack when matching memories exist.
- The retrieval test set has queries for each agent and still passes.
- A run with `MEMORY_MODE=off` gives the same behavior as before memory.

### 15.9 M8: Controlled comparison

**Key goal:** a fair measurement of whether memory helps, which can be
an experiment in the final report.

Deliverables:
- [ ] Written test plan before the first run
      (`reports/comparisons/<group>.json`): topics, conditions, repeats,
      primary metrics, exclusion rules (11.8).
- [ ] `MEMORY_SNAPSHOT` and `MEMORY_COMPARISON_GROUP`; frozen snapshot;
      held-out topics; test runs do not feed each other (11.8, 13).
- [ ] At least 3 runs per topic and condition (`off`, `read`, optional
      no-warnings), in alternating order, with the same commit and provider.
- [ ] `compare` output grouped by assigned condition, with actual
      exposure, n, spread, test (n ≥ 3), and confounders (11.7).
- [ ] Optional: pipeline tuning (`MEMORY_TUNING`, 7.6), tested as its own
      condition; embeddings only if the retrieval test shows missed
      paraphrases (9.5).

Done when:
- The comparison table exists for all planned runs, including failed
  ones, with no run moved between groups after its result.
- The result is stated honestly: improvement, no difference, or worse,
  with its limits.

### 15.10 M9: Dashboard and report

**Key goal:** people can see what memory did and how runs compare, and
the final report uses only recorded values.

Deliverables:
- [ ] Dashboard **Compare runs** view (read-only, 11.9).
- [ ] Dashboard **Memory** view: best and worst per agent, search, one
      memory with its evidence, state and score reasons (14).
- [ ] Rating button, only if the team approves it (17, question 3).
- [ ] Report Agent: "Memory used" section (reused scripts named as
      adapted from run X) and the M8 comparison, read only from
      `run_metrics` (11.8, 14).

Done when:
- The dashboard and the `compare` CLI show the same values for the same
  runs.
- Each number in the report's memory and comparison sections can be
  traced to a `run_metrics` row.

---

## 16. Risks

| Risk | Mitigation |
|------|------------|
| Agents copy a past metric or link without verification | Prompt rule 7.3; every existing check still runs; Master Agent checks `memory_used`; the report names reused memories. |
| A wrong memory spreads to all team members | Use score lowers it after bad outcomes; quarantine (8.4) stops it at once, also in the local cache; curator lessons start as candidates; trust labels; tuning is off by default and bounded. |
| An approved attempt from a reverted rerun is reused as a good solution | Lineage (8.5): only `kept` attempts become positive solutions. |
| Ranking by paper gap punishes correct validity work | Evaluation is not ranked by gap; validity fixes count as success (9.1). |
| A memory run that got no memory is counted as a memory run | Assigned condition and actual exposure are recorded separately (11.3, 11.7). |
| Memory runs in a test see memories from earlier test runs | Frozen snapshot and held-out topics (11.8). |
| "Used" is read as proof that memory helped | Labelled as a self-report; causal claims only from the controlled test (9.3). |
| Weak free models ignore the memory protocol | Prefetch is deterministic and does not depend on the model. The request loop is limited to 2. |
| Prompts get too long | Pack size budget; summaries first; content only for top solutions. |
| Hosted project pauses or is offline | Local cache and outbox; fail-soft; the run continues. |
| Secrets in memory content (public team store) | Scrubber before each write; `memory_agent` role cannot read other schemas. |
| A dataset's license prevents sharing | No data rows in memory; only traits and links. |
| Two runs at the same time | Append-only writes, client UUIDs, counters updated with `UPDATE ... SET n = n + 1`. |
| Results are less reproducible | Each run records the memory ids it used (`memory_usage`), so the exact context can be rebuilt. `MEMORY_MODE=off` gives the old behavior. |
| A comparison of runs gives a false conclusion | Same topic and Parent Paper by default; n and spread always shown; no test with fewer than 3 runs per group; confounders listed (11.7). |
| Few runs per group (FreeLLMAPI noise is large) | At least 3 runs per group; alternate order; report "too few runs" instead of a conclusion. |
| A metric changes meaning between code versions | `metrics_version`; old values are kept, never rewritten (11.1). |
| Coordination with Codex's parallel work | `src/memory/` is new and does not conflict. The small changes in `llm.py`, `master_agent.py` and the agents must be merged with care. Do them after Codex's current changes in those files are committed. |

---

## 17. Open questions

1. **Account owner:** who makes the Supabase project, and how is the
   `memory_agent` connection string given to team members (not in Slack
   or in the repository)?
2. **Contributor names:** what short name does each team member use?
3. **Dashboard rating button:** may the dashboard write ratings to the
   memory store? (Today the dashboard writes only to `dashboard_files/`.)
4. **Seed memories:** do we use seed lessons or DS-Agent's case bank (12),
   or only our own runs?
5. **Report:** must the memory on/off comparison (11) be an experiment in
   the final report?
6. **Tuning:** may the past-reproducibility term (7.6) change the Parent
   Paper ranking, or must it stay off for the course submission?
7. **Test topics:** which topics, and how many runs per group, does the
   team use for the memory on/off comparison (11.8)? Each run uses LLM and
   search quota (Brave: 1,000 requests per month).
8. **Shared metrics:** may run metrics of every team member go to the
   shared store, also runs with memory off? (They hold no content, only
   numbers, names and ids.)
