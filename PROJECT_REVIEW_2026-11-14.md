# Project review — 2026-11-14

Review only. No application code, test code, configuration, or production databases were intentionally changed. This document is the review deliverable.

## Executive assessment

The largest risks are in the integrity of the memory bank and the evidence used to score it. A run can appear successful while retaining incomplete memory, discarding contrary evidence, or assigning positive quality to an invalid baseline. The controlled comparison also does not currently freeze the ranking it is meant to evaluate.

The architecture has useful safeguards: separate recall and retention, explicit lineage, deterministic scoring with reasons, bounded model-request counts, operation IDs for synchronization, secret scrubbing, and isolated test fixtures. Those safeguards are valuable, but several boundary cases undermine them.

Severity: **P1** = high priority; can corrupt research conclusions, trust, or isolation. **P2** = important correctness or reliability problem. Findings below distinguish executed reproductions from conclusions established by tracing code. These are defects and concrete failure scenarios, not claims that every existing stored run is affected.

## Scope and method

Reviewed the first-party Python project, memory migrations, tests, dashboard JavaScript, configuration, and project documentation. Traced agent supervision, retries, reruns, artifacts, data cleaning, evaluation, reports, retrieval, retention, learning, and hosted synchronization. Parsed all 75 Python source files and 41 Python test files with `ast.parse`; checked dashboard JavaScript with `node --check`.

Used existing tests and small isolated checks. No live pipeline, hosted database writes, paid LLM calls, Slack messages, or dependency installations were requested. Database reproductions used a disposable SQLite file, not `database/memory_local.db`. One reproduction completed its assertions/output but failed during temporary-directory cleanup because SQLite connections remained open; that limitation is documented in finding 12. External services and the actual historical memory bank were not audited live.

This is a broad source review, not a proof that every possible defect has been found. Test-suite results are recorded at the end.

## How memory currently moves through the system

1. `run_store.start_run()` records the run conditions and a memory creation-time cutoff.
2. An agent constructs `recall.for_agent()` or `context_for()` from its current inputs. Recall fetches active memories for specified agent names, then filters and ranks them.
3. Ranking uses feature relevance × `(0.5 + 0.5 × effective_quality)` × code/schema freshness. It returns up to three positive cases and two negative cases. Neutral cases are excluded.
4. `llm.chat()` appends the memory pack to the task prompt. The model can request additional packs, and self-reports which labels it used.
5. `MemoryContext.finish()` stores usage rows and increments recall/use counters. Agent decisions and rerun boundaries are recorded as events.
6. At run completion, lineage is resolved; `retain_run()` builds memories; the curator may propose or activate lessons; `learn_run()` updates helped/hurt counters.
7. SQLite stores everything locally. With Postgres configured, writes also enter an outbox; reads normally come from Postgres and replace local cached memory rows. End-of-run synchronization sends queued operations and refreshes known IDs.

The local caches for papers, datasets, repositories, and column mappings are additional forms of cross-run memory. They affect behavior even when shared agent memory is turned off.

## Memory findings

### 1. P1 — The supposedly frozen comparison bank keeps learning

**Locations:** `src/memory/recall.py:409`, `src/memory/learning.py:82`, `src/orchestration/run_store.py:375`, `src/orchestration/comparison.py`.

`MEMORY_MODE=read` blocks retention but does not block usage counter updates or `learn_run()` updating `times_helped`/`times_hurt`. Those counters feed effective quality and therefore the next run's ranking. The snapshot filters creation dates; it does not restore historical counter values, quality, or state.

**Evidence:** With mode patched to `read`, finishing a recalled case wrote one usage row and incremented `times_used`. A base-quality 0.9 memory scores 0.7583 after one negative outcome. Both outcomes are intended behavior in normal learning mode, but contradict a frozen experimental bank.

**Impact:** Later comparison runs receive a different ranking influenced by earlier test outcomes. The comparison cannot cleanly measure a fixed memory intervention.

**Proposed solution:** Record usage separately from mutable bank learning. Suppress bank-quality changes in frozen mode, and materialize a versioned bank including counters, states, and quality for each comparison. Add a sequential comparison test asserting identical retrieval scores before and after completed test runs.

### 2. P1 — Hosted reads overwrite unsynchronized local state and counters

**Locations:** `src/memory/__init__.py:104`, `src/memory/backends/sqlite.py:214`.

Postgres writes update the local row and queue an operation. A subsequent successful hosted read calls `cache_memories()`, which uses `INSERT OR REPLACE` with the remote row. That remote row may predate pending local increments, quarantine changes, or quality changes. Hosted fetch results also omit local-only inserted memories until synchronization.

**Evidence:** In an isolated store, a local counter changed from 0 to 1; caching the earlier remote-shaped row immediately reset it to 0. The same replacement mechanism applies to `state` and `quality`.

**Impact:** The local mirror no longer represents the user's pending changes. A pending quarantine can be overwritten locally by an active remote row; recall can still supply it before sync. Pending operations remain in the outbox, so this is a consistency and immediate-read problem rather than proof that every queued operation is permanently lost.

**Proposed solution:** Keep remote base rows separate from pending local operations and overlay pending changes on reads. Alternatively synchronize relevant pending changes before serving them, with explicit offline behavior. Test local quarantine/increment → hosted read → offline read → sync.

### 3. P1 — Offline fallback can resurrect a remotely quarantined memory

**Locations:** `src/memory/__init__.py:104`, `src/memory/__init__.py:146`, `src/memory/recall.py:447`.

An active-only hosted query returns no row for a remotely quarantined memory. `cache_memories([])` does not invalidate its older active local copy. If the remote then becomes unavailable, the still-eligible local cache can return that memory as active. The two-minute candidate cache adds a separate stale-state window. A full successful sync of known IDs helps, but normal reads do not reconcile removed members of an active result set.

**Impact:** Quarantine is not reliably enforced during fallback. The global seven-day cache age permits a substantial stale-state window.

**Proposed solution:** Reconcile membership for authoritative queries or pull state-change tombstones/version watermarks. Define a shorter safety policy for stale quarantine information. Test a remote active → quarantined transition followed by an outage, without a full sync in between.

### 4. P1 — Content-only deduplication discards different outcomes and lineage

**Location:** `src/memory/retain.py:749`.

Deduplication uses only `(agent, content_sha)`, both against earlier runs and within one run. It ignores kind, polarity, lineage, task, quality, evaluation, and evidence. On a duplicate, the old memory gets `seen_count += 1`; the new evidence is discarded.

**Concrete scenario:** The same script first appears as a neutral unevaluated/superseded case and later succeeds as a kept solution. The kept case can disappear behind the older neutral case. Conversely, the same script can later be identified as leaky, but its new warning can be discarded behind an earlier positive case. Within a run, an earlier superseded copy can consume the hash before the final kept copy is considered.

**Impact:** The bank can fail to learn a success or retain stale positive advice despite contrary evidence. Learning also derives warning invocation IDs from retained rows, so discarded warnings affect credit assignment.

**Proposed solution:** Deduplicate artifact content, not observations. Preserve an observation per invocation/run with independent outcome and lineage; link observations to shared content. Aggregate trust only after considering all observations. Add same-content/different-outcome tests in both ordering directions.

### 5. P1 — Failed memory writes can still mark a run fully retained

**Locations:** `src/memory/retain.py:716`, `src/memory/retain.py:763`, `src/memory/retain.py:826`.

`put_memory()` fails soft and returns `None`. Retention does not check that result: it appends the row to `written`, then inserts the permanent `memory_runs` completion marker. Later retention skips the run as already retained. Rescoring similarly can supersede an old memory even when inserting its replacement failed.

**Impact:** Transient storage failures produce permanently incomplete memory, misleading success reports, and potentially a superseded case with no replacement. The curator can also receive rows reported as written that were not persisted.

**Proposed solution:** Make retention transactional locally, or persist per-memory completion and insert the run marker only after all required operations succeed. Check write results before superseding or curating. Use deterministic IDs for safe retries. Inject a failed `put_memory()` and verify a retry repairs the run.

### 6. P2 — Approved Research Agent outputs are never retained

**Locations:** `src/agents/research_agent.py:223`, `src/memory/retain.py:393`.

The Research Agent returns a list. `_approved_memory()` immediately returns `None` unless the output is a dictionary. The list-handling branch in `_describe()` is therefore unreachable for the normal successful Research Agent output.

**Evidence:** Passing a valid approved list output to `_approved_memory()` returned `None`.

**Impact:** Research Agent recall exists, but ordinary successful research runs do not populate its solution bank.

**Proposed solution:** Accept and normalize the documented list output before the dictionary guard. Test retention from a real Research Agent-shaped output rather than a dictionary stand-in.

### 7. P1 — Recall and fast reuse lack important current-task features

**Locations:** `src/agents/replication_agent.py:530`, `src/memory/features.py:174`, `src/orchestration/run_metrics.py:136`, `src/memory/recall.py:502`.

Replication constructs its memory task with parent paper and cleaning data but no intake, dataset identity, topic, or current script. Consequently task type is absent even though fast reuse claims to match it. Other recall sites also omit available run context. The methodology fallback calls a Python identifier detector on natural-language prose: ordinary names such as `XGBoost`, `random forest`, and `logistic regression` are not recognized.

**Evidence:** All three prose names returned no model families; `from xgboost import XGBClassifier` worked. A normal replication memory query built from a parent title/methodology and row count contained no task type, topic, or model family for `XGBoost`.

**Impact:** Useful cases are missed; weak matches dominate; fast mode often does not activate, and when it does its task-type check is bypassed if the current type is unknown. Stored memories and live queries describe the same task differently.

**Proposed solution:** Pass an explicit immutable task descriptor through agent decisions. Separate prose model-name normalization from AST analysis. Require known compatibility fields for fast reuse, or explicitly mark a reuse as uncertain. Add end-to-end query-feature assertions at generation, repair, review, and validation call sites.

### 8. P1 — Retention overlooks verified analyst leakage findings

**Locations:** `src/memory/retain.py:269`, `src/memory/retain.py:393`, `src/memory/learning.py:64`.

`_leakage()` reads only `review.static_findings`. The Evaluation Agent also produces verified `review.findings` with category `data_leakage`. Those analyst findings are not included in baseline validity or the special conversion of a kept leaky script into a warning.

**Evidence:** An evaluation containing a verified critical analyst leakage finding and no static findings produced `_leakage(...) == []`.

**Impact:** A baseline with a known validity defect can remain a positive solution, receive gap credit, and positively reinforce the memories it used. Static heuristics do not catch every leakage pattern, so the second evidence channel matters.

**Proposed solution:** Centralize validity assessment using both evidence channels, with category and severity requirements. Reuse it in retention, learning, rerun acceptance, and reports. Test an analyst-only leakage finding.

### 9. P2 — Ranking silently ignores everything older than 1,000 active cases per agent

**Locations:** `src/memory/recall.py:447`, `src/memory/backends/sqlite.py:230`, `src/memory/backends/postgres.py` (`fetch_memories`).

Recall fetches at most 1,000 rows, ordered newest first, before applying relevance, current-run exclusion, snapshot, or kind filtering. This is not a ranked top-1,000 search.

**Impact:** A highly relevant older solution or critical warning becomes unreachable as the bank grows. In a frozen comparison, more than 1,000 newer post-snapshot rows can crowd out every eligible historical row before the snapshot filter runs.

**Proposed solution:** Apply eligibility filters in the backend and use paginated or indexed candidate retrieval. Measure candidate truncation explicitly. Test with a relevant case just beyond row 1,000 and with a large post-snapshot bank.

### 10. P2 — Memory prompt budgets are not enforced

**Locations:** `src/memory/recall.py:345`, `src/llm.py:83`.

When full content does not fit, `block()` falls back to a summary block but appends it regardless of remaining space. The rules themselves can exceed a small configured budget. Separator overhead is also unaccounted for. Additional model requests append whole packs to the existing prompt, without an aggregate memory budget.

**Evidence:** A configured 300-character pack returned 2,305 characters in the isolated check.

**Impact:** Unexpected context growth, higher cost, provider rejection, or truncation of task evidence. A fast-mode script may be selected although only its summary fits in the supplied pack.

**Proposed solution:** Enforce a cumulative token/character budget, reserving overhead before selection. Trim or omit cases with explicit accounting, and choose fast reuse only from content actually delivered. Test the final complete prompt, including extra requests.

### 11. P2 — Rescoring retires curator lessons that it does not rebuild

**Location:** `src/memory/retain.py:781`.

`rescore_run()` loads all memories belonging to the run, including curator lessons. It rebuilds only deterministic stage/rerun memories. Its final unmatched-memory loop retires any active/candidate row not matched, so lessons are retired even though their rule is outside the rescorer's scope.

**Impact:** Running the advertised rescore maintenance command can silently remove learned lessons from recall or confirmation.

**Proposed solution:** Restrict rescoring to memories owned by the deterministic retention rules, using explicit provenance. Leave curator and manual records untouched unless a separate review policy invalidates them. Test a run containing an active lesson and a candidate lesson.

### 12. P2 — SQLite connections are not deterministically closed

**Locations:** `src/memory/backends/sqlite.py:67`, `src/orchestration/run_store.py:183`, callers using `with ..._connect() as conn`.

The helpers return raw SQLite connections. A SQLite connection context manager commits or rolls back; it does not close the connection. Most callers rely on eventual garbage collection. The migration path explicitly closes one connection, but ordinary reads/writes do not.

**Evidence:** The isolated database checks finished, then `TemporaryDirectory` cleanup failed with Windows error 32 because `m.db` was still open. This was a cleanup failure after reproduction output, not a failed behavioral assertion.

**Impact:** Nondeterministic handle retention, Windows cleanup/rename failures, unnecessary resource usage, and harder database lifecycle management during sustained runs and tests.

**Proposed solution:** Use a context manager that explicitly closes connections, separately preserving transaction semantics. Verify immediate file release without forcing garbage collection.

### 13. P2 — Learning measures local approval more than research validity

**Locations:** `src/memory/learning.py:59`, `src/analysis.py` (`effective_quality`).

For ordinary agents, an approved, kept attempt without a retained negative memory is enough to call a used memory helpful. A kept replication with a poor measurable gap can therefore positively reinforce every case it self-reported using. Multiple LLM calls within the same invocation can each credit the same memory from one outcome. This becomes especially misleading when warnings were lost through findings 4 or 8.

**Impact:** Repetition and permissive stage checks can amplify weak advice. The score should not be interpreted as demonstrated causal usefulness or improved replication accuracy.

**Proposed solution:** Define and expose separate signals for execution success, validity, fidelity, and metric improvement. Aggregate one bounded outcome per memory/invocation or run. Keep causal claims out of labels unless supported by an experiment. Test a passing but invalid or substantially worse replication and repeated uses within one invocation.

### 14. P2 — Reported learning counts can claim success after failed writes

**Location:** `src/memory/learning.py:106`.

`record_outcome()` returns `False` on failure, but `learn_run()` increments its helped/hurt report regardless of the return value.

**Impact:** Operational reports can say the bank learned when no update was recorded, making fail-soft behavior difficult to detect.

**Proposed solution:** Count successful writes separately from attempted classifications and return explicit failed-update IDs. Test a storage failure during outcome recording.

## Pipeline findings that affect memory quality

### 15. P1 — Provenance prevents deduplication across datasets

**Locations:** `src/agents/cleaning_agent.py:116`, `src/agents/cleaning_agent.py:357`.

`_clean_dataframe()` adds `_source_dataset`. The merged frame then calls `drop_duplicates()` across every column, including that provenance value. Identical observations from different sources remain distinct.

**Evidence:** Two identical feature/label rows tagged with different sources remained two rows after the exact production deduplication sequence.

**Impact:** Duplicate examples can land in different train/test splits and inflate evaluation. Dataset counts and positive cleaning/replication memory can then be based on leaked data.

**Proposed solution:** Deduplicate on canonical feature/label identity while preserving provenance in a separate list/table. Check row identity across splits independently of source metadata. Test duplicated dataset mirrors with different links.

### 16. P1 — Target handling can approve data without a usable target; multiclass/regression are unsupported

**Locations:** `src/orchestration/dataset_budget.py:39`, `src/orchestration/dataset_budget.py:71`, `src/agents/cleaning_agent.py:245`, `src/agents/cleaning_agent.py:318`, `src/agents/master_agent.py` (`_check_cleaning`).

Label detection requires exactly two distinct values and a target substring in the column name. Target-fit can pass on dataset-description text even without an identified label. If no loaded dataset has a matching label, cleaning falls back to name-based merging. Master checks only that at least one dataset merged, so this can proceed as approved without a label. If a binary-named target has two non-0/1 numeric values, the binary mapping path rejects it; regression and multiclass targets have no corresponding handling.

**Impact:** A generic `Class` label can be missed even for a valid fraud dataset; a dataset can be approved without a usable target; nonbinary tasks accepted by intake cannot be faithfully supported downstream. Retention records some missing-label cases as warnings, but that does not prevent the run from generating or evaluating the wrong task.

**Proposed solution:** Carry explicit task type and target-column evidence from data verification into cleaning. Support appropriate multiclass/regression label schemas, or clearly reject unsupported tasks at intake. Treat unknown target identity as an explicit unresolved state before training.

### 17. P2 — Column normalization can still produce duplicate names

**Location:** `src/agents/cleaning_agent.py:97`.

The collision suffix is checked only against the original normalized base, not all names already emitted or yet to be emitted.

**Evidence:** Input columns `A`, `a`, `a_2` normalized to `a`, `a_2`, `a_2`.

**Impact:** Selection by name can return a DataFrame instead of a Series; mappings become ambiguous; CSV readers may rename columns again. This breaks the promise that normalization resolves collisions.

**Proposed solution:** Reserve all normalized names and generate globally unique output names. Assert uniqueness before mapping and writing data.

### 18. P1 — Metric comparison loses units and accepts unverified published values

**Locations:** `src/agents/evaluation_agent.py:98`, `src/agents/evaluation_agent.py:126`, `src/agents/evaluation_agent.py:157`, `src/agents/evaluation_agent.py:592`.

The metric-line parser accepts `%` but discards that unit. All numeric values with absolute value at most one are then multiplied by 100. The generic extractor also accepts arbitrary numeric metrics without distinguishing bounded scores from RMSE, MAE, losses, or time. When official text is unavailable, search-snippet values still produce ordinary `difference_points` and classifications; a warning string is added, but downstream gap calculation and memory quality still use those numbers.

**Evidence:** `METRIC recall=0.5%` parsed as 0.5 and normalized to 50 percentage points. Natural metric units and explicit percentages are not preserved.

**Impact:** False matches, false gaps, and quality/rollback decisions based on another paper's snippet or the wrong scale. Regression metrics can be distorted by 100×.

**Proposed solution:** Store metric identity, units, scale, averaging, dataset/split, and source evidence together. Use metric-specific comparison rules. Exclude unverified published values from trusted gap/quality calculations while still displaying them as provisional. Test explicit percentages below 1%, fraction scores, and regression errors.

### 19. P1 — Experiment reruns can keep a newly leaky baseline because its gap improved

**Location:** `src/agents/master_agent.py:1576`.

`_close_rerun()` decides whether an Experiment Agent rerun is kept by whether the mean gap widened. It does not veto new verified leakage or require the same matched metric set. Evaluation can request a later fix, but the invalid candidate has already become the kept baseline and may remain if that fix fails or the loop ends. Dropping a difficult matched metric can also reduce the mean without improving the remaining task comparability.

**Impact:** The advertised closest valid baseline invariant is not enforced at the acceptance boundary. Its lineage and retained memory can describe a numerically closer but invalid result.

**Proposed solution:** Require validity and comparable metric coverage before judging gap improvement. Preserve invalid candidates as evidence without replacing the last valid baseline. Test an improved-gap rerun with new leakage and one that stops reporting a metric.

### 20. P1 — Rerun rollback restores only a subset of the baseline

**Locations:** `src/agents/master_agent.py:1199`, `src/agents/master_agent.py:1207`, `src/agents/replication_agent.py:331`.

Backups include CSVs, stdout/stderr, reconstructed `train.py`, reconstruction notes, and an environment document. They do not preserve the complete cloned repository, its commit, the virtual environment/dependencies, generated model artifacts, or all configuration files. A Code Discovery rerun can replace a cloned repository, while rollback restores old result metadata. Restore errors are printed and swallowed while the caller continues with the old result.

**Impact:** The restored baseline can point to files/environment from the rejected rerun. Historical metrics and current executable artifacts diverge. A report saying the previous baseline was restored can be false after a copy failure.

**Proposed solution:** Use immutable per-run/per-rerun working directories and atomically select the accepted baseline by manifest. Validate restored hashes and fail visibly if restoration is incomplete. Test rejected repository changes and simulated restore failure, not only reconstructed-script changes.

### 21. P1 — An unavailable evaluation review falls back to acceptance

**Locations:** `src/agents/evaluation_agent.py:412`, `src/agents/evaluation_agent.py:487`.

If the review call raises or returns malformed JSON, `raw` becomes empty. `_decide()` defaults to `accept`, subject to the static checks that happened to fire. The review error is put in notes, but there is no required incomplete-review status or retry at this boundary.

**Impact:** A provider outage or parsing failure looks like a valid decision to accept a baseline that never received the promised analyst review. That approval can contribute positive memory quality.

**Proposed solution:** Return a separate `REVIEW_UNAVAILABLE` state and route it through retry or explicit incomplete evaluation. Keep deterministic metric comparison available, but do not equate unavailable review with verified validity.

### 22. P1 — Executed generated/repository code inherits project credentials

**Location:** `src/agents/replication_agent.py:157`.

Every subprocess gets a copy of `os.environ`, plus data paths. Configuration loads `.env` into that environment. Training code, installation hooks, and repository entry points therefore inherit the application's LLM, Slack, Kaggle, and database credentials when configured. A virtual environment isolates packages, not filesystem/network access or secrets.

**Impact:** Untrusted code can read credentials, modify memory or project artifacts, or make external requests with the user's accounts. Memory scrubbing happens after execution and cannot protect this boundary.

**Proposed solution:** Run replication and builds in an OS/container boundary with an allowlisted environment, no mounted secrets, restricted filesystem access, and deliberate network access. Preserve only required data/runtime variables. Validate the boundary using a harmless dummy secret, without exposing a real credential.

### 23. P2 — Download ceilings are checked after the response is already loaded

**Location:** `src/agents/data_agent.py:665`.

The generic download path calls `page_fetch.get()` without `stream=True`, then checks `_too_large(resp)`. Requests has already buffered the response by that point. JSON and Parquet conversion can also load whole datasets before row sampling.

**Impact:** Large or misleadingly declared downloads can exhaust memory before the advertised ceiling or row budget takes effect. A row cap is not a byte/memory cap.

**Proposed solution:** Stream downloads, enforce a cumulative byte limit while reading, close responses deterministically, and use bounded readers for large formats. Test missing/false Content-Length and a body exceeding the limit.

### 24. P1 — Memory-off comparisons still share mutable discovery and mapping caches

**Locations:** `src/orchestration/comparison.py`, `src/orchestration/dataset_cache.py`, `src/orchestration/code_discovery_cache.py`, `src/orchestration/column_mapping.py:96`, `src/agents/data_agent.py:817`.

Comparison conditions toggle shared agent memory but retain the same mutable paper/dataset/repository/mapping caches and output directories. Dataset cache hits return earlier verification decisions immediately; column mapping proposals are cached by dataset links, not schema/data version, target task, or supervisor correction. A later condition can reuse decisions or avoid searches/LLM calls performed by an earlier one.

**Impact:** Timing, call counts, reliability, and even selected inputs depend on run order. A retry can replay a cached bad mapping despite a new diagnosis. This is a second contamination channel independent of finding 1.

**Proposed solution:** Freeze and clone all relevant caches per assigned comparison run, or disable them consistently for the experiment. Namespace normal caches by task/data/schema versions, revalidate stale hits, and permit diagnosed decisions to be invalidated. Test the same comparison order reversed from identical initial caches.

## Additional limitations to address explicitly

- **Quote existence is not semantic verification.** Evaluation's `_verify()` marks any finding with a matching quote of at least four characters verified. An isolated check accepted the unsupported claim “All labels are leaked into the test set” with evidence `import pandas`. Curator quote checks have the same entailment limitation, though they require longer quotes. Preserve the distinction between “quote found” and “claim validated”; require structured evidence checks for critical decisions and lesson activation.
- **The final split is unstratified despite knowing the label.** `cleaning_agent._split_train_test_validation()` shuffles and slices. Rare positives can disappear from a split, causing undefined metrics or misleading results. Its documentation says the target is unknown, which no longer describes the label-aware path. Use a task-appropriate split and record group/time constraints where needed.
- **Curator support is not an idempotent observation.** `curator.py:163` increments counts for each support item without storing a unique supporting-run/lesson relationship. Repeated calls or duplicate support entries can inflate evidence. Contradiction writes likewise lack an immutable unique evidence relationship. Persist supporting and contradicting observations with uniqueness constraints before aggregating scores.
- **Frozen mode is not a historical state snapshot.** Even after stopping learning, a creation cutoff alone cannot preserve lesson activation, quarantine, rescoring, or externally changed quality. A comparison should identify an immutable bank version, not just a timestamp.
- **Operational fail-soft behavior needs structured health signals.** Many memory failures are console warnings while the pipeline advances. Return per-run memory health and incomplete-write counts in the run database/dashboard. Avoid interpreting an empty bank and an unavailable bank as the same experimental exposure.

## Recommended implementation order

1. Protect research integrity: fix cross-source duplicates, metric provenance/units, analyst-leakage validity, review-unavailable handling, and rerun validity acceptance.
2. Protect the memory bank: fix observation-level deduplication, transactional retention/replacement, and remote/local pending-write reconciliation.
3. Make experiments valid: snapshot the complete bank and other caches; stop rank-changing learning in frozen mode.
4. Improve retrieval: consistent current-task descriptors, prose model normalization, backend eligibility filters/pagination, and cumulative prompt budgets.
5. Improve maintenance and operation: deterministic connection closure, scoped rescoring, reliable learning counts, idempotent curator evidence, and visible memory health.
6. Isolate execution and enforce streaming byte limits before using the pipeline with unfamiliar repositories or large datasets.

Before bulk rescoring or backfilling existing memory, take a database backup and use dry-run reports. Fix deduplication and replacement failure handling first; otherwise maintenance can discard evidence or retire useful cases.

## Validation record

- Python syntax: **75 source files and 41 test files parsed successfully**.
- JavaScript syntax: **`node --check src/dashboard/static/app.js` passed**.
- Isolated reproductions: confirmed budget overflow, read-mode usage writes, cache counter overwrite, missing Research Agent retention, missing prose model families/current task features, duplicate normalized column names, surviving cross-source duplicates, lost percent units, and ignored analyst-only leakage.
- Temporary database reproduction: outputs completed; Windows cleanup failed because a database handle remained open. No production database was used.
- Full existing test suite: **861 passed in 432.91 seconds** using `.venv/Scripts/python.exe -m pytest -q`.

The existing suite passes despite these findings: many concern interactions between individually tested layers, real agent return shapes, growing banks, stale remote state, and failure injection. Git status after verification showed no tracked code changes; the only new review deliverable is this Markdown file. Previously existing untracked files were left untouched.
