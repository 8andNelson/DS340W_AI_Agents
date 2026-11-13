# Shared Agent Memory: Backend Development Plan

Date: 2026-11-09

Status: design proposal; no backend implementation is included in this document.

## 1. Goal and agreed scope

Give every pipeline agent access to useful experience from earlier runs, including successful approaches, unsuccessful approaches, verified corrections, and conditions under which an approach did or did not work. Agents should adapt that experience to the current task and verify it again. Learning here means improving decisions through retrieved experience; it does not mean changing model weights.

Store every run and its available metrics, whether memory is enabled or disabled and whether the run succeeds, fails, is interrupted, or remains inconclusive. The backend must support comparing runs, especially runs supplied with cross-run memory against runs without it. Run measurement is a first-class capability independent of lesson publication: a run with no publishable lesson still belongs in the comparison record.

User decisions made while preparing this plan:

- Shared memory must work across machines from the first release.
- Memory may include summaries and controlled access to verified script artifacts.
- Raw dataset rows are excluded from shared memory by default.
- This task authorizes a planning document only. Provisioning, code changes, database migrations, uploads, and deployment remain future work.

Recommended first release: one shared HTTPS memory API, managed PostgreSQL, private artifact storage, and a durable local upload queue on each pipeline machine. All agents use the same memory service through the orchestrator, with role-specific retrieval. PostgreSQL holds searchable knowledge and provenance; object storage holds permitted artifact bytes. Existing local run archives remain intact.

The central design rule is: **remember what happened, distinguish what was verified, and retrieve only what applies.** A long transcript alone is insufficient, and a confident agent explanation is not evidence that its proposed fix worked.

## 2. What exists and what must change

This assessment comes from reading the working tree on the date above. Claude is working concurrently, so implementers must recheck integration points before editing them.

| Existing component | Observed behavior | Consequence for memory |
|---|---|---|
| `src/orchestration/run_store.py` | SQLite tables for runs, stage attempts, artifacts, and compressed content-addressed blobs | Reuse historical evidence instead of requiring every agent to invent a new persistence format. |
| `record_stage` | Stores structured output, decision, report, and discoverable files | Useful foundation, but needs explicit invocation identity, check results, and branch lineage. |
| `record_stage` return value | Returns a stage row ID; current supervision calls do not retain it | Capture the ID or create a durable invocation UUID to associate memory with the exact attempt. |
| Artifact capture | Files above 200 MiB have path/hash metadata without stored content; missing paths are skipped | A historical path or hash does not establish that the bytes are available on another machine. |
| `get_stage_output` | Returns the latest recorded attempt for an agent | Cannot safely identify the accepted baseline: the latest attempt can be rejected or reverted. |
| `src/agents/master_agent.py` | Supervision, handoff validation, rerun snapshots, keep/revert decisions, run finalization | Central place to collect trustworthy outcomes and retrieve context for every agent. |
| Current rerun handling | An individually approved stage can belong to a branch later reverted | Separate stage approval from baseline acceptance. |
| Evaluation | A useful approved evaluation can report failed reproduction or inconclusive comparison | Agent task success and scientific result quality are different dimensions. |
| Crash finalization | May preserve stage records without final result/rerun details | Historical lineage can be unknown; do not infer acceptance from approval alone. |
| `src/llm.py` | Supervisor notes are process-global so worker threads inherit them | Adding another process-global memory list would risk cross-invocation contamination. |
| `requirements.txt` | Does not list scikit-learn as a direct application dependency | Do not assume TF-IDF is dependency-free just because reproduced projects may use scikit-learn. |

The existing run store is a local evidence archive. The proposed backend is the shared source of truth for published memory, its revisions, and its visibility. A machine's local paths never become remote artifact download instructions.

## 3. First-release architecture

### 3.1 Components

1. **Pipeline memory client:** builds task descriptions, asks for memory before agent execution, records the exact material supplied, and writes upload events locally.
2. **Local durable outbox:** a separate SQLite file, outside source control, containing immutable upload envelopes and delivery state. It is not a second full copy of the run database.
3. **Shared memory API:** authenticates clients, enforces workspace boundaries, validates submissions, serves retrieval, and controls artifact access.
4. **PostgreSQL:** stores source manifests, episodes, evidence, lesson revisions, retrieval records, review actions, and background jobs.
5. **Private object storage:** stores allowlisted, sanitized scripts and small supporting artifacts by checksum. It does not hold raw datasets in the first release.
6. **Background worker:** reconciles episodes into lessons, builds search documents, processes optional summaries, and handles interrupted jobs.
7. **Dashboard integration:** displays what was remembered, why it was retrieved, whether it was used, and what happened afterward.

Suggested deployment is managed PostgreSQL and private storage through Supabase, plus a small Python API and worker on a container host. Use the existing Flask stack for the API unless the implementation team has a specific reason to introduce another framework. Deploy API and worker as separate processes from the same versioned backend package. Do not put the shared service inside a desktop dashboard process.

The logical flow is:

**Machine A/B pipeline → local outbox → authenticated shared API → PostgreSQL + private storage → retrieval → agent invocation → new outcome evidence.**

No Redis, dedicated vector database, model fine-tuning, or remote execution service is needed for the first release. PostgreSQL can also hold the job queue. Hosted vendor choice is replaceable behind storage and database boundaries; the first release still requires an actual shared deployment.

### 3.2 Responsibility boundaries

- Agents produce results and may suggest lessons. They do not publish trusted memory directly.
- Master Agent remains the workflow authority. Memory cannot approve a paper, skip a prerequisite, authorize a rerun, or weaken a check.
- The API establishes whether a submission is structurally valid and attributable to a permitted client. Authentication does not prove the submitted scientific claim is true.
- Deterministic extraction and evidence checks establish supported observations. Human review handles disputed or broadly generalized lessons.
- Local run persistence continues even when the backend is unavailable. Backend failure must be visible, but it must not break an otherwise executable research pipeline.

## 4. Memory model

Use three concepts, with separate records for their evidence and use.

### 4.1 Episodes: what happened

An episode represents one agent invocation attempt or one orchestration decision. It captures task conditions, action, result, checks, and lineage. Episodes are append-only observations. Corrections are new events referencing the old observation.

Examples: a Replication attempt raised an import error; a Validation attempt recommended a review paper; an Evaluation invocation correctly identified leakage; Master Agent reverted a rerun.

Store failures immediately when possible. Do not wait until a run ends: a crash should not erase lessons from earlier attempts.

### 4.2 Lessons: what another agent may learn

A lesson is a concise, scoped claim derived from one or more episodes. It contains an applicability condition, observed action, observed outcome, supporting evidence, limits, and a recommendation or warning.

Suggested kinds:

- **Successful procedure:** an approach executed and passed relevant checks under stated conditions.
- **Failure warning:** a specific approach failed under stated conditions.
- **Verified correction:** a change addressed a recorded failure and passed an appropriate subsequent check.
- **Source lead:** a previously verified paper, dataset landing page, or repository to investigate again.
- **Reporting practice:** a verified presentation/reporting correction, when those agents exist.

No failure implies its opposite will succeed. A failed HTTP request is evidence about that attempt, not proof a dataset is permanently unavailable. A reverted experiment may still have executed correctly; preserve both facts.

### 4.3 Uses: what memory influenced

Record retrieval and use separately. A memory may be returned, included in the prompt, cited by an agent, actually adapted, or explicitly rejected. These are different events.

The system can prove what context it supplied. It usually cannot prove the model relied on that context. Agent self-reports are recorded as self-reports, never as verified causal attribution.

## 5. Data schema and invariants

Use UUIDs for shared identifiers and retain machine-local row IDs as source references. All dates use UTC. Every domain record belongs to a workspace. A workspace is a team memory boundary; projects within it can share experience only under an explicit sharing policy.

| Table | Main fields and purpose |
|---|---|
| `workspaces` | ID, name, sharing policy, retention settings, created time. |
| `clients` | ID, workspace, display name, credential digest, scopes, revocation time, last seen. Human reviewer identity is distinct from machine identity. |
| `source_runs` | Shared ID, origin archive UUID, local run ID, project, topic, status, code commit, dirty flag/fingerprint when available, provider metadata, start/end, source schema version, manifest completeness. |
| `run_configurations` | Immutable run-start configuration: requested memory mode, enabled roles/kinds, memory snapshot and policy versions, task/input fingerprints, pipeline configuration, model/provider settings, environment, seeds where supported, comparison group and assigned condition. |
| `run_metrics` | Run and optional invocation/baseline reference, versioned metric definition, value or explicit missing state, unit, population/split, aggregation, evidence reference, measurement time, producer/checker version, and validity status. Corrections append a superseding measurement. |
| `run_memory_exposure` | Run/invocation, requested mode, retrieval status, actual context-supplied status, included revision IDs/digest, artifact fetches, and failure/empty-result reasons; derives observed exposure separately from assigned condition. |
| `comparison_groups` | Workspace, task/input cohort, planned conditions, controlled configuration, fixed memory snapshot, repetition/seed assignment, eligibility rules, primary metrics, and analysis version. Observational groups are labeled separately from controlled trials. |
| `source_events` | Event UUID, client, run, event type, payload hash, payload schema, sequence, event time, receive time. Immutable ingestion ledger. |
| `episodes` | Invocation UUID, run, canonical agent role, attempt, branch, parent invocation, task snapshot, action summary, stage decision, observed execution outcome, check status, lineage status, error category, source event. |
| `rerun_decisions` | Rerun UUID, requester invocation, target role, before/after baseline IDs, exact affected invocation IDs, kept/reverted/failed/declined, reason, comparable metric evidence, decision time. |
| `evidence` | ID, episode/event/artifact reference, structured field path or bounded excerpt offsets, digest, evidence category, verification state, verification time. |
| `artifacts` | ID, workspace, source episode, original content hash, shared content hash, MIME type, size, storage key, availability, scan status, license/provenance, redaction manifest. |
| `lessons` | Stable lesson ID, workspace, owner role, kind, canonical scope fingerprint, current revision, lifecycle state, created time. |
| `lesson_revisions` | Immutable revision ID, lesson ID, condition/action/outcome text, constraints, evidence links, confidence dimensions, extractor version, prompt/model identity if summarized, author/reviewer, search document. |
| `lesson_support` | Lesson revision, evidence/episode, supporting or contradicting relation, independent run group, artifact availability. |
| `retrievals` | Request ID, invocation, task snapshot digest, query, candidate/rejection reasons, selected revision IDs, component scores, policy version, visibility watermark, duration. |
| `memory_uses` | Invocation, retrieval, exact included revisions and excerpts, inclusion digest, fetched artifacts, reported adaptation/ignore reason, later outcome references. |
| `review_actions` | Actor, target revision, approve/quarantine/supersede/dismiss, reason, time. Append-only audit. |
| `jobs` | Type, deduplication key, payload reference, pending/leased/done/failed state, lease expiry, retry count, error category. |
| `schema_migrations` | Applied migration version and checksum. |

Required constraints and indexes:

- Uniqueness on workspace + origin archive UUID + local run ID; run IDs alone do not identify imported archives reliably.
- Uniqueness on workspace + event UUID. Same ID and same payload hash is a retry; same ID with different payload is a conflict, not an overwrite.
- A committed event and its processing job are created in the same PostgreSQL transaction.
- Evidence and artifact references must belong to the same authorized workspace. Enforce this through composite relationships or equivalent database constraints, not only endpoint logic.
- Index episodes by workspace/run/role; lessons by workspace/state/role; pending jobs by state/lease; retrievals by invocation; evidence by source reference.
- Use a GIN index for the lesson search document. Add exact-match indexes for paper identity, dataset identity, task type, failure category, and environment family where needed.
- Lesson revisions never change in place. Suppression takes effect immediately in retrieval even while an index rebuild is pending.
- Keep scientific result status, local stage decision, branch acceptance, evidence completeness, and lesson lifecycle in separate fields.
- Every registered run has a configuration and metric manifest, including memory-disabled runs. Missing measurements carry a reason such as not reached, unavailable, interrupted, or invalid; they are never silently recorded as zero.
- Metric definitions identify direction, units, denominator, evaluation population/split, and aggregation. Compare only compatible definitions; retain raw measurements and evidence alongside derived differences.
- Index run metrics by workspace/run/definition and runs by comparison group, assigned memory condition, observed exposure, and outcome. Comparison membership never grants access across workspace/project visibility boundaries.

Lifecycle states for lessons: candidate, active, quarantined, superseded, retired. A revision becomes active only after its evidence and applicability pass the publication rules below. An unsupported LLM summary remains a candidate.

## 6. Task descriptions and applicability

Build a structured task snapshot at each invocation from information already known then. Never populate a pre-execution retrieval query with later outcomes.

Common fields: role, topic, target terms, task type, parent-paper identity if known, dataset identity/version/license, label definition, feature families, row/column counts, class balance bands, split strategy, metric definitions, code origin, environment, resource limits, current failure signature, and requested action.

Record absent information as unknown. Intake runs before a Parent Paper exists and must be able to search with topic and task type only.

Applicability rules depend on lesson kind:

- An environment fix needs compatible OS, interpreter/package family, and matching error context.
- A model/preprocessing procedure needs compatible task, label, feature structure, split assumptions, and feasible resource requirements.
- A source lead can be relevant by exact paper/dataset identity even when modeling details are unknown, but must be checked again.
- A general leakage warning can cross domains; a particular column-mapping instruction cannot.
- A reporting lesson can influence structure or checks, but historical result values cannot enter a current report as results.

Only permit cross-role retrieval through an explicit map. For example, Evaluation leakage findings can inform Cleaning and Replication; Validation source-quality findings can inform Research; Master Agent can inspect orchestration and role-level outcomes. Unrelated role matches are excluded.

## 7. Ingestion, outcome reconciliation, and publication

### 7.1 Events to capture

Capture run started; invocation started; memory context supplied; invocation finished; checks recorded; handoff accepted/declined; rerun opened; rerun kept/reverted/failed; baseline selected; run finalized; artifact verified; and review performed.

Also capture run configuration, metric measurements, memory retrieval/exposure outcomes, and run metric-manifest updates. Capture applies to all runs even when retrieval is disabled. Persist measurements incrementally through the same outbox and reconcile at finalization or after interruption; do not depend on a successful final stage to preserve a run's metrics. A disabled memory client must still allow run telemetry to be recorded and delivered independently.

Every envelope includes event UUID, workspace/client identity, origin archive UUID, run and invocation IDs where relevant, event schema version, occurrence time, sequence, producer version, and payload digest. The API derives workspace from credentials and rejects mismatches.

Current checks are partly available as log prose. Future integration should preserve machine-readable check IDs, pass/fail/unknown, severity, evidence reference, and checker version. Do not treat a missing check as passed.

### 7.2 Delivery and offline behavior

1. Persist the local run record using existing behavior.
2. Commit the corresponding upload envelope to the local outbox before marking it ready for delivery.
3. A background sender posts bounded batches and retains records until durable server acknowledgement.
4. The API commits event and processing job, then acknowledges. A worker crash after acknowledgement cannot lose accepted work.
5. Retry network errors, rate limits, and transient server failures with bounded exponential backoff and jitter. Respect server retry instructions. Quarantine permanent validation failures for review.
6. If local stage persistence succeeds but outbox creation fails, reconciliation scans the run archive and creates missing envelopes. If both fail, report a persistence gap explicitly; do not invent the missing event.
7. Events may arrive out of order. Resolve dependencies asynchronously; incomplete lineage remains unknown until supporting events arrive.

Start with a single PostgreSQL worker queue using short claim transactions, leases, and bounded retries. A job that dies while leased becomes eligible after lease expiry. Idempotent processing makes duplicate delivery safe.

Default retry delay: start at 2 seconds and cap at 5 minutes. Default retrieval deadline: 2 seconds end to end; return no memory when unavailable. These are initial operating targets to validate, not measured performance claims. Uploading artifacts must never block an agent on the critical retrieval path.

### 7.3 Publication rules

- A structurally valid observed failure with check or exception evidence may become an active, narrowly scoped warning.
- An executed procedure may become an active success lesson only when its relevant checks pass, required evidence exists, and branch acceptance is known where that matters.
- A verified correction requires a supported before/after relationship and a check showing the original issue was resolved.
- A diagnosis alone is a hypothesis. Store it as such; it cannot establish a verified correction.
- An inconclusive scientific outcome does not invalidate a correct data-discovery or evaluation lesson.
- A pipeline status of success does not certify valid metrics, complete data, or faithful replication.
- Broad lessons generalized from multiple cases require human review initially. Narrow deterministic templates can publish automatically.
- Optional LLM summarization may improve phrasing after deterministic extraction. Every factual clause must link to evidence; unsupported clauses are dropped or the revision remains a candidate.

When a rerun closes, update derived eligibility through new revisions/events. Do not delete its approved attempts. A reverted branch becomes a warning or a bounded observation, not the accepted solution. Validity corrections kept despite a larger paper gap count as successful validity work.

## 8. Retrieval and ranking

### 8.1 First-release search

Use PostgreSQL full-text retrieval plus structured identity and applicability matching. This puts the search index beside shared authoritative memory and avoids fitting different TF-IDF vocabularies on different machines. Exact identity searches must still work for terms that tokenization handles poorly.

Candidate process:

1. Authenticate and apply workspace/project visibility.
2. Exclude inactive revisions, unsupported schema versions, prohibited artifact content, and memory derived from the current run.
3. Apply the run's visibility watermark so later-arriving lessons do not change an in-flight benchmark unexpectedly.
4. Union exact paper/dataset/error matches with full-text candidates; cap at 50 candidates per request initially.
5. Apply kind-specific compatibility rules, distinguishing incompatible from unknown.
6. Rank successes and warnings independently.
7. Deduplicate near-identical lessons and diversify by source run and action family.
8. Select up to three actionable lessons and two warnings within a 2,000-token memory budget, subject to the current model's remaining context budget.
9. Return reasons, evidence references, limits, revision IDs, and artifact availability. Returning fewer or zero lessons is valid.

Current-run retries continue using current supervisor notes and rerun history. Excluding the current run from cross-run retrieval avoids duplicates and circular reinforcement. The backend still records those episodes immediately.

### 8.2 Rank useful experience without manufacturing a universal quality score

Do not rank all agents by distance to a paper's published score. That rewards accidental similarity and can penalize agents that correctly expose leakage or insufficient evidence.

Use an eligibility gate first, then separate relevance and trust dimensions. Initial relevance components: task-text match, structured task match, action/error match, and environmental compatibility. Initial trust components: directness of evidence, relevant check coverage, known outcome, and independent replication support.

For a first calibration, use relevance at 60%, evidence support at 25%, and compatible independent confirmations at 15%, each normalized within the candidate set under a versioned policy. Keep raw components and explanations. Missing evidence is unknown and lowers eligibility or confidence; it must never receive a positive default.

Paper gap is an optional role-specific signal only for comparable evaluation definitions, populations, splits, and units. Never combine accuracy gaps with arbitrary regression errors. A lower gap is not automatically a better method, and changing the comparison metric is not improvement evidence.

Independent support means distinct run groups with sufficiently different executions, not five retries from one run or copies of the same imported archive. Do not use raw retrieval count, agent praise, or repeated self-citation as evidence of quality.

Warning ranking emphasizes applicability and evidence that a failure occurred. A low success score must not bury the warning that would prevent repeating that failure.

### 8.3 Confidence and contradictions

Display confidence as dimensions and labels: direct observed evidence, single-run support, multiple independent runs, unknown lineage, contradictory evidence, or stale verification. It is not a calibrated probability of future success.

When seemingly conflicting lessons exist, first compare scope. Different datasets or package versions may explain the difference. If truly contradictory under matching conditions, link them, lower confidence, and present the conflict or quarantine a broad recommendation. A human correction produces an audited revision and suppresses superseded material.

Initial policy: source leads older than 30 days are marked stale and always require revalidation; environment-specific lessons with incompatible major versions cannot be recommended as direct fixes. The 30-day value is a tuning default. Stable procedural knowledge should not expire merely because it is old.

### 8.4 Later search improvements

Add embeddings only if a labeled retrieval benchmark shows lexical retrieval misses useful paraphrases. Store embedding model, dimension, input digest, and index version. Never mix vectors from incompatible models. PostgreSQL remains the metadata authority if vector indexing is added.

Optional LLM reranking can reorder eligible candidates using current feedback. It cannot restore filtered lessons, remove warning labels, or manufacture evidence. Invalid output or timeout falls back to deterministic ranking. Keep it off in the initial deployment to simplify cost and evaluation.

## 9. How every agent learns

| Agent | Memory to retrieve | Outcome to retain |
|---|---|---|
| Master | Invalid handoffs, redundant reruns, known prerequisite failures, effective supervision patterns | Whether a decision respected constraints, reduced repeated invalid actions, or resolved a verified blocker. |
| Intake | Useful task decompositions and target-definition corrections | Downstream confirmation of target alignment; never reuse another user's unstated intent. |
| Research | Effective search formulations and credible source leads | Verified qualifying papers found; stale/unverified leads remain labeled. |
| Validation | Replication-feasibility evidence, review/survey misclassification warnings, verification failures | Whether a selected paper actually had its own reproducible experiment. |
| Data | Dataset landing pages, access procedures, schema/target mismatch warnings | Current acquisition and target-fit checks; absence of a download is not permanent unavailability. |
| Cleaning | Label mappings under matching schema, merge pitfalls, leakage prevention | Verified retention/merge/label/split outcomes; no row content shared. |
| Code Discovery | Paper/repository identity leads and provenance mistakes | Current repository verification, license, authorship, and accessibility. |
| Replication | Verified scripts, environment fixes, execution failure warnings | Executed outcome, dependency snapshot, inference boundaries, validity and baseline acceptance. |
| Evaluation | Metric-definition mistakes and verified diagnostic procedures | Accurate supported findings; exposing invalid results is successful work. |
| Experiment | Interventions under compatible conditions, kept/reverted outcomes | Evidence-backed before/after observations, resource cost, validity effects, and uncertainty. |
| Report | Citation/integrity corrections and approved structural practices | Numeric provenance and claim checks; historical metrics are never current-run results. |
| Presentation | Effective figure/section structure and report-to-slide corrections | Verified consistency with the current approved report. |

Start with shadow retrieval for all existing roles. Activate context first for Replication, Experiment, Data, and Validation after review, then remaining roles. Provide hooks for Report and Presentation even while those agents are unfinished.

### 9.1 Prompt and invocation integration

The memory client receives explicit workspace, run, invocation, role, and task context. Master Agent performs retrieval before each supervised invocation; retries may retrieve warnings using the new failure signature. Record the exact prompt excerpts supplied, not just mutable lesson IDs.

Keep authoritative policy and current supervisor instructions separate from historical context. Historical content is bounded reference data and must not be promoted into system-level authority. State its evidence status, limits, and requirement for current-run verification.

Do not reuse the current global supervisor-notes variable for memory. Use an explicit invocation context that worker calls receive. If context variables are chosen, explicitly propagate context into thread-pool work; creating threads does not by itself guarantee inheritance. Concurrent runs must not share memory prompts or current-run IDs.

Give deterministic agents structured hints only at existing decision points. Do not add an LLM call simply to make every role appear to use memory. Add a small optional memory-use explanation to future agent outputs where useful; absence of that field must not break existing agents.

## 10. Script artifacts and shared evidence

Scripts are allowed only when provenance, content availability, license handling, scanning, and relevant execution/check evidence are recorded. “Verified script” means the stated checks passed for the recorded environment and input conditions; it does not mean universally safe or correct.

Artifact pipeline:

1. Select allowlisted source/configuration/log excerpts referenced by an episode. Exclude dataset files, environment secrets, credentials, local private paths, and unrestricted directory archives.
2. Scan for secrets and prohibited content before upload. Suspicious content is quarantined for review; scanning is best effort and not a proof of absence.
3. Produce a sanitized derivative when appropriate. Store original digest, derivative digest, exact redactions, and provenance. Do not label edited bytes as the exact script that executed.
4. For direct reuse as an executed script, retain original bytes only if upload is permitted and checks pass. Otherwise expose the derivative as an unexecuted reference template.
5. Upload to private workspace-scoped storage with a short-lived, exact-object upload grant and a proposed 5 MiB per-object limit. Verify final size and checksum server-side before marking available.
6. Authorize each download. Prefer streaming through the API initially; if signed URLs are introduced, bound their lifetime and record that revocation cannot instantly invalidate already downloaded bytes.
7. Fetch into an isolated temporary location and validate the hash. Never execute as part of retrieval. Normal replication safeguards, source attribution, and Master Agent checks still apply.

No remote fetch of arbitrary historical local paths. No automatic upload of raw datasets, full private documents, or third-party material without permission to share. Dataset metadata includes identity, schema summary, target meaning, license, and verified access pointers; it must avoid row values and sensitive column examples.

Evidence can be a structured, sanitized source excerpt plus hash and field reference. A hash alone cannot prove the interpretation of unavailable bytes. Mark hash-only artifacts unavailable; do not fabricate a download option. The backend must be able to show the actual permitted evidence supporting an active lesson.

## 11. API contracts

These are proposed interfaces, not implemented endpoints. Version the API and payloads independently.

| Operation | Contract |
|---|---|
| Ingest events | Authenticated bounded batch; returns per-event accepted/duplicate/conflict/rejected status and request ID. Accept out-of-order events without declaring them complete. |
| Retrieve memory | Task snapshot, role, invocation, visibility watermark, budget, allowed kinds; returns immutable revisions, reasons, evidence status, warnings, artifact references, and retrieval ID. |
| Record memory use | Retrieval ID, included revisions/excerpts digest, self-reported adaptations and verification references; idempotent per invocation/use event. |
| Register/upload artifact | Metadata and proposed hash/size; returns upload grant or rejection. Upload is provisional until content verification completes. |
| Read artifact/evidence | Workspace authorization, content restrictions, availability status, bounded download. |
| Inspect lesson | Current and prior revisions, supporting/contradicting evidence, usage history, lifecycle state. |
| Review lesson | Human reviewer role required; reason and expected current revision prevent overwriting concurrent review. |
| Reconcile run | Source manifest and known event IDs; identifies missing material and unresolved lineage without overwriting contradictory records. |
| Inspect/list runs | Filter by task, configuration, assigned memory mode, observed exposure, status, and comparison group; returns paginated run manifests and available/missing metrics, including failed and interrupted runs. |
| Compare runs | Authorized run IDs or comparison group plus versioned metric definitions; returns compatible per-run values, differences, sample counts, exclusions and reasons, missingness, exposure labels, and provenance. Incompatible metrics are flagged rather than combined. |
| Health/readiness | Minimal public health; authenticated dependency and backlog diagnostics. |

Errors use stable categories: unauthorized, forbidden, invalid schema, payload too large, idempotency conflict, missing dependency, rate limited, temporarily unavailable. Keep stack traces and secrets out of client responses. Bound batches initially to 100 events and 1 MiB of JSON. Paginate inspection endpoints.

Use one high-entropy, revocable credential per machine with workspace-limited ingestion/retrieval scopes. Store only credential digests server-side; keep client secrets in local environment configuration. Review permission requires a separate human credential/session. Never give every pipeline machine a database-owner or storage service-role key.

## 12. Historical backfill

Backfill reads existing archives; it does not alter historical results or silently repair them.

1. Assign a durable origin archive UUID and register the machine/import source.
2. Inventory runs, attempts, artifacts, blob availability, schema shape, and completeness. Produce a dry-run report before upload.
3. Parse structured fields first. Use report prose only as explicitly labeled weaker evidence. Do not execute archived scripts during import.
4. Map stage rows to exact attempts. Resolve rerun membership from recorded structured history when unambiguous; leave it unknown otherwise.
5. Import episodes with evidence completeness and original code identity. A Git commit without dirty-state information is incomplete version provenance.
6. Candidate success lessons from uncertain or reverted branches remain ineligible as accepted solutions. Failure observations can still be useful when directly supported.
7. Import available allowlisted artifacts under the artifact policy. Record missing/oversize/hash-only content accurately.
8. Deduplicate by origin IDs and payload hashes. Detect copied archives; repeated imports never increase independent support.
9. Generate candidate lessons and inspect all first-import lessons manually, given the small historical archive described in project status.
10. Publish approved memory, run retrieval benchmarks, and save an import summary with counts, exclusions, unresolved lineage, and policy versions.

A literature-review Parent Paper reduces trust in reproduction claims tied to that paper. It does not invalidate an independently evidenced import-error fix or a correct diagnosis that no original experiment existed. Apply trust restrictions to the affected claim, not blindly to every stage in the run.

## 13. Consistency, reproducibility, and failure handling

- Preserve local run archive and shared event ledger independently. Upload acknowledgement means the shared record is durable, not that its lesson is published.
- Avoid a distributed transaction between local SQLite and PostgreSQL. Outbox delivery plus reconciliation handles gaps explicitly.
- Attach lessons to exact baseline branches. An accepted baseline points to a set of invocation/artifact identities, not whichever agent record happens to be newest.
- At run start, establish a server visibility watermark. A benchmark reuses a fixed memory snapshot; a normal run may refresh only at recorded boundaries. New quarantines and access revocations always override the snapshot.
- Snapshot model/provider identities actually served when available, policy/checker versions, task context, selected revisions, and artifact hashes.
- A deleted or quarantined lesson must not survive through an unbounded local cache. First release uses no offline retrieval cache; during an outage, run without cross-run memory and continue queuing evidence.
- Preserve delivery events on disk through restarts. Alert on queue growth; never silently discard events to make an outage look healthy.
- If a critical check cannot run, memory cannot make it pass. If retrieval is empty or fails, existing agent behavior remains the fallback.
- Crash reconciliation labels abandoned running invocations interrupted only after a configured inactivity window and confirmation that their producer is no longer active. Do not auto-classify a long model call as a scientific failure.

## 14. Security, operations, and deployment

### 14.1 Deployment sequence

Create separate development/test and team-production workspaces. Provision managed PostgreSQL and a private bucket, apply versioned migrations, deploy the API and worker, configure TLS and secrets, register two machine clients, and run the cross-machine acceptance scenario before agent prompts consume memory.

The pipeline machines need the API URL, workspace identifier, and their own scoped credential. Only the hosted backend needs database and storage credentials. Add configuration names to the authoritative environment-variable documentation during implementation; never commit values.

Use a dedicated database schema with no public direct table access. The API database role has only required privileges; the migration role is separate. If Supabase is used, explicitly restrict exposed schemas and storage policies. Service-role/secret keys are backend-only, as required by [Supabase's data security guidance](https://supabase.com/docs/guides/database/secure-data). Private storage access must be authorized, following [the bucket access model](https://supabase.com/docs/guides/storage/buckets/fundamentals).

### 14.2 Operational controls

- Request IDs, structured logs, redacted errors, and audit events for reads of scripts and changes to lesson visibility.
- Metrics: ingestion delay, outbox depth/oldest age, failed jobs, unresolved lineage, published/quarantined lessons, retrieval latency, empty retrieval rate, warning inclusion, artifact failures, and cross-workspace access denials.
- Initial targets to validate: metadata delivery within one minute when online; retrieval within two seconds; no cross-workspace results; exact duplicate imports create zero new independent support.
- Database backup daily with an initial 24-hour recovery-point target and one-working-day restore target. Configure object retention/versioning to match. Verify available provider features and pricing before choosing a plan; this document assumes no free-tier durability guarantee.
- Perform a restore drill using a database backup plus artifact inventory. Check hashes, lesson evidence links, and retrieval visibility afterward.
- Track DB size, artifact bytes, request volume, worker utilization, and optional model tokens. Set explicit team spending limits when choosing hosting. Keep automated summarization/reranking off until budget is set.
- Database and object deletion are separate operations. Mark unpublished objects for collection only after a grace period and an evidence/reference audit.

### 14.3 Retention and correction

Keep run evidence according to team/project policy; retain summary revisions and audit history by default. Archive redundant artifacts only after ensuring required evidence remains inspectable. A human can quarantine misleading memory immediately without deleting the source record.

For privacy removal, suppress retrieval first, then remove prohibited content from active records, object storage, caches, and retained exports under a documented backup-expiry policy. Retain a minimal tombstone if permitted so reconciliation does not reimport deleted content. Explain that downloaded copies and older backups require separate handling.

## 15. Implementation work packages

These are future changes. No item below is implemented by this planning task.

| Phase | Deliverables | Exit criteria |
|---|---|---|
| 0: Freeze contracts | Agree event schemas, workspace model, evidence definitions, branch IDs, and artifact policy; coordinate with Claude on current orchestration changes | Sample historical records map without guessed outcomes; ownership of shared-file edits is explicit. |
| 1: Shared foundation | Hosted API/PostgreSQL/private storage, migrations, scoped client authentication, isolated test environment | Two machines authenticate; cross-workspace denial, idempotent ingestion, and persistent acknowledgements pass. |
| 2: Reliable recording | Durable outbox, event capture hooks, lineage, reconciliation, job leases | Offline/restart/reordered-event tests lose no acknowledged data and expose missing evidence. |
| 2a: Run measurement and comparison | All-run configuration/metric manifests, actual memory exposure, comparison groups, run inspection/comparison API and dashboard | Memory-enabled and disabled runs, including failures/interruption, appear with attributable metrics; incompatible comparisons and missing values are explicit. Required before controlled memory rollout. |
| 3: Evidence and backfill | Deterministic extractors, artifact verification, candidate lesson revisions, import report | All initial lessons have inspectable support; uncertain branches are excluded as accepted solutions. |
| 4: Retrieval in shadow | Structured/full-text retrieval, separate warnings, audit records, dashboard inspection | Labeled retrieval benchmark passes; no prompt behavior changed yet. |
| 5: Controlled use | Invocation-scoped context, bounded prompts, Replication/Experiment/Data/Validation rollout | Exact supplied context is recorded; normal checks remain effective; cross-run metrics do not appear as current results. |
| 6: All roles | Remaining roles and future Report/Presentation hooks, contradiction review, reporting provenance | Every role has a meaningful retrieval/retention contract; deterministic roles receive only actionable hints. |
| 7: Validate learning | Fixed-snapshot ablations, repeated trials, outcome/cost analysis | Evidence supports benefit or identifies specific failure modes; disable harmful lesson classes or policies. |

Suggested module boundaries: a memory client and outbox under orchestration; a separate backend package for API, persistence, extraction, ranking, and worker; versioned database migrations; dashboard adapters; integration tests. Exact names can follow repository conventions. Avoid putting networking, lesson scoring, and database migrations directly into Master Agent.

Coordinate edits to `master_agent.py`, `llm.py`, `run_store.py`, configuration, and dashboard readers with Claude. Build the backend and event contracts independently first. Do not replace the handoff driver or merge unrelated pending work as a side effect of backend development.

The first useful vertical slice is: **Machine A finishes one evidenced successful attempt and one failed attempt; Machine B retrieves the relevant procedure and warning, verifies any script artifact, runs normally, and sends its own outcome back.** Finish this before advanced ranking, embeddings, or automatic broad generalization.

## 16. Validation plan

### 16.1 Backend correctness

- Same event delivered repeatedly creates one ledger record; conflicting payload under the same key is rejected.
- Worker crash after acknowledgement and before processing does not lose the event. Lease expiry safely retries processing.
- Rerun decisions arriving before attempts are reconciled later; unresolved membership remains unknown.
- An approved attempt from a reverted branch is never returned as the accepted baseline solution.
- Correct leakage detection is retained as successful Evaluation work even when paper similarity declines.
- Missing checks, metrics, artifacts, and version information remain unknown.
- Identical imported archives and repeated retries do not multiply independent support.
- Quarantined or superseded revisions disappear from new retrieval immediately.
- Unicode survives end to end, including Windows upload queues and paper-derived text.

### 16.2 Access and artifact correctness

- Machine A and Machine B in the same workspace can exchange permitted memory; another workspace cannot enumerate or download it.
- Revoked credentials stop ingestion and retrieval. Machine credentials cannot approve disputed lessons.
- Artifact checksum mismatch, oversize upload, path traversal, secret-bearing content, and unavailable hash-only bytes are handled explicitly.
- Sanitized script derivatives retain their changed-byte status and are never represented as executed originals.
- No raw dataset rows or source credentials appear in uploads, prompts, logs, or retrieval results.
- Memory containing hostile instructions cannot alter policy, authorize tools, or bypass Master Agent checks.

### 16.3 Retrieval and learning quality

Create a small human-labeled benchmark of at least 30 task queries across available roles, including no-match, exact identity, similar-topic/different-target, environment mismatch, relevant failure warning, and contradictory-lesson cases. Label acceptable lessons and known harmful matches before tuning.

Initial acceptance targets: at least one useful result among the top three for 80% of queries that have an applicable lesson; zero known incompatible recommendations in the labeled set; all specifically labeled critical warnings included when their conditions match. These are proposed release gates, not claims about current performance. Report sample size and misses.

Compare three conditions under fixed source data and memory snapshot: no cross-run memory; procedure memory only; procedures plus failure warnings. Use repeated trials where feasible because model outputs and providers vary. Keep current-run retry feedback enabled in all conditions.

Measure valid execution rate, repeated known failures, verified corrective success, invalid result/leakage rate, appropriate inconclusive conclusions, comparable reproduction gap where available, time, model calls/tokens, and manual intervention. Do not summarize learning with only one paper-gap number.

Test on held-out tasks and exclude their runs from the memory bank. Historical timestamps alone do not prevent leakage if future outcomes were backfilled into an earlier simulated query. Freeze the published revision set explicitly.

### 16.4 Every-run metrics and comparison workflow

The dashboard should provide a run table and a side-by-side comparison selected by run IDs or comparison group. Show task/input identity, run outcome, assigned memory condition, actual memory exposure, metric values and units, validity, missingness, and configuration differences. Link each value to its source measurement and accepted baseline where relevant. Keep rejected/rerun attempt measurements inspectable without presenting them as final accepted results.

Required measurement families:

- **Scientific outcomes:** current-run metrics under their exact definitions, paper reference values where verified, comparable reproduction gaps, split/data identity, and validity/check results. A paper reference is a comparison target, not a measured run result.
- **Execution outcomes:** completed/failed/interrupted/inconclusive status, stages reached, check failures, retries/reruns, repeated known failures, verified corrections, and manual intervention counts.
- **Resources:** elapsed run/stage time, model calls, input/output tokens, retrieval time, and observed cost with currency and pricing basis when available. Unknown provider usage or cost stays unknown.
- **Memory exposure:** assigned mode (disabled, shadow, procedures only, procedures plus warnings), retrieval attempts/errors/empty results, supplied lesson/warning counts and revisions, and agent-reported adaptations. Shadow retrieval supplies no historical context to agents; adaptation reports remain self-reports.

Record both assigned condition and actual exposure. An enabled run with an outage or zero relevant results remains assigned to its original condition and is labeled as receiving no memory. A run that receives context for only some invocations is labeled partial exposure. Preserve these distinctions in exports and analyses; do not reclassify runs after seeing their results. Historical runs with unknown memory configuration remain unknown rather than presumed controls.

For a controlled comparison, define the task cohort, metric definitions, condition assignment, repetition count, and exclusion rules before execution. Hold task/input data, pipeline/code configuration, model settings, resource limits, and evaluation protocol constant where feasible; record deviations and provider variability. Pair repeated runs by task and seed when supported. Freeze one eligible memory snapshot for treatment runs, exclude all trial runs from that snapshot, and prevent lessons from earlier trial executions entering later trial executions. Current-run retry feedback remains enabled consistently.

Show per-run results first, then paired differences or group summaries with sample size, spread/uncertainty, completion and validity rates, and missing-value counts. Primary summaries retain all assigned runs and report failures; scientific metrics available only for completed valid runs must identify that subset and its denominator. Never make memory look beneficial by silently dropping failed runs or treating absent scores as zero. Exposure-based summaries may supplement the assigned-condition comparison but must disclose their selection bias.

Arbitrary historical runs can be compared descriptively, with configuration differences visible. Such comparisons can suggest useful patterns but cannot establish that memory caused an improvement. Controlled repeated trials provide stronger evidence. Store comparison definitions and analysis versions so the same selection and calculations can be reproduced.

Acceptance checks must demonstrate that every run remains inspectable after offline delivery/restart; disabled and shadow runs record metrics without supplied context; partial/outage exposure is visible; corrected metrics preserve prior evidence; incompatible units/splits are rejected or separated; and failed/interrupted runs remain in group counts. Verify that the dashboard and API show the same attributable values and comparison membership.

## 17. Illustrative learning cycle

This example is hypothetical and describes expected behavior; it is not a claim about an existing run.

1. Cleaning produces data with an unsuitable split. Evaluation supplies verifiable evidence of leakage.
2. The first Cleaning attempt is recorded as locally approved but later shown invalid for the affected experiment.
3. A rerun changes the split; the subsequent Evaluation verifies the leakage condition is resolved. Master Agent keeps the valid branch, even if its published-score gap is larger.
4. The backend retains a narrowly scoped leakage warning, a supported correction, exact branch relationships, and the limited conditions under which they apply.
5. On another machine, a compatible task retrieves that warning and procedure. A different label or split design would restrict applicability.
6. The new agents verify the current dataset and execute the current pipeline. They do not import the old metrics.
7. The new outcome adds independent supporting or contradicting evidence. The original record remains unchanged, while a new lesson revision reflects the additional evidence.

This is the desired learning loop: observe, validate, retain, retrieve, adapt, execute, and revise confidence.

## 18. Remaining decisions and proposed defaults

| Decision | Proposed default | When needed |
|---|---|---|
| Hosting account and monthly ceiling | Team-owned managed PostgreSQL/private storage; small API/worker deployment; explicit spending ceiling | Before provisioning. |
| Sharing boundary | One team workspace, project labels, explicit private-project exclusion | Before first upload. |
| Human reviewer | Named team member; separate review credentials | Before initial backfill publication. |
| Historical content sharing rights | Summaries plus permitted script artifacts; no raw rows or unrestricted paper copies | Before artifact upload. |
| First retrieval policy | Structured matches plus PostgreSQL full-text; no embeddings or LLM reranking | Before phase 4. |
| Retention period | Retain evidence during the project; agree an end-of-course export/deletion policy | Before production launch. |
| Existing large dataset blobs | Stay local; never automatically replicated to shared storage | Before backfill. |

The first two user preferences in section 1 are settled. Remaining choices should not block designing contracts or preparing isolated backend work, but production provisioning and uploads need the actual account, budget, sharing boundary, and credentials.

## 19. Source notes

Repository evidence: `docs/PROJECT_STATUS.md`, `src/orchestration/run_store.py`, `src/orchestration/handoff.py`, `src/agents/master_agent.py`, `src/agents/experiment_agent.py`, `src/llm.py`, `requirements.txt`, and applicable integrity requirements in local `CLAUDE.md`. Working-tree observations are a point-in-time assessment, not a guarantee that Claude's concurrent changes preserve the same interfaces.

External documentation checked while planning:

- [PostgreSQL full-text search](https://www.postgresql.org/docs/current/textsearch.html): basis for database-native lexical search. Ranking weights and release thresholds in this plan are proposed application policy.
- [PostgreSQL trigram matching](https://www.postgresql.org/docs/17/pgtrgm.html): optional approximate identity/string matching if needed; not required to establish the first retrieval path.
- [Supabase data security](https://supabase.com/docs/guides/database/secure-data): keep secret/service-role keys on the backend and restrict data access.
- [Supabase storage buckets](https://supabase.com/docs/guides/storage/buckets/fundamentals) and [storage access control](https://supabase.com/docs/guides/storage/security/access-control): basis for private, authorized artifact access.

No pricing, hosting capacity, measured performance, scientific success, or completed backend implementation is asserted by this plan.
