# Research replication pipeline walkthrough

## Master Agent · Walk the file from top to bottom

Open [master_agent.py](../src/agents/master_agent.py). Start at the logging helpers, then scroll down in the order below.

### 1. Logging helpers

`_log_entry` gives each approval, correction, or retry a timestamp, target, reason, and next action. `_append_log` saves entries under a lock so parallel stages can log safely, while `_print_entry` surfaces decisions in the console and Slack. [Logging helpers](../src/agents/master_agent.py#L91)

### 2. Format functions

`_format_*_report` turns each stage’s structured output into text for the attempt report and Slack summary. The content changes by stage. Research lists papers, Replication shows what ran, and Evaluation explains the metric comparison. [Formatter section](../src/agents/master_agent.py#L163) · [Evaluation formatter](../src/agents/master_agent.py#L312)

### 3. Check functions and retries

`_check_*` returns blocker entries. An empty list lets the stage advance. Paper checks enforce eligibility, Data and Cleaning require usable outputs, Replication requires a working baseline, and Report and Presentation check their deliverables. Code Discovery, Evaluation, and Experiment can return meaningful negative outcomes and still pass their checkers. [Checkers](../src/agents/master_agent.py#L429)

```python
            report_text = report_fn(output)
            slack_client.post_message(message_formatter.format_finished(
                agent_name, "Completed", _summarize_for_slack(report_text),
            ))
            blockers = checker_fn(output)
```

Point to **`report_fn` and `checker_fn`**. The formatter explains the attempt and the checker decides whether it can advance. If checks fail or the agent raises an exception, `_diagnose` gives the next attempt guidance. The default limit is five attempts. [Diagnosis](../src/agents/master_agent.py#L810) · [Supervision](../src/agents/master_agent.py#L853)

### 4. Rerun logic

- **`_review_rerun_request`** checks whether the proposal is actionable and avoids redundant work. **`_open_rerun`** saves the current files and results and records the old metrics. [Review](../src/agents/master_agent.py#L1309) · [Open](../src/agents/master_agent.py#L1627)
- **`_close_rerun`** compares the new evaluation with the saved baseline. **`_undo_rerun`** restores the snapshot if the rerun fails or an experiment produces an invalid or materially worse result. A validity fix can be kept even when the score drops. See the [rerun decision excerpt](#reverting-a-rerun). [Restore](../src/agents/master_agent.py#L1685) · [Close](../src/agents/master_agent.py#L1727)

### 5. Run logic

```python
        requests = _agent_module(target).handoff(output, ctx)
        if target == "Experiment Agent" and not requests:
```

- **`_drive_initial`** dispatches independent first-pass stages. **`_drive`** follows requests through the pipeline and reruns. Point to **`handoff`**. Agents propose the next stage and Master decides what runs. An Experiment handoff with no requests ends the loop. [Initial dispatch](../src/agents/master_agent.py#L1851) · [Driver](../src/agents/master_agent.py#L1937)
- **`run_pipeline`** is the entry point. **`_run_stages`** creates shared context, branches between a supplied paper and a topic, and sends the final results to Report and Presentation after the rerun loop. [Entry point](../src/agents/master_agent.py#L2074) · [Stage sequence](../src/agents/master_agent.py#L2158)

## Research, Parent Paper, and handoffs

### How search builds a candidate list

```python
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        for query in search_queries:
            print(f"  > '{query}'")
            try:
                results = web_search.search_academic_papers(query, count=RESULTS_PER_QUERY)
            except Exception as e:
                print(f"  WARNING: Search failed for '{query}' - {e}")
                continue
            futures.append((query, llm.submit_with_context(pool, _extract_papers_from_results,
                                                         results, project_topic, query)))
```

- **`search_academic_papers`** routes queries through Brave with SerpApi fallback. **`continue`** lets the next query run after a failure.

- **`submit_with_context`** overlaps extraction with later searches. Results are collected in query order before deduplication.

[Open the Research loop](../src/agents/research_agent.py#L239) · [Open the search router](../src/search/web_search.py#L63).

### How Validation ranks the Parent Paper

```python
    rows = [_compute_criteria_row(p) for p in pool]
    ranked_indices = sorted(range(len(pool)), key=lambda i: rows[i]["composite_score"], reverse=True)

    chosen_idx = ranked_indices[0]
    runner_up_indices = ranked_indices[1:JUSTIFICATION_TOP_N]

    audit = _code_availability_audit(pool)
    deciding = _is_code_deciding_factor(rows, chosen_idx)
```

- **`composite_score`** orders the comparison rows. The highest score becomes the recommendation.
- **`_is_code_deciding_factor`** checks whether removing the code bonus changes the winner.
- **`_check_validation_and_selection`** is Master’s separate approval gate. It requires a verified pool of at least five and checks the recommended paper.

[Open the ranking](../src/agents/validation_agent.py#L561) · [Open the scoring rules](../src/agents/validation_agent.py#L365) · [Open the approval checks](../src/agents/master_agent.py#L518).

### How the handoff leads to a baseline

```python
    return [handoff_request("validation_agent", "Data Agent",
                            {"parent_paper": parent, "ranked_pool": (result or {}).get("ranked_pool") or []},
                            f"Parent Paper recommended: '{parent.get('title', '')}'. Its dataset is needed next."),
            handoff_request("validation_agent", "Code Discovery Agent", {},
                            "Find the Parent Paper's implementation alongside dataset discovery.")]
```

- **`handoff_request`** gives Master two independent searches to dispatch after approval.

```python
        code, cleaning = ctx["code_discovery_result"], ctx["cleaning_result"]
        return lambda: module.run(parent, code, cleaning, **kwargs)
    if target == "Evaluation Agent":
        args = (parent, ctx["replication_result"], ctx["cleaning_result"], ctx["dataset_result"])
```

- **`code` and `cleaning`** are where the two branches meet. Replication needs the code discovery result and prepared data.
- **`args`** gives Evaluation the baseline output, split details, and dataset provenance alongside the paper.

[Open the Validation handoff](../src/agents/validation_agent.py#L737) · [Open Master’s stage dispatch](../src/agents/master_agent.py#L1561).

### Reverting a rerun

```python
        invalid = _rerun_invalid(rerun, ctx)
        if invalid:
            _undo_rerun(f"{invalid} ({change})", ctx, drive, decision="Reverted")
            return False
        lowered = margin_before is not None and (margin_after is None
                                                 or margin_after < margin_before - GAP_TOLERANCE_POINTS)
        if lowered:
            _undo_rerun(f"the score went down ({change})", ctx, drive, decision="Reverted")
            return False
```

- **`_rerun_invalid`** checks whether the experiment produced a valid, comparable baseline.
- **`_undo_rerun`** restores the previous result when the experiment is invalid or the margin drops beyond tolerance.
- These checks apply to Experiment requests. Fixing leakage can justify keeping an Evaluation rerun with a lower score.

[Open the rerun decision](../src/agents/master_agent.py#L1727).

## Memory files · Follow a case across runs

### 1. Build and save records · `retain.py`

```python
        retain.retain_run(run_id)
```

**`finish_run`** passes the saved run ID to retention. [Open the run store call](../src/orchestration/run_store.py#L375).

```python
    memories.extend(_rerun_memories(run))
    memories.extend(_published_leak_memories(run))
    memories.extend(_setting_change_memories(run))
    return memories, lineage_source
```

- **`build_memories`** walks attempts, keeps their lineage, and extracts approved work and failure warnings. These final lines add records that depend on evidence across the run.
- **`retain_run`** writes the records. It can then turn evidence into lessons through reflection and check whether earlier runs support them.

[Open record construction](../src/memory/retain.py#L875) · [Open retention](../src/memory/retain.py#L915) · [Open reflection and confirmation](../src/memory/retain.py#L979).

### 2. Score the original evidence · `ranking.py`

```python
    parts = {"outcome": outcome, "approval": approval, "validity": validity, "downstream": downstream,
             "trust": (TRUST_SCORE[trust], f"trust: {trust}")}
    quality = sum(WEIGHTS[k] * float(v) for k, (v, _) in parts.items())
    reasons = {"formula": "0.35 outcome + 0.20 approval + 0.20 validity + 0.15 downstream + 0.10 trust",
               **{k: {"value": round(float(v), 3), "reason": r} for k, (v, r) in parts.items()}}
    return round(quality, 4), reasons
```

- **`parts`** holds the five evidence values. **`reasons`** stores what contributed to the score and why.

[Open base quality and warning usefulness](../src/memory/ranking.py#L53) · [Open the adjustment for judged help or harm](../src/analysis.py#L182).

**Warning usefulness** uses a separate set of weights. A cause earns 0.4, a passing next attempt 0.3, a specific signature 0.2, and trust up to 0.1.

```python
    quality = sum(WARNING_WEIGHTS[k] * v for k, (v, _) in parts.items())
```

[Open warning scoring](../src/memory/ranking.py#L63).

### 3. Update quality from judged use · `analysis.py`

**Effective quality** uses the stored score until a use is judged. As help or harm is recorded, the stored score gets less weight.

```python
    base = float(memory.get("quality") or 0)
    helped, hurt = int(memory.get("times_helped") or 0), int(memory.get("times_hurt") or 0)
    judged = helped + hurt
    if not judged:
        return round(base, 4)
    weight = 1 / (1 + judged / 3)
    return round(weight * base + (1 - weight) * (helped + 1) / (judged + 2), 4)
```

[Open the effective quality calculation](../src/analysis.py#L182).

### 4. Rank and log recalled cases · `recall.py`

```python
        rel, parts = relevance(query, m)
        if rel < MIN_RELEVANCE:
            continue
        score = rel * (0.5 + 0.5 * ranking.effective_quality(m)) * freshness(m, code_version)
        scored[group].append((group, m, round(score, 4), parts))
    for group in scored.values():
        group.sort(key=lambda item: (-item[2], item[1].get("created_at") or ""))
    return scored["good"][:k_good] + scored["warning"][:k_warn]
```

- **`MIN_RELEVANCE`** filters cases before the final score combines task fit, effective quality, and version compatibility.

- **`k_good` and `k_warn`** give positive cases and warnings separate slots in the prompt.

- **`MemoryContext.finish`** logs the cases shown and the labels the model reports using. Labels outside the prompt are dropped.

[Open recall ranking](../src/memory/recall.py#L225) · [Open the prompt pack builder](../src/memory/recall.py#L459). · [Open usage logging](../src/memory/recall.py#L555).

### 5. Record outcomes · `learning.py`

```python
            helped, _why = _outcome(row, attempts, by_agent, lineage, warnings, final)
            if helped is None:
                report["not_judged"] += 1
                continue
            if record_outcome(row["id"], row["memory_id"], helped):
                report["helped" if helped else "hurt"] += 1
```

- **`_outcome`** checks approval, whether the attempt was kept or reverted, and validity where relevant. For Master’s diagnosis, it checks whether the next attempt passed.
- **`record_outcome`** records one judged use per memory per attempt. Those help and harm counts feed `effective_quality` on later runs.

[Open outcome rules](../src/memory/learning.py#L104) · [Open the learning loop](../src/memory/learning.py#L132).

## Quick reference

- **Stage order** → Intake → Research → Validation → Data and Code Discovery → Cleaning and Replication → Evaluation → optional reruns → Report and Presentation. Cleaning prepares Data’s output. Replication waits for Cleaning and Code Discovery. [Master dispatch](../src/agents/master_agent.py#L1533)
- **Run state and memory** → Shared context carries the current inputs and outputs. The saved run supplies evidence for future cases and warnings. [Run context](../src/orchestration/run_context.py) · [Memory retention](../src/memory/retain.py#L875)
- **Rebuild order** → Start with a supplied paper, repeatable data splits, and a comparable baseline. Add discovery and approval, reversible reruns, and cross-run memory. Use a fixed paper for memory comparisons. [Run instructions](../README.md#8-run-the-pipeline) · [Comparison controls](../src/orchestration/comparison.py)

Source excerpts reflect the working tree during the in-progress performance pass.
