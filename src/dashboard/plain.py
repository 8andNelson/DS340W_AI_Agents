"""
Plain-language summaries of agent outputs, for readers who are not
technical (the Agent outputs view's "In plain words" card).

Every sentence is built by code from fields the agent already recorded in
the run database -- no LLM call, so a summary is instant, free, and cannot
add a fact that is not in the record. A field that is missing is left out,
never guessed. Like the rest of the dashboard, this reads only what the
pipeline wrote and imports nothing from it.

summarize(attempt) -> {"headline": str, "points": [str], "next": str}
"""
import re

# Same pattern as the Evaluation Agent's (copied: the dashboard imports no pipeline module).
LOWER_IS_BETTER = re.compile(r"brier|loss|error|mse|mae|fpr|fnr|falsepositive|falsenegative|falsealarm|misclassif")

DECISIONS = {
    "Approved": "The Master Agent checked this work and accepted it.",
    "Retrying": "The Master Agent found a problem and asked the agent to try again.",
    "Rejected": "The Master Agent did not accept this work.",
}

# The Evaluation Agent's labels, in everyday words.
VERDICTS = {
    "EXACT": "matches the paper's results",
    "SURPASSED": "scores higher than the paper",
    "APPROXIMATE": "is close to the paper, but a little lower",
    "FAILED": "is clearly lower than the paper",
    "INCONCLUSIVE": "could not be compared with the paper",
}
METRIC_WORDS = {
    "NOT_IN_PAPER": "the paper does not report this one",
    "NOT_A_SCORE": "a time, loss or cost, so it is not compared",
    "INCONCLUSIVE": "could not be compared",
}
MAX_POINTS = 6


def _num(value) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() and abs(value) >= 1000 else f"{value:.4g}"
    return str(value)


# Technical words in the agents' own text (warnings, reasons), in everyday words.
JARGON = ((re.compile(r"\bLLM\b"), "the AI model"), (re.compile(r"\bvenv\b"), "Python environment"),
          (re.compile(r"\bstderr\b"), "error output"), (re.compile(r"\bstdout\b"), "printed output"))


def _first_sentence(text, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    for pattern, words in JARGON:
        text = pattern.sub(words, text)
    if not text:
        return ""
    end = text.find(". ")
    sentence = text if end < 0 else text[:end + 1]
    return sentence if len(sentence) <= limit else sentence[:limit - 1].rstrip() + "..."


def _plural(n, word: str) -> str:
    return f"{_num(n)} {word}{'' if n == 1 else 's'}"


def _points(value: float) -> float:
    return value * 100 if abs(value) <= 1 else value


def _signed_points(m: dict) -> float | None:
    """A compared metric's difference from the paper, + better. Records
    from before signed_difference_points existed get it from the values
    (lower is better for error-type rates, as in the Evaluation Agent)."""
    if m.get("difference_points") is None:
        return None
    if isinstance(m.get("signed_difference_points"), (int, float)):
        return m["signed_difference_points"]
    pub, rep = m.get("published_value"), m.get("reproduced_value")
    if not isinstance(pub, (int, float)) or not isinstance(rep, (int, float)):
        return None
    diff = _points(rep) - _points(pub)
    return -diff if LOWER_IS_BETTER.search(re.sub(r"[^a-z0-9]+", "", str(m.get("metric", "")).lower())) else diff


def _paper_line(paper: dict) -> str:
    if not paper.get("title"):
        return ""
    extra = ", ".join(filter(None, [_num(paper.get("year")) if paper.get("year") else "", paper.get("venue") or ""]))
    return f"\"{paper['title']}\"" + (f" ({extra})" if extra else "")


# --- One function per agent ---

def _intake(o: dict) -> dict:
    if "title" in o:  # it read a supplied paper (intake_agent.read_paper), not a topic
        points = [p for p in [
            f"Title: {o['title']}." if o.get("title") else "The title printed on the paper was not found.",
            f"What the models will try to predict: {', '.join(o['target_terms'])}." if o.get("target_terms")
            else "No prediction target that the paper itself uses was found.",
            *[str(n) for n in o.get("notes") or []],
        ] if p]
        return {"headline": "Read the title and prediction target from your paper", "points": points,
                "next": "The run continues with your paper as the Parent Paper."}
    points = [p for p in [
        f"Research area: {o['domain']}." if o.get("domain") else "",
        f"Kind of problem: {o['ml_task'] if isinstance(o.get('ml_task'), str) else ', '.join(o.get('ml_task') or [])}."
        if o.get("ml_task") else "",
        f"What the models will try to predict: {', '.join(o['target_terms'])}." if o.get("target_terms") else "",
        f"It wrote {_plural(len(o['search_queries']), 'search phrase')} to look for papers."
        if o.get("search_queries") else "",
    ] if p]
    return {"headline": "Understood the research topic", "points": points,
            "next": "The Research Agent searches for academic papers on this topic."}


def _research(o) -> dict:
    papers = o if isinstance(o, list) else []
    years = sorted({p.get("year") for p in papers if isinstance(p, dict) and isinstance(p.get("year"), int)})
    with_code = sum(1 for p in papers if isinstance(p, dict) and p.get("has_code"))
    points = [f"It found {_plural(len(papers), 'candidate paper')}."]
    if years:
        points.append(f"They were published between {years[0]} and {years[-1]}." if len(years) > 1
                      else f"They were published in {years[0]}.")
    if papers:
        points.append(f"{_num(with_code)} of them say their code is available online.")
    return {"headline": f"Found {_plural(len(papers), 'paper')}", "points": points,
            "next": "The Validation Agent checks that each paper is real and peer reviewed, then picks one."}


def _validation(o: dict) -> dict:
    parent = o.get("recommended_parent_paper") or {}
    counts = [(len(o.get(k) or []), word) for k, word in
              (("verified", "confirmed as real and peer reviewed"), ("unverified", "could not be fully checked"),
               ("rejected", "rejected (for example too old, or not peer reviewed)"))]
    points = [f"{_plural(n, 'paper')} {word}." for n, word in counts if n]
    if parent:
        points.append(f"Chosen paper to reproduce (the \"Parent Paper\"): {_paper_line(parent)}.")
    why = _first_sentence(o.get("justification"))
    if why:
        points.append(f"Why: {why}")
    return {"headline": "Checked the papers and chose one to reproduce" if parent else "Checked the papers",
            "points": points,
            "next": "The Data Agent gets the paper's dataset, and the Code Discovery Agent looks for its code."}


def _parent_intake(o: dict) -> dict:
    parent = o.get("parent_paper") or {}
    points = [p for p in [
        f"Paper: {_paper_line(parent)}." if parent else "",
        f"What the models will try to predict: {', '.join(o['target_terms'])}." if o.get("target_terms") else "",
        "The paper search and selection were skipped, because you supplied this paper.",
    ] if p]
    return {"headline": "Using the paper you supplied", "points": points,
            "next": "The Data Agent gets the paper's dataset, and the Code Discovery Agent looks for its code."}


def _data(o: dict) -> dict:
    datasets = o.get("selected_datasets") or []
    points = []
    if datasets:
        names = ", ".join(str(d.get("display_name") or d.get("name") or "a dataset") for d in datasets[:3])
        points.append(f"It downloaded {_plural(len(datasets), 'dataset')}: {names}.")
    if o.get("total_entries"):
        points.append(f"In total that is {_num(o['total_entries'])} rows of data.")
    if o.get("parent_dataset_found") is True:
        points.append("This is the same dataset the paper used.")
    elif o.get("parent_dataset_found") is False:
        points.append("The paper's own dataset could not be downloaded, so a similar dataset is used instead.")
    if any(d.get("sampled") for d in datasets):
        points.append("To keep the run fast, a sample of the rows is used, with the same share of each class.")
    if not datasets:
        points.append("No usable dataset could be downloaded.")
    return {"headline": f"Collected {_plural(len(datasets), 'dataset')}" if datasets else "Found no usable data",
            "points": points, "next": "The Cleaning Agent merges and cleans the data, then splits it."}


def _cleaning(o: dict) -> dict:
    points = []
    if o.get("rows_total") is not None:
        points.append(f"The clean dataset has {_num(o['rows_total'])} rows"
                      + (f" and {_num(o['columns_total'])} columns." if o.get("columns_total") else "."))
    if o.get("datasets_attempted"):
        points.append(f"It merged {_num(o.get('datasets_merged') or 0)} of {_num(o['datasets_attempted'])} "
                      "downloaded dataset(s).")
    if o.get("label_column"):
        points.append(f"The column the models learn to predict is \"{o['label_column']}\".")
    counts = o.get("label_counts") or {}
    values = [v for v in counts.values() if isinstance(v, (int, float))] if isinstance(counts, dict) else []
    if len(values) >= 2 and sum(values):
        share = min(values) / sum(values) * 100
        points.append(f"The rarer outcome is {share:.2g}% of the rows, so it is much less common than the other."
                      if share < 20 else f"The rarer outcome is {share:.2g}% of the rows.")
    split = o.get("split_rows") or {}
    if split:
        points.append(f"Split for training and testing: {_num(split.get('train'))} rows to learn from, "
                      f"{_num(split.get('test'))} to test on, {_num(split.get('validation'))} kept aside.")
    if o.get("conflicts"):
        points.append(f"{_plural(len(o['conflicts']), 'dataset')} could not be merged; the reason is logged.")
    return {"headline": "Cleaned and prepared the data", "points": points,
            "next": "The Replication Agent trains the paper's model on this data."}


def _code(o: dict) -> dict:
    repo = (o.get("repository") or {}).get("repo_url") or ""
    if o.get("status") == "FOUND" and repo:
        return {"headline": "Found the authors' code", "points": [f"Code: {repo}.",
                "The Master Agent checks it for safety before it is run, and runs it only in a locked-down sandbox."],
                "next": "The Replication Agent runs this code."}
    return {"headline": "Did not find the authors' code",
            "points": ["No published code could be confirmed as the authors' own.",
                       "This is normal: the code is rebuilt from the paper's description instead."],
            "next": "The Replication Agent writes the model from the paper's description and runs it."}


def _replication(o: dict) -> dict:
    if o.get("status") == "INFEASIBLE":
        # Stopped before running: this computer cannot run the paper's method.
        return {"headline": "This computer cannot run the paper's method",
                "points": [_first_sentence(w, 400) for w in (o.get("warnings") or [])[-1:]]
                + [_first_sentence(o.get("notes"), 300)],
                "next": "The run goes on without a baseline, and the report says why. A more powerful computer "
                        "(more memory, or a GPU) could run it."}
    source = ("the authors' own code" if o.get("source") == "cloned_repository"
              else "code rebuilt from the paper's description")
    ok = o.get("status") == "OK"
    points = [f"It used {source}."]
    points.append("The model trained and produced results." if ok else
                  "The model did not finish successfully." if o.get("status") != "TIMEOUT" else
                  "The model took too long and was stopped.")
    if o.get("duration_seconds") is not None:
        points.append(f"Training and testing took about {_num(round(o['duration_seconds']))} seconds.")
    if o.get("inferred_assumptions"):
        points.append(f"The paper did not state {_plural(len(o['inferred_assumptions']), 'detail')} (such as "
                      "settings), so they were filled in with reasonable choices and marked as such.")
    if o.get("revision_round"):
        points.append(f"This is revision {_num(o['revision_round'])}, made on request after an earlier check.")
    if not ok and o.get("warnings"):
        points.append(f"What went wrong: {_first_sentence(o['warnings'][-1])}")
    return {"headline": "Ran the paper's model" if ok else "The model did not run", "points": points,
            "next": "The Evaluation Agent compares the results with the paper's." if ok else
            "The Master Agent decides whether to try again."}


def _evaluation(o: dict) -> dict:
    status = o.get("status") or "INCONCLUSIVE"
    points = []
    for m in (o.get("metrics") or [])[:MAX_POINTS]:
        name, pub, rep = m.get("metric", "metric"), m.get("published_value"), m.get("reproduced_value")
        diff = _signed_points(m)
        if diff is not None and m.get("classification") in VERDICTS:
            direction = "better" if diff > 0 else "worse"
            change = "the same" if abs(diff) < 0.05 else f"{abs(diff):.1f} points {direction}"
            points.append(f"{name}: paper {_num(pub)}, this run {_num(rep)} ({change}).")
        elif m.get("classification") in METRIC_WORDS:
            value = _num(rep if rep is not None else pub)
            why = METRIC_WORDS[m["classification"]]
            if m["classification"] == "INCONCLUSIVE":
                why = ("the paper gives no value for it" if pub is None else
                       "this run did not print it" if rep is None else why)
            points.append(f"{name}: {value + ' -- ' if value else ''}{why}.")
    if any(f.get("category") == "data_leakage" for f in ((o.get("review") or {}).get("static_findings") or [])):
        points.append("It found a flaw that can make a score look better than it really is (\"data leakage\") "
                      "and asked for it to be fixed.")
    decision = (o.get("decision") or {}).get("action")
    nxt = {"revise_replication": "The Replication Agent fixes the code, and the results are checked again.",
           "rerun_data_agent": "The Data Agent collects more data, and the results are checked again.",
           }.get(decision, "The Experiment Agent looks for valid ways to reach or beat the paper's score.")
    return {"headline": f"The result {VERDICTS.get(status, 'was checked')}", "points": points, "next": nxt,
            "note": "\"Points\" are percentage points: 0.93 against 0.98 is 5 points."}


def _experiment(o: dict) -> dict:
    d = o.get("decision") or {}
    points = []
    if d.get("action") == "rerun_agent":
        what = (d.get("instructions") or [""])[0]
        points.append(f"Next idea: ask the {d.get('target_agent') or 'agent'} to change one thing"
                      + (f" -- {_first_sentence(what)}" if what else "."))
        if d.get("reason"):
            points.append(f"Why: {_first_sentence(d['reason'])}")
        if d.get("method_source") == "beyond_paper":
            points.append("This tries a method the paper does not describe"
                          + (f" ({', '.join(d.get('new_methods') or [])})" if d.get("new_methods") else "")
                          + ". The report will say so.")
        headline, nxt = "Proposed a change to try", "The Master Agent checks the idea, then runs it if it is sound."
    else:
        points.append("It decided that no further change is likely to give a valid, higher score.")
        headline, nxt = "Decided to stop experimenting", "The Report Agent writes the final report."
    for note in (o.get("decision_notes") or [])[:1]:
        points.append(_first_sentence(note))
    return {"headline": headline, "points": points, "next": nxt}


def _report(o: dict) -> dict:
    points = []
    if o.get("report_path"):
        points.append("The final research report is written. Open it on the Report & slides page.")
    if o.get("removed_claims"):
        points.append(f"{_plural(len(o['removed_claims']), 'sentence')} were removed, because the run's records "
                      "did not support them.")
    if o.get("status") == "DEGRADED":
        points.append("Some written sections are missing, because the language model did not answer; the tables "
                      "and numbers are still there.")
    return {"headline": "Wrote the final report", "points": points,
            "next": "The Presentation Agent makes the slides."}


def _presentation(o: dict) -> dict:
    points = [f"The slide deck has {_plural(len(o.get('slides') or []), 'slide')}. Open it on the Report & slides page."]
    if o.get("removed_claims"):
        points.append(f"{_plural(len(o['removed_claims']), 'bullet point')} were removed, because the records did "
                      "not support them.")
    return {"headline": "Made the presentation slides", "points": points, "next": "The run is complete."}


SUMMARIZERS = {
    "Intake Agent": _intake, "Research Agent": _research, "Validation Agent": _validation,
    "Parent Paper Intake": _parent_intake, "Data Agent": _data, "Cleaning Agent": _cleaning,
    "Code Discovery Agent": _code, "Replication Agent": _replication, "Evaluation Agent": _evaluation,
    "Experiment Agent": _experiment, "Report Agent": _report, "Presentation Agent": _presentation,
}


def summarize(attempt: dict) -> dict:
    """The plain summary of one recorded attempt (see the module docstring).
    Never raises: an output it cannot read gets a short generic summary."""
    decision = attempt.get("decision") or ""
    output = attempt.get("output")
    summary = None
    if output is not None and attempt.get("agent") in SUMMARIZERS:
        try:
            summary = SUMMARIZERS[attempt["agent"]](output)
        except Exception:
            summary = None
    if summary is None:
        first = _first_sentence(attempt.get("report_text"))
        summary = {"headline": "This attempt ended with an error" if output is None else "Finished its work",
                   "points": [first] if first else [], "next": ""}
    summary["points"] = [p for p in summary.get("points") or [] if p][:MAX_POINTS + 1]
    summary["decision"] = DECISIONS.get(decision, f"Master Agent decision: {decision}." if decision else "")
    if decision in ("Retrying", "Rejected"):
        summary["next"] = ("The agent tries again with the Master Agent's advice." if decision == "Retrying"
                           else "After several tries the Master Agent did not accept this step. Its log says what happens next.")
    return summary
