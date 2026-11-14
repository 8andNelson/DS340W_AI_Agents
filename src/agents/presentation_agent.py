"""
Presentation Agent (Milestone 14).

Builds the project's slides (presentations/<slug>/slides.pptx, 16:9) from
the Report Agent's output -- never from a new number:

  * report_facts.json (the run's recorded facts) gives every table, chart
    and value. Charts are native PowerPoint charts (editable), made from
    the same values as the report's tables.
  * The report's prose, already checked against the facts, is the only
    narrative source. One LLM call shortens it into slide bullets; every
    bullet then goes through the Report Agent's grounding check (_ground)
    again, so a bullet cannot add a number, a citation or a claim the
    report could not. If the call fails, the first sentences of the
    checked prose are used instead.
  * Each slide's speaker notes name the facts it was made from.

The slides follow CLAUDE.md's presentation structure (SLIDES). Master Agent
checks the file itself (_check_presentation): the slide titles, and that
the comparison table and the results chart hold exactly the recorded values.
"""
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from src import llm
from src.agents import report_agent

REPO_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = REPO_ROOT / "presentations"
SLIDE_TIMEOUT_SECONDS = 180
MAX_BULLETS = 3
BULLET_MAX_CHARS = 170

COMPARISON_TABLE, RESULTS_CHART = "Comparison table", "Results chart"
SLIDES = ("Title", "Research Problem", "Motivation", "Literature Review", "Parent Paper", "Dataset", "Methodology",
          "Reproduction Workflow", "Baseline Reproduction", "Proposed Changes", "Experimental Results", "Comparison",
          "Conclusions", "Limitations", "Future Work", "References")
BULLET_KEYS = ("problem", "motivation", "methodology", "conclusions", "limitations", "future_work")
# Report prose each bullet set is shortened from (and falls back to).
PROSE_SOURCE = {"problem": "introduction", "motivation": "introduction", "methodology": "methodology",
                "conclusions": "conclusion", "limitations": "limitations", "future_work": "future_work"}

INK, INK_2, MUTED_FILL = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0xF1, 0xF0, 0xEC)
GRID, NOT_KEPT = RGBColor(0xE1, 0xE0, 0xD9), RGBColor(0xC3, 0xC2, 0xB7)
BLUE, ORANGE, WHITE = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34), RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0xB2, 0x6A, 0x00)


def _fmt(value) -> str:
    """How a recorded value is written on a slide (the check uses this too)."""
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _clip(text, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


# --- Bullets ---

def _sentences(text: str) -> list:
    return [s for s in report_agent._SENTENCE.split(re.sub(r"\s+", " ", text or "").strip()) if s]


def fallback_bullets(prose: dict) -> dict:
    """The first sentences of the checked report prose, per slide."""
    out = {}
    for key in BULLET_KEYS:
        sentences = _sentences(prose.get(PROSE_SOURCE[key], ""))
        if key == "motivation":
            sentences = sentences[1:] or sentences
        out[key] = [_clip(s, BULLET_MAX_CHARS) for s in sentences[:MAX_BULLETS]]
    return out


SYSTEM = (
    "You turn the checked prose of a research report into short slide bullets. Use only the REPORT PROSE and "
    "FACTS given. Never add a number, paper, dataset or result that is not in them; copy numbers exactly; cite "
    "papers only as [P1]-style keys that appear in the prose. State the reproduction outcome exactly as "
    "facts.evaluation_status says. Answer with one JSON object and nothing else, with these keys, each a list of "
    f"at most {MAX_BULLETS} bullets of at most 18 words: " + ", ".join(BULLET_KEYS) + "."
)


def write_bullets(facts: dict, prose: dict) -> tuple[dict, str, str]:
    """({slide key: [bullets]}, the model that wrote them, error or "")."""
    if not any(prose.values()):
        return {}, "", "the report has no prose to shorten"
    brief = {"topic": facts["topic"], "research_question": facts["research_question"],
             "evaluation_status": facts["evaluation"]["status"], "mean_gap_points": facts["evaluation"]["mean_gap_points"],
             "replication_source": facts["replication"]["source"], "rerun_count": facts["experiments"]["rerun_count"]}
    prompt = (f"FACTS:\n{json.dumps(brief, default=str)}\n\nREPORT PROSE:\n{json.dumps(prose, ensure_ascii=False)}\n\n"
              "Write the slide bullets as one JSON object.")
    try:
        content = llm.chat(prompt, system=SYSTEM, timeout=SLIDE_TIMEOUT_SECONDS, agent="presentation_agent")
    except Exception as e:
        return {}, "", f"{type(e).__name__}: {e}"
    parsed = llm.parse_json(content, default={})
    if not isinstance(parsed, dict):
        return {}, "", "the model's answer was not a JSON object"
    bullets = {k: [_clip(b, BULLET_MAX_CHARS) for b in (parsed.get(k) or []) if isinstance(b, str) and b.strip()]
               [:MAX_BULLETS] for k in BULLET_KEYS}
    if not any(bullets.values()):
        return {}, "", "the model's answer had no bullets"
    return bullets, llm.last_served_by(), ""


def ground_bullets(bullets: dict, facts: dict) -> tuple[dict, list]:
    known = report_agent.fact_numbers(facts)
    kept, removed = {}, []
    for key, items in bullets.items():
        kept[key] = []
        for b in items:
            text, dropped = report_agent._ground(b, facts, known)
            if dropped:
                removed += [dict(r, slide=key) for r in dropped]
            if text and not dropped:
                kept[key].append(text)
    return kept, removed


# --- Slide building blocks ---

def _text(slide, x, y, w, h, lines, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = str(line)
        run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color
    return box


def _bullets(slide, x, y, w, h, items, size=18):
    return _text(slide, x, y, w, h, [f"•  {i}" for i in items], size=size)


TITLE_SHAPE = "Slide title: "  # each slide's title box is named after its SLIDES entry


def _slide(prs, title: str, subtitle: str = "", notes: str = "", heading: str = None):
    """A blank 16:9 slide with its title (or `heading`, shown instead)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.45), Inches(0.12), Inches(0.75))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    if heading is None:
        box = _text(slide, 0.45, 0.35, 12.4, 0.8, title, size=30, bold=True)
    else:
        box = _text(slide, 0.9, 2.2, 11.5, 1.6, heading, size=34, bold=True)
    box.name = TITLE_SHAPE + title
    if subtitle:
        _text(slide, 0.45, 1.05, 12.4, 0.5, subtitle, size=14, color=INK_2)
    number = len(prs.slides)
    _text(slide, 12.2, 7.0, 0.9, 0.35, str(number), size=11, color=INK_2, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def _table(slide, x, y, w, headers, rows, widths=None, size=12, row_h=0.36):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * (len(rows) + 1)))
    table = shape.table
    for c, width in enumerate(widths or [w / len(headers)] * len(headers)):
        table.columns[c].width = Inches(width)
    for r, row in enumerate([headers] + rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if r == 0 else (MUTED_FILL if r % 2 == 0 else WHITE)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = r == 0
                    run.font.color.rgb = WHITE if r == 0 else INK
    return shape


def _bar_chart(slide, x, y, w, h, categories, series, point_colors=None, number_format="General", name=None):
    data = CategoryChartData()
    data.categories = categories
    for label, values, _ in series:
        data.add_series(label, values)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    if name:
        frame.name = name
    chart = frame.chart
    chart.has_legend = len(series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
    chart.category_axis.reverse_order = True  # first category at the top
    chart.category_axis.tick_labels.font.size = Pt(12)
    chart.value_axis.tick_labels.font.size = Pt(11)
    chart.value_axis.major_gridlines.format.line.color.rgb = GRID
    chart.value_axis.minimum_scale = 0  # bars start at zero, so lengths are true to the values
    top = max((v for _, values, _ in series for v in values), default=0)
    if 0 < top <= 1:
        chart.value_axis.maximum_scale = 1
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.overlap = -5
    plot.has_data_labels = True
    plot.data_labels.number_format = number_format
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.font.size = Pt(11)
    for s, (_, _, color) in zip(plot.series, series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = color
    for i, color in enumerate(point_colors or []):
        point = plot.series[0].points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = color
    return chart


# --- The deck ---

def comparable_metrics(facts: dict) -> list:
    return [m for m in facts["evaluation"]["metrics"] if isinstance(m.get("published_value"), (int, float))
            and isinstance(m.get("reproduced_value"), (int, float))]


def build(facts: dict, bullets: dict, path: Path) -> None:
    parent, ev, exp = facts["parent_paper"], facts["evaluation"], facts["experiments"]
    rep, data, cl, lit = facts["replication"], facts["data"], facts["cleaning"], facts["literature"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 Title
    slide = _slide(prs, "Title", notes="Source: report_facts.json (topic, parent_paper, run).",
                   heading=f"Reproducing \"{_clip(parent['title'] or facts['topic'], 120)}\"")
    _text(slide, 0.9, 3.9, 11.5, 0.6, facts["topic"], size=20, color=INK_2)
    _text(slide, 0.9, 5.6, 11.5, 0.8, [f"Run {facts['run']['run_id'] or 'not recorded'} | {facts['run']['date']}",
                                      "A multi-agent reproduction: research, data, code, evaluation, report"],
          size=14, color=INK_2)

    # 2 Research Problem
    slide = _slide(prs, "Research Problem", notes="Source: research_question, ml_task, target_terms; bullets from "
                                                  "the report's checked introduction.")
    _table(slide, 0.6, 1.6, 7.0, ["Item", "Value"],
           [["Research question", _clip(facts["research_question"] or facts["topic"], 70)],
            ["ML task", _fmt(facts["ml_task"])], ["Prediction target", ", ".join(facts["target_terms"]) or "n/a"]],
           widths=[2.2, 4.8], size=14, row_h=0.5)
    _bullets(slide, 0.6, 3.9, 12.0, 3.0, bullets.get("problem") or [])

    # 3 Motivation
    slide = _slide(prs, "Motivation", notes="Source: the report's checked introduction.")
    _bullets(slide, 0.6, 1.7, 12.0, 4.5, bullets.get("motivation") or ["See the report's introduction."], size=22)

    # 4 Literature Review
    slide = _slide(prs, "Literature Review",
                   f"{lit['count']} papers from {lit['window_start']} on; {lit['verified_count']} fully verified",
                   notes="Source: literature.papers (rejected papers are excluded).")
    shown = sorted(lit["papers"], key=lambda p: (not p["is_parent"], p["status"] != "VERIFIED"))[:8]
    _table(slide, 0.6, 1.7, 12.1, ["Key", "Title", "Year", "Status"],
           [[p["key"] + (" (Parent)" if p["is_parent"] else ""), _clip(p["title"], 85), _fmt(p["year"]), p["status"]]
            for p in shown], widths=[1.4, 8.3, 0.9, 1.5])
    if len(lit["papers"]) > len(shown):
        _text(slide, 0.6, 6.8, 12.0, 0.4, f"{len(lit['papers']) - len(shown)} more in the report's references.",
              size=12, color=INK_2)

    # 5 Parent Paper
    slide = _slide(prs, "Parent Paper", notes="Source: parent_paper, selection.")
    _text(slide, 0.6, 1.5, 12.0, 1.0, _clip(parent["title"], 140), size=22, bold=True)
    _table(slide, 0.6, 2.7, 6.6, ["Item", "Value"],
           [["Authors", _clip(", ".join(parent["authors"]) or "not recorded", 50)], ["Year", _fmt(parent["year"])],
            ["Venue", _clip(parent["venue"] or "not recorded", 50)], ["DOI", parent["doi"] or "n/a"],
            ["Status", parent["status"]], ["Text read", parent["paper_text_status"] or "n/a"]],
           widths=[1.6, 5.0], size=13)
    why = _sentences(parent["selection_justification"])[:2]
    if why:
        _text(slide, 7.6, 2.7, 5.2, 3.8, ["Why it was selected:"] + [_clip(s, 200) for s in why], size=14)

    # 6 Dataset
    slide = _slide(prs, "Dataset", notes="Source: data.datasets, cleaning.")
    if data["parent_dataset_found"] is False:
        _text(slide, 0.6, 1.4, 12.0, 0.5, "The Parent Paper's own dataset could not be obtained: substitute data.",
              size=14, color=WARN, bold=True)
    _table(slide, 0.6, 2.0, 12.1, ["Dataset", "Source", "Rows used", "Rows in source", "Sampled"],
           [[_clip(d["name"], 50), _fmt(d["source"]), _fmt(d["rows"]), _fmt(d["original_rows"]), _fmt(d["sampled"])]
            for d in data["datasets"][:5]], widths=[4.6, 2.0, 1.9, 2.0, 1.6])
    split = cl["split_rows"]
    _bullets(slide, 0.6, 2.0 + 0.36 * (len(data["datasets"][:5]) + 1) + 0.5, 12.0, 2.2, [
        f"Merged: {_fmt(cl['rows_total'])} rows, {_fmt(cl['columns_total'])} columns",
        f"Train / test / validation: {_fmt(split.get('train'))} / {_fmt(split.get('test'))} / "
        f"{_fmt(split.get('validation'))} (seed {_fmt(cl['split_seed'])})"]
        + ([f"Paper's dataset: {_clip(data['paper_dataset'], 80)}"] if data["paper_dataset"] else []), size=16)

    # 7 Methodology
    slide = _slide(prs, "Methodology", notes="Source: the report's checked methodology; replication.inferred_assumptions.")
    _bullets(slide, 0.6, 1.6, 12.0, 2.8, bullets.get("methodology") or ["See the report's methodology."])
    if rep["inferred_assumptions"]:
        _text(slide, 0.6, 4.5, 12.0, 2.6, [f"Inferred, not stated by the authors ({len(rep['inferred_assumptions'])}):"]
              + [f"•  {_clip(a, 140)}" for a in rep["inferred_assumptions"][:3]], size=14, color=INK_2)

    # 8 Reproduction workflow (what this project ran)
    slide = _slide(prs, "Reproduction Workflow", "What the agents ran for this reproduction",
                   notes="Source: data, cleaning, replication, evaluation, experiments.")
    steps = [("Data", f"{_fmt(data['total_entries'])} rows"),
             ("Cleaning", f"{_fmt(cl['rows_total'])} rows, split"),
             ("Baseline", "authors' code" if rep["source"] == "cloned_repository" else "reconstructed"),
             ("Evaluation", _fmt(ev["status"])),
             ("Reruns", f"{exp['rerun_count']} ({exp['kept_count']} kept)")]
    for i, (name, detail) in enumerate(steps):
        shape = slide.shapes.add_shape(MSO_SHAPE.PENTAGON, Inches(0.45 + i * 2.5), Inches(3.0), Inches(2.4), Inches(1.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = BLUE if i % 2 == 0 else ORANGE
        shape.line.fill.background()
        frame = shape.text_frame
        frame.word_wrap = True
        for j, line in enumerate((name, detail)):
            p = frame.paragraphs[0] if j == 0 else frame.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = line
            run.font.size, run.font.bold, run.font.color.rgb = Pt(16 if j == 0 else 12), j == 0, WHITE

    # 9 Baseline Reproduction
    slide = _slide(prs, "Baseline Reproduction", notes="Source: replication, code.")
    source = (f"Authors' code: {rep['repo_url']} (commit {_fmt(rep['commit_sha'])})" if rep["source"] == "cloned_repository"
              else "Reconstructed from the paper: not the authors' code")
    _text(slide, 0.6, 1.5, 12.0, 0.6, source, size=18, bold=True)
    _table(slide, 0.6, 2.4, 6.5, ["Item", "Value"],
           [["Status", _fmt(rep["status"])], ["Entry point", _fmt(rep["entry_point"])],
            ["Exit code", _fmt(rep["exit_code"])], ["Duration (s)", _fmt(rep["duration_seconds"])],
            ["Revision round", _fmt(rep["revision_round"])]], widths=[2.3, 4.2], size=14, row_h=0.45)
    if rep["generated_by"]:
        _text(slide, 7.6, 2.4, 5.2, 2.0, ["Script written by:"] + [f"•  {m}" for m in rep["generated_by"][:4]],
              size=14, color=INK_2)

    # 10 Proposed Changes
    slide = _slide(prs, "Proposed Changes", f"{exp['rerun_count']} rerun(s), {exp['kept_count']} kept",
                   notes="Source: experiments.reruns. A rerun that moved the score away was reverted.")
    if exp["reruns"]:
        _table(slide, 0.6, 1.7, 12.1, ["#", "Agent rerun", "Instruction", "Gap before", "Gap after", "Kept"],
               [[_fmt(r["rerun"]), _fmt(r["target_agent"]), _clip("; ".join(r["instructions"]) or r["goal"], 70),
                 _fmt(r["gap_before"]), _fmt(r["gap_after"]), _fmt(r["kept"])] for r in exp["reruns"][:8]],
               widths=[0.6, 2.4, 5.6, 1.3, 1.3, 0.9])
    else:
        _text(slide, 0.6, 1.8, 12.0, 0.6, "No rerun was run.", size=20)
    if exp["end_reason"]:
        _text(slide, 0.6, 6.4, 12.0, 0.6, f"Loop ended: {_clip(exp['end_reason'], 150)}", size=13, color=INK_2)

    # 11 Experimental Results (chart)
    metrics = comparable_metrics(facts)
    slide = _slide(prs, "Experimental Results", f"Overall: {_fmt(ev['status'])}",
                   notes="Source: evaluation.metrics (metrics stated on both sides).")
    if metrics:
        _bar_chart(slide, 0.6, 1.6, 12.1, 5.4, [m["metric"] for m in metrics],
                   [("Published (Parent Paper)", [m["published_value"] for m in metrics], BLUE),
                    ("Reproduced (this run)", [m["reproduced_value"] for m in metrics], ORANGE)],
                   name=RESULTS_CHART)
    else:
        _text(slide, 0.6, 1.8, 12.0, 1.0, "No metric is stated on both sides, so no value can be compared.", size=20)

    # 12 Comparison (table, and the gap over the reruns)
    slide = _slide(prs, "Comparison",
                   (f"Mean gap to the paper: {_fmt(ev['mean_gap_points'])} percentage points"
                    if ev["mean_gap_points"] is not None else "The gap to the paper is not measurable"),
                   notes="Source: evaluation.metrics, experiments.")
    rows = [[m["metric"], _fmt(m["published_value"]), _fmt(m["reproduced_value"]), _fmt(m["difference_points"]),
             _fmt(m["classification"])] for m in ev["metrics"][:8]]
    if rows:
        _table(slide, 0.6, 1.6, 6.9 if exp["reruns"] else 12.1, ["Metric", "Published", "Reproduced", "Diff (pts)",
                                                                "Class"], rows, size=12).name = COMPARISON_TABLE
    steps = ([("Baseline", exp["baseline_gap_points"], True)] if exp["baseline_gap_points"] is not None else []) + \
        [(f"Rerun {r['rerun']}", r["gap_after"], r["kept"]) for r in exp["reruns"]
         if isinstance(r.get("gap_after"), (int, float))]
    if len(steps) >= 2:
        _bar_chart(slide, 7.8, 1.6, 5.0, 5.0, [s[0] for s in steps],
                   [("Gap to the paper (points)", [s[1] for s in steps], BLUE)],
                   point_colors=[BLUE if s[2] else NOT_KEPT for s in steps])
        _text(slide, 7.8, 6.6, 5.0, 0.4, "Grey: rerun not kept", size=11, color=INK_2)

    # 13-15 Conclusions, Limitations, Future Work
    slide = _slide(prs, "Conclusions", notes="Source: the report's checked conclusion.")
    _bullets(slide, 0.6, 1.7, 12.0, 4.5, bullets.get("conclusions") or
             [f"Reproduction status: {_fmt(ev['status'])}. See the report."], size=22)
    slide = _slide(prs, "Limitations", notes="Source: the report's checked limitations and recorded limitations.")
    recorded = []
    if data["parent_dataset_found"] is False:
        recorded.append("Substitute data: the paper's own dataset was not used")
    if rep["source"] != "cloned_repository":
        recorded.append("Baseline reconstructed from the paper, not the authors' code")
    if lit["verified_count"] < lit["count"]:
        recorded.append(f"{lit['count'] - lit['verified_count']} paper(s) could not be fully verified")
    _bullets(slide, 0.6, 1.7, 12.0, 2.6, recorded or ["No recorded limitation."], size=18)
    _bullets(slide, 0.6, 1.7 + 0.5 * max(len(recorded), 1) + 0.4, 12.0, 2.6, bullets.get("limitations") or [], size=16)
    slide = _slide(prs, "Future Work", notes="Source: the report's checked future work.")
    _bullets(slide, 0.6, 1.7, 12.0, 4.5, bullets.get("future_work") or ["See the report's future work."], size=22)

    # 16 References
    slide = _slide(prs, "References", notes="Source: literature.papers; the report lists all of them.")
    refs = [f"[{p['key']}] {_clip(', '.join(p['authors'][:3]) or 'Authors not recorded', 40)} ({_fmt(p['year'])}). "
            f"{_clip(p['title'], 90)}." + (f" doi:{p['doi']}" if p["doi"] else "")
            + ("" if p["status"] == "VERIFIED" else f" ({p['status']})")
            for p in sorted(lit["papers"], key=lambda p: not p["is_parent"])[:10]]
    _text(slide, 0.6, 1.5, 12.1, 5.6, refs or ["No references recorded."], size=11)

    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)


# --- The agent ---

def run(report_result: dict) -> dict:
    """Build the slides from the Report Agent's facts and checked prose."""
    facts_path = Path((report_result or {}).get("facts_path") or "")
    if not facts_path.is_file():
        raise FileNotFoundError(f"The report's facts file {facts_path} does not exist; the slides need it.")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    prose = (report_result or {}).get("prose") or {}

    bullets, written_by, error = write_bullets(facts, prose)
    removed = []
    if bullets:
        bullets, removed = ground_bullets(bullets, facts)
    fallback = fallback_bullets(prose)
    used_fallback = [k for k in BULLET_KEYS if not bullets.get(k) and fallback.get(k)]
    for key in used_fallback:
        bullets[key] = fallback[key]

    path = PRESENTATIONS_DIR / facts_path.parent.name / "slides.pptx"
    build(facts, bullets, path)
    warnings = []
    if error:
        warnings.append(f"The bullets were not shortened by the LLM ({error}); the report's own sentences are used.")
    if not any(prose.values()):
        warnings.append("The report has no prose, so the narrative slides show only recorded facts.")
    return {"status": "DEGRADED" if not any(prose.values()) else "OK", "presentation_path": str(path),
            "facts_path": str(facts_path), "slides": list(SLIDES), "bullets": bullets, "written_by": written_by,
            "fallback_slides": used_fallback, "removed_claims": removed, "warnings": warnings}


def handoff(result: dict, context: dict) -> list:
    """The last stage: the Presentation Agent requests nothing more."""
    return []
