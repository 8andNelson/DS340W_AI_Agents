"""Milestone 14: the Presentation Agent builds the slides only from the report's facts and checked prose."""
import json

import pytest
from pptx import Presentation

from src.agents import master_agent, presentation_agent, report_agent
from tests.test_report_agent import project, prose


@pytest.fixture
def report(monkeypatch):
    """A report written with checked prose (the LLM is mocked)."""
    text = prose(introduction="Fraud is rare and costly. Detecting it early limits losses. Models help.",
                 conclusion="The reproduction of [P1] is approximate.")
    monkeypatch.setattr(report_agent.llm, "chat", lambda *a, **k: text)
    return report_agent.run(project(), run_id="run-1")


def _bullets(monkeypatch, answer):
    def chat(*a, **k):
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)
    monkeypatch.setattr(presentation_agent.llm, "chat", chat)


def _shapes(path):
    return [s for slide in Presentation(path).slides for s in slide.shapes]


def test_the_deck_has_every_slide_and_the_recorded_values(monkeypatch, report):
    _bullets(monkeypatch, {"conclusions": ["The reproduced AUC is 0.9412 against 0.97 [P1]."]})
    result = presentation_agent.run(report)
    deck = Presentation(result["presentation_path"])
    assert len(deck.slides) == len(presentation_agent.SLIDES) == 16
    assert master_agent._check_presentation(result) == []
    shapes = _shapes(result["presentation_path"])
    table = next(s for s in shapes if s.name == presentation_agent.COMPARISON_TABLE).table
    assert [c.text for c in table.rows[1].cells] == ["auc", "0.97", "0.9412", "2.88", "APPROXIMATE"]
    chart = next(s for s in shapes if s.name == presentation_agent.RESULTS_CHART).chart
    assert [list(s.values) for s in chart.plots[0].series] == [[0.97], [0.9412]]
    assert result["bullets"]["conclusions"] == ["The reproduced AUC is 0.9412 against 0.97 [P1]."]
    notes = [slide.notes_slide.notes_text_frame.text for slide in deck.slides]
    assert all(n.startswith("Source:") for n in notes)


def test_a_bullet_with_an_invented_number_is_removed_and_the_report_text_used(monkeypatch, report):
    _bullets(monkeypatch, {"motivation": ["Fraud costs 32 billion dollars a year."]})
    result = presentation_agent.run(report)
    assert result["removed_claims"][0]["slide"] == "motivation"
    assert result["bullets"]["motivation"] == ["Detecting it early limits losses.", "Models help."]
    assert "motivation" in result["fallback_slides"]


def test_without_the_llm_the_report_sentences_are_used(monkeypatch, report):
    _bullets(monkeypatch, RuntimeError("router down"))
    result = presentation_agent.run(report)
    assert result["status"] == "OK" and "router down" in result["warnings"][0]
    assert result["bullets"]["conclusions"] == ["The reproduction of [P1] is approximate."]
    assert master_agent._check_presentation(result) == []


def test_a_report_without_prose_gives_a_degraded_deck_of_facts(monkeypatch):
    monkeypatch.setattr(report_agent.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    report = report_agent.run(project())
    result = presentation_agent.run(report)
    assert result["status"] == "DEGRADED" and master_agent._check_presentation(result) == []


def test_master_agent_rejects_a_deck_whose_values_differ_from_the_record(monkeypatch, report):
    _bullets(monkeypatch, RuntimeError("off"))
    result = presentation_agent.run(report)
    deck = Presentation(result["presentation_path"])
    table = next(s for slide in deck.slides for s in slide.shapes if s.name == presentation_agent.COMPARISON_TABLE)
    table.table.cell(1, 2).text = "0.97"  # a reproduced value moved to the published one
    deck.save(result["presentation_path"])
    issue = master_agent._check_presentation(result)[0]["issue"]
    assert issue.startswith("The comparison table shows") and "0.9412" in issue


def test_master_agent_rejects_a_missing_slide(monkeypatch, report):
    _bullets(monkeypatch, RuntimeError("off"))
    result = presentation_agent.run(report)
    deck = Presentation(result["presentation_path"])
    slides = deck.slides._sldIdLst
    slides.remove(slides[2])  # drop "Motivation"
    deck.save(result["presentation_path"])
    assert "not the required structure" in master_agent._check_presentation(result)[0]["issue"]


def test_the_slides_need_the_reports_facts(tmp_path):
    with pytest.raises(FileNotFoundError):
        presentation_agent.run({"facts_path": str(tmp_path / "none.json")})


def test_the_report_hands_off_to_the_presentation_agent(report):
    (req,) = report_agent.handoff(report, {})
    assert req["target_agent"] == "Presentation Agent" and req["inputs"]["report_result"] is report
    assert presentation_agent.handoff({}, {}) == []


def test_approved_slides_are_never_turned_into_a_memory():
    from src.memory import retain
    from src.orchestration import run_store
    run_id = run_store.start_run("fraud")
    run_store.record_stage(run_id, "Presentation Agent", 1, "Approved", {"status": "OK"})
    memories, _ = retain.build_memories(retain.load_run(run_id))
    assert not [m for m in memories if m["agent"] == "Presentation Agent"]
