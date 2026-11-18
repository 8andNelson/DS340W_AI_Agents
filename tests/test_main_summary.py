"""The closing summary of run.py states what the run did, not a fixed claim."""
from src.main import _final_summary


def base(**overrides):
    result = {"replication_result": {"source": "reconstructed", "status": "OK"},
              "evaluation_result": {"status": "APPROXIMATE"}, "reruns": [], "rerun_end_reason": "",
              "agents_off": [], "report_result": None, "presentation_result": None,
              "baseline_failed": False, "phase": "EXPERIMENTATION"}
    result.update(overrides)
    return result


def test_a_failed_baseline_is_never_called_a_reproduction():
    # Live run: every replication attempt failed, yet the summary said
    # "a baseline reproduction established".
    text = _final_summary(base(baseline_failed=True, replication_result={"source": "reconstructed",
                                                                          "status": "BLOCKED"}))
    assert "No working baseline" in text
    assert "established" not in text and "Baseline ran" not in text


def test_reruns_and_turned_off_agents_are_stated():
    text = _final_summary(base(reruns=[{"applied": True}, {"applied": False}],
                               rerun_end_reason="the Experiment Agent is turned off for this run",
                               agents_off=["Report Agent", "Presentation Agent"]))
    assert "Baseline ran (reconstructed) and was evaluated against the paper: APPROXIMATE." in text
    assert "2 rerun(s), 1 kept; the loop ended because the Experiment Agent is turned off" in text
    assert "Report: turned off" in text and "Slides: turned off" in text


def test_written_report_and_slides_are_named():
    text = _final_summary(base(report_result={"report_path": "reports/x/report.md"},
                               presentation_result={"presentation_path": "presentations/x/slides.pptx"},
                               phase="COMPLETE"))
    assert "Report: reports/x/report.md" in text and "Slides: presentations/x/slides.pptx" in text
    assert "Final phase: COMPLETE" in text
