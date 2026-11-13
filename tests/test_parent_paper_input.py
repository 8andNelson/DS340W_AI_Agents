import io
import json

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from src import config, paper_input
from src.agents import master_agent
from src.orchestration import run_metrics, run_store
from src.search import paper_fulltext

TITLE = "Machine Learning for Credit Card Fraud"
TOPIC = "machine learning for detecting credit card fraud"


def paper_pdf(extra=""):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    body = "We classify fraud in a fixed dataset using a train test split and report measured results. " * 40
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 750 Td ({TITLE}) Tj 0 -15 Td ({body}{extra}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def saved_input(tmp_path, extra="", target="fraud"):
    return paper_input.save_input(tmp_path / "inputs", paper_pdf(extra), TOPIC, TITLE, target)


def test_repeated_input_has_stable_identity_and_preserves_pdf(tmp_path):
    first, manifest = saved_input(tmp_path)
    second, repeated = saved_input(tmp_path)
    assert first == second and manifest == repeated
    loaded, pdf = paper_input.load_input(first, TOPIC)
    assert loaded == manifest and pdf.read_bytes() == paper_pdf()
    _, different_target = saved_input(tmp_path, target="anomaly")
    _, different_pdf = saved_input(tmp_path, extra="Revision two.")
    assert len({manifest["input_fingerprint"], different_target["input_fingerprint"],
                different_pdf["input_fingerprint"]}) == 3


def test_invalid_unreadable_and_mismatched_pdfs_are_rejected(tmp_path):
    with pytest.raises(paper_input.PaperInputError, match="PDF"):
        paper_input.save_input(tmp_path, b"not a PDF", TOPIC, TITLE, "fraud")
    blank = PdfWriter()
    blank.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    blank.write(output)
    with pytest.raises(paper_input.PaperInputError, match="readable"):
        paper_input.save_input(tmp_path, output.getvalue(), TOPIC, TITLE, "fraud")
    with pytest.raises(paper_input.PaperInputError, match="title was not found"):
        paper_input.save_input(tmp_path, paper_pdf(), TOPIC, "A Different Paper", "fraud")


def test_pinned_pdf_never_falls_back_to_search_or_another_cached_paper(tmp_path, monkeypatch):
    path, manifest = saved_input(tmp_path)
    ctx = master_agent._supplied_parent_context(TOPIC, path)
    parent = ctx["parent_paper"]
    def unexpected(*args, **kwargs):
        pytest.fail("A supplied PDF must not trigger network discovery")
    monkeypatch.setattr(paper_fulltext, "candidate_pdf_urls", unexpected)
    result = paper_fulltext.get_parent_paper_text(parent)
    assert result["status"] == "FULL_TEXT" and TITLE in result["text"]
    assert result["pdf_sha256"] == manifest["pdf_sha256"]
    _, pdf = paper_input.load_input(path)
    pdf.write_bytes(b"different bytes")
    with pytest.raises(paper_input.PaperInputError, match="checksum"):
        paper_fulltext.get_parent_paper_text(parent)


def test_settings_and_topic_cannot_change_under_a_fixed_input(tmp_path):
    path, manifest = saved_input(tmp_path)
    with pytest.raises(paper_input.PaperInputError, match="topic differs"):
        paper_input.load_input(path, "another topic")
    manifest["target_terms"] = ["different"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(paper_input.PaperInputError, match="settings changed"):
        paper_input.load_input(path)


def test_fixed_paper_run_bypasses_discovery_and_preserves_target(tmp_path, monkeypatch):
    from src.agents import (intake_agent, research_agent, validation_agent, data_agent, cleaning_agent,
                            code_discovery_agent, replication_agent, evaluation_agent, experiment_agent)
    from tests.test_master_agent import (make_data_result, make_dataset_candidate, make_cleaning_result,
                                        make_code_discovery_result, make_replication_result,
                                        make_evaluation_result, make_experiment_result)
    path, manifest = saved_input(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Intake, Research, and paper selection must be skipped")
    for module in (intake_agent, research_agent, validation_agent):
        monkeypatch.setattr(module, "run", forbidden)
    def data(parent, pool, **kwargs):
        assert parent["pdf_sha256"] == manifest["pdf_sha256"]
        assert parent["peer_reviewed"] is None
        assert kwargs["target_terms"] == ["fraud"]
        assert paper_fulltext.get_parent_paper_text(parent)["status"] == "FULL_TEXT"
        return make_data_result(selected_datasets=[make_dataset_candidate(entry_count=12000)])
    monkeypatch.setattr(data_agent, "run", data)
    for module, output in ((cleaning_agent, make_cleaning_result()),
                           (code_discovery_agent, make_code_discovery_result()),
                           (replication_agent, make_replication_result()),
                           (evaluation_agent, make_evaluation_result()),
                           (experiment_agent, make_experiment_result())):
        monkeypatch.setattr(module, "run", lambda *a, _output=output, **kw: _output)
    result = master_agent.run_pipeline(TOPIC, parent_paper_input=path)
    assert result["success"] and result["input_mode"] == "parent_pdf"
    assert result["input_fingerprint"] == manifest["input_fingerprint"]
    with run_store._connect() as conn:
        stages = conn.execute("SELECT agent FROM stages WHERE run_id=?", (result["run_id"],)).fetchall()
        artifacts = conn.execute("SELECT original_path FROM artifacts WHERE run_id=?", (result["run_id"],)).fetchall()
    assert "Parent Paper Intake" in [s["agent"] for s in stages]
    assert not {"Intake Agent", "Research Agent", "Validation Agent"}.intersection(s["agent"] for s in stages)
    assert any(a["original_path"].endswith(".pdf") for a in artifacts)


def test_metrics_group_by_exact_input_across_memory_modes(tmp_path, monkeypatch):
    path, manifest = saved_input(tmp_path)
    conditions = []
    for mode in ("off", "read_write"):
        monkeypatch.setattr(config, "MEMORY_MODE", mode)  # read when the run starts
        run_id = run_store.start_run(TOPIC)
        ctx = master_agent._supplied_parent_context(TOPIC, path)
        # Same archived intake record used for a crash before a final result exists.
        run_store.record_stage(run_id, "Parent Paper Intake", 1, "Approved", {
            "parent_paper": ctx["parent_paper"], "input_fingerprint": manifest["input_fingerprint"]})
        run_store.finish_run(run_id, "crashed")
        conditions.append(next(r["conditions"] for r in run_metrics.load_runs()
                               if r["conditions"]["run_id"] == run_id))
    assert conditions[0]["comparison_group"] == conditions[1]["comparison_group"] == f"parent_pdf:{manifest['input_fingerprint']}"
    assert conditions[0]["parent_paper_title"] == TITLE
    assert run_metrics._task(conditions[0]) == run_metrics._task(conditions[1])
    assert [c["memory_mode"] for c in conditions] == ["off", "read_write"]
