"""A run deletes the reconstructed baselines' venvs it made when it ends
(success, halted or crashed): each is several hundred MB, and
pip_freeze.txt keeps the packages."""
from unittest.mock import patch

import pytest

from src.agents import master_agent, replication_agent


@pytest.fixture(autouse=True)
def _clean_registry():
    replication_agent._RUN_VENVS.clear()
    yield
    replication_agent._RUN_VENVS.clear()


def make_venv(path):
    (path / "Lib").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("home = x")
    return path


def test_only_this_runs_venvs_are_deleted(tmp_path):
    made = make_venv(tmp_path / "a" / "venv")
    other = make_venv(tmp_path / "b" / "venv")
    replication_agent._RUN_VENVS.add(made)
    assert replication_agent.remove_run_venvs() == [str(made)]
    assert not made.exists() and other.exists()
    assert replication_agent._RUN_VENVS == set()


def test_a_folder_without_pyvenv_cfg_is_never_deleted(tmp_path):
    folder = tmp_path / "not_a_venv"
    folder.mkdir()
    (folder / "keep.txt").write_text("x")
    replication_agent._RUN_VENVS.add(folder)
    assert replication_agent.remove_run_venvs() == []
    assert (folder / "keep.txt").exists()


def test_create_venv_registers_the_folder(tmp_path):
    with patch.object(replication_agent, "_run_bounded", return_value={"exit_code": 0}):
        replication_agent._create_venv(tmp_path / "v")
    assert tmp_path / "v" in replication_agent._RUN_VENVS


def test_the_run_deletes_its_venvs_when_it_finishes(tmp_path):
    venv = make_venv(tmp_path / "venv")

    def stages(_topic):
        replication_agent._RUN_VENVS.add(venv)
        return {"success": False, "phase": "TOPIC_ANALYSIS", "parent_paper": None}

    with patch.object(master_agent, "_run_stages", side_effect=stages):
        master_agent.run_pipeline("topic")
    assert not venv.exists()


def test_a_crashed_run_also_deletes_its_venvs(tmp_path):
    venv = make_venv(tmp_path / "venv")

    def crash(_topic):
        replication_agent._RUN_VENVS.add(venv)
        raise RuntimeError("stage crashed")

    with patch.object(master_agent, "_run_stages", side_effect=crash), pytest.raises(RuntimeError):
        master_agent.run_pipeline("topic")
    assert not venv.exists()
