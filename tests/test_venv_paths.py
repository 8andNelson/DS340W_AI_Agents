from pathlib import Path

from src.orchestration import venv_paths


def test_short_replication_folder_keeps_its_venv(tmp_path):
    folder = tmp_path / "r"
    if len(str((folder / "venv").resolve())) <= venv_paths.MAX_VENV_PATH_CHARS:
        assert venv_paths.venv_dir_for(folder, windows=True) == folder / "venv"


def test_long_replication_folder_on_windows_gets_a_short_stable_venv(tmp_path):
    folder = tmp_path / ("x" * 120)
    venv = venv_paths.venv_dir_for(folder, windows=True)
    assert venv.parent == venv_paths.short_venv_root()
    assert venv == venv_paths.venv_dir_for(folder, windows=True)
    assert venv != venv_paths.venv_dir_for(tmp_path / ("y" * 120), windows=True)


def test_other_systems_always_use_the_replication_folder(tmp_path):
    folder = tmp_path / ("x" * 120)
    assert venv_paths.venv_dir_for(folder, windows=False) == folder / "venv"


def test_path_too_long_detects_the_windows_error():
    assert venv_paths.path_too_long("FileNotFoundError: [WinError 206] The filename or extension is too long")
    assert not venv_paths.path_too_long("ModuleNotFoundError: No module named 'sklearn'")
    assert not venv_paths.path_too_long(None)


def test_slug_is_at_most_40_characters():
    from src.agents import replication_agent
    slug = replication_agent._slugify("An ensemble learning approach for anomaly detection in credit card data")
    assert len(slug) <= 40 and not slug.endswith("-")
