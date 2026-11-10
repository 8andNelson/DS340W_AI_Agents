import subprocess
import sys
import textwrap

import pytest

from src.dashboard import freellmapi, server
from src.dashboard.freellmapi import FreeLLMAPIManager
from src.dashboard.runner import PipelineRunner


@pytest.fixture
def compose_dir(tmp_path, monkeypatch):
    folder = tmp_path / "freellmapi"
    folder.mkdir()
    (folder / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setenv("LLM_PROVIDER", "freellmapi")
    monkeypatch.setenv("FREELLMAPI_COMPOSE_DIR", str(folder))
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
    monkeypatch.setattr(freellmapi, "find_docker", lambda: "docker")
    monkeypatch.setattr(freellmapi.time, "sleep", lambda s: None)
    return folder


class FakeDocker:
    """Records docker commands; the router turns healthy after `compose up`."""

    def __init__(self, engine_up=True, up_returncode=0):
        self.engine_up = engine_up
        self.up_returncode = up_returncode
        self.calls = []
        self.router_up = False

    def install(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_docker", self.docker)
        monkeypatch.setattr(manager, "_engine_up", lambda: self.engine_up)
        monkeypatch.setattr(manager, "healthy", lambda use_cache=True: self.router_up)

    def docker(self, args, timeout, cwd=None):
        self.calls.append((args, cwd))
        if args[:2] == ["compose", "up"]:
            self.router_up = self.up_returncode == 0
            return subprocess.CompletedProcess(args, self.up_returncode, "", "pull access denied")
        if args[:2] == ["compose", "down"]:
            self.router_up = False
        return subprocess.CompletedProcess(args, 0, "", "")


def test_start_runs_compose_up_in_the_configured_folder(compose_dir, monkeypatch):
    manager, fake = FreeLLMAPIManager(), FakeDocker()
    fake.install(manager, monkeypatch)

    manager._start()
    assert fake.calls == [(["compose", "up", "-d"], str(compose_dir))]
    assert manager.state == "running"


def test_start_does_nothing_when_router_already_answers(compose_dir, monkeypatch):
    manager, fake = FreeLLMAPIManager(), FakeDocker()
    fake.install(manager, monkeypatch)
    fake.router_up = True

    manager._start()
    assert fake.calls == []
    assert manager.state == "running"


def test_start_launches_docker_desktop_when_engine_is_down(compose_dir, monkeypatch, tmp_path):
    manager, fake = FreeLLMAPIManager(), FakeDocker(engine_up=False)
    fake.install(manager, monkeypatch)
    desktop = tmp_path / "Docker Desktop.exe"
    monkeypatch.setattr(freellmapi, "find_docker_desktop", lambda: desktop)
    launched = []

    def fake_popen(args, **kwargs):
        launched.append(args)
        fake.engine_up = True  # the engine comes up after Docker Desktop starts

    monkeypatch.setattr(freellmapi.subprocess, "Popen", fake_popen)
    if not hasattr(freellmapi.subprocess, "DETACHED_PROCESS"):
        monkeypatch.setattr(freellmapi.subprocess, "DETACHED_PROCESS", 0, raising=False)
        monkeypatch.setattr(freellmapi.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, raising=False)

    manager._start()
    assert launched == [[str(desktop)]]
    assert fake.calls == [(["compose", "up", "-d"], str(compose_dir))]
    assert manager.state == "running"


def test_start_reports_compose_failure(compose_dir, monkeypatch):
    manager, fake = FreeLLMAPIManager(), FakeDocker(up_returncode=1)
    fake.install(manager, monkeypatch)

    manager._start()
    assert manager.state == "error"
    assert "pull access denied" in manager.message


def test_start_needs_a_compose_folder(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "freellmapi")
    monkeypatch.setenv("FREELLMAPI_COMPOSE_DIR", "")
    manager = FreeLLMAPIManager()
    monkeypatch.setattr(manager, "healthy", lambda use_cache=True: False)

    manager._start()
    assert manager.state == "error"
    assert "FREELLMAPI_COMPOSE_DIR" in manager.message


def test_stop_runs_compose_down(compose_dir, monkeypatch):
    manager, fake = FreeLLMAPIManager(), FakeDocker()
    fake.install(manager, monkeypatch)
    manager._start()

    manager._stop()
    assert fake.calls[-1] == (["compose", "down"], str(compose_dir))
    assert manager.state == "stopped"


def test_status_is_disabled_for_other_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert FreeLLMAPIManager().status()["enabled"] is False


def test_health_check_uses_the_router_ping(monkeypatch):
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
    urls = []

    class Ok:
        ok = True

    monkeypatch.setattr(freellmapi.requests, "get", lambda url, timeout: urls.append(url) or Ok())
    assert FreeLLMAPIManager().healthy(use_cache=False) is True
    assert urls == ["http://localhost:3001/api/ping"]


# --- Server rules ---

class StubRouter:
    def __init__(self, enabled=True, healthy=False):
        self._status = {"enabled": enabled, "healthy": healthy, "busy": False, "configured": True,
                        "compose_dir": "", "url": "", "state": "stopped", "message": "", "log": []}
        self.stopped = False

    def status(self):
        return self._status

    def start_async(self):
        return True

    def stop_async(self):
        self.stopped = True
        return True

    def stop_now(self):
        self.stopped = True


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUN_DB", tmp_path / "none.db")
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    script = tmp_path / "stand_in_run.py"
    script.write_text(textwrap.dedent("import time\ntime.sleep(30)\n"))
    runners = []

    def make(router):
        runner = PipelineRunner(console_log=tmp_path / "console.log", command_prefix=[sys.executable, str(script)])
        runners.append(runner)
        return server.create_app(runner, router).test_client()

    yield make
    for r in runners:
        if r.is_running():
            r.stop()


def test_run_is_refused_while_router_is_down(make_client):
    client = make_client(StubRouter(enabled=True, healthy=False))
    response = client.post("/api/run", json={"topic": "fraud"})
    assert response.status_code == 409
    assert "FreeLLMAPI router is not running" in response.get_json()["error"]


def test_run_starts_when_router_is_up_or_not_used(make_client):
    assert make_client(StubRouter(enabled=True, healthy=True)).post("/api/run", json={"topic": "a"}).status_code == 200
    assert make_client(StubRouter(enabled=False)).post("/api/run", json={"topic": "b"}).status_code == 200


def test_router_cannot_be_stopped_during_a_run(make_client):
    router = StubRouter(enabled=True, healthy=True)
    client = make_client(router)
    client.post("/api/run", json={"topic": "a"})
    assert client.post("/api/freellmapi/stop").status_code == 409
    assert router.stopped is False


def test_dashboard_exit_stops_the_router(monkeypatch):
    from src.dashboard import __main__ as dashboard_main
    monkeypatch.setattr(dashboard_main, "run_in_progress", lambda runner: False)
    router = StubRouter(enabled=True, healthy=True)
    dashboard_main.stop_router_on_exit(router, runner=None)
    assert router.stopped is True


def test_dashboard_exit_keeps_the_router_during_a_run(monkeypatch):
    from src.dashboard import __main__ as dashboard_main
    monkeypatch.setattr(dashboard_main, "run_in_progress", lambda runner: True)
    router = StubRouter(enabled=True, healthy=True)
    dashboard_main.stop_router_on_exit(router, runner=None)
    assert router.stopped is False


def test_dashboard_exit_ignores_the_router_for_other_providers(monkeypatch):
    from src.dashboard import __main__ as dashboard_main
    monkeypatch.setattr(dashboard_main, "run_in_progress", lambda runner: False)
    router = StubRouter(enabled=False)
    dashboard_main.stop_router_on_exit(router, runner=None)
    assert router.stopped is False


def test_stop_now_runs_compose_down(compose_dir, monkeypatch):
    manager, fake = FreeLLMAPIManager(), FakeDocker()
    fake.install(manager, monkeypatch)
    manager.stop_now()
    assert fake.calls[-1] == (["compose", "down"], str(compose_dir))
    assert manager.state == "stopped"
