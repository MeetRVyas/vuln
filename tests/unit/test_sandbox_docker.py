from __future__ import annotations

import pytest

from vulnrem.core.enums import Language
from vulnrem.core.errors import SandboxUnavailableError
from vulnrem.sandbox.docker_sandbox import DockerSandbox
from vulnrem.sandbox.schemas import ExecutionRequest


class _FakeContainer:
    def __init__(self, exit_code=0, raise_on_wait=None, oom=False, stdout=b"", stderr=b""):
        self._exit_code = exit_code
        self._raise_on_wait = raise_on_wait
        self._oom = oom
        self._stdout = stdout
        self._stderr = stderr
        self.id = "fake-container-id"
        self.killed = False
        self.removed = False

    def wait(self, timeout=None):
        if self._raise_on_wait:
            raise self._raise_on_wait
        return {"StatusCode": self._exit_code}

    def kill(self):
        self.killed = True

    def logs(self, stdout=True, stderr=True):
        if stdout:
            return self._stdout
        if stderr:
            return self._stderr
        return b""

    def remove(self, force=True):
        self.removed = True


class _FakeContainersAPI:
    def __init__(self, container: _FakeContainer):
        self._container = container
        self.run_kwargs: dict | None = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._container


class _FakeLowLevelAPI:
    def __init__(self, oom: bool):
        self._oom = oom

    def inspect_container(self, container_id):
        return {"State": {"OOMKilled": self._oom}}


class _FakeDockerClient:
    def __init__(self, container: _FakeContainer, oom: bool = False, ping_ok: bool = True):
        self.containers = _FakeContainersAPI(container)
        self.api = _FakeLowLevelAPI(oom)
        self._ping_ok = ping_ok

    def ping(self):
        if not self._ping_ok:
            raise RuntimeError("no daemon")
        return True


def _request(**overrides) -> ExecutionRequest:
    defaults = dict(
        language=Language.PYTHON,
        files={"main.py": "print('hi')"},
        command=["python3", "main.py"],
        timeout_s=5,
    )
    defaults.update(overrides)
    return ExecutionRequest(**defaults)


def test_backend_name():
    sandbox = DockerSandbox(image_by_language={}, client=_FakeDockerClient(_FakeContainer()))
    assert sandbox.backend_name == "docker"


def test_run_returns_success_result():
    container = _FakeContainer(exit_code=0, stdout=b"hi\n", stderr=b"")
    client = _FakeDockerClient(container)
    sandbox = DockerSandbox(image_by_language={Language.PYTHON: "vulnrem-sandbox-python:latest"}, client=client)

    result = sandbox.run(_request())

    assert result.succeeded
    assert result.stdout == "hi\n"
    assert container.removed is True
    assert client.containers.run_kwargs["network_disabled"] is True
    assert client.containers.run_kwargs["image"] == "vulnrem-sandbox-python:latest"


def test_run_reports_non_zero_exit():
    container = _FakeContainer(exit_code=1, stderr=b"Traceback...\n")
    client = _FakeDockerClient(container)
    sandbox = DockerSandbox(image_by_language={Language.PYTHON: "img"}, client=client)

    result = sandbox.run(_request())

    assert not result.succeeded
    assert result.exit_code == 1
    assert "Traceback" in result.stderr


def test_run_detects_timeout_and_kills_container():
    container = _FakeContainer(raise_on_wait=TimeoutError("wait timed out"))
    client = _FakeDockerClient(container)
    sandbox = DockerSandbox(image_by_language={Language.PYTHON: "img"}, client=client)

    result = sandbox.run(_request())

    assert result.timed_out is True
    assert container.killed is True
    assert container.removed is True


def test_run_detects_oom_kill():
    container = _FakeContainer(exit_code=137)
    client = _FakeDockerClient(container, oom=True)
    sandbox = DockerSandbox(image_by_language={Language.PYTHON: "img"}, client=client)

    result = sandbox.run(_request())

    assert result.resource_limit_exceeded is True


def test_run_raises_when_language_has_no_configured_image():
    client = _FakeDockerClient(_FakeContainer())
    sandbox = DockerSandbox(image_by_language={}, client=client)

    with pytest.raises(SandboxUnavailableError):
        sandbox.run(_request())


def test_health_check_true_when_ping_succeeds():
    client = _FakeDockerClient(_FakeContainer(), ping_ok=True)
    sandbox = DockerSandbox(image_by_language={}, client=client)
    assert sandbox.health_check() is True


def test_health_check_false_when_ping_fails():
    client = _FakeDockerClient(_FakeContainer(), ping_ok=False)
    sandbox = DockerSandbox(image_by_language={}, client=client)
    assert sandbox.health_check() is False


def test_no_daemon_reachable_raises_sandbox_unavailable(monkeypatch):
    sandbox = DockerSandbox(image_by_language={Language.PYTHON: "img"})  # no client injected

    def _boom():
        raise RuntimeError("Docker socket not found")

    import docker

    monkeypatch.setattr(docker, "from_env", _boom)

    with pytest.raises(SandboxUnavailableError):
        sandbox.run(_request())
