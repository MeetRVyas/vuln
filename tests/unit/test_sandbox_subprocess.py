from __future__ import annotations

from vulnrem.core.enums import Language
from vulnrem.sandbox.schemas import ExecutionRequest
from vulnrem.sandbox.subprocess_sandbox import SubprocessSandbox


def _sandbox() -> SubprocessSandbox:
    return SubprocessSandbox()


def test_backend_name():
    assert _sandbox().backend_name == "subprocess"


def test_health_check_is_always_true():
    assert _sandbox().health_check() is True


def test_runs_a_simple_python_script_successfully():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "print('hello from sandbox')"},
        command=["python3", "main.py"],
        timeout_s=10,
    )
    result = _sandbox().run(request)

    assert result.succeeded
    assert result.exit_code == 0
    assert "hello from sandbox" in result.stdout
    assert result.duration_seconds >= 0


def test_captures_non_zero_exit_code():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "import sys; sys.exit(3)"},
        command=["python3", "main.py"],
        timeout_s=10,
    )
    result = _sandbox().run(request)

    assert not result.succeeded
    assert result.exit_code == 3
    assert result.timed_out is False


def test_captures_stderr_and_traceback():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "raise ValueError('boom')"},
        command=["python3", "main.py"],
        timeout_s=10,
    )
    result = _sandbox().run(request)

    assert result.exit_code == 1
    assert "ValueError: boom" in result.stderr


def test_writes_multiple_files_into_same_workspace():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={
            "helper.py": "def double(x):\n    return x * 2\n",
            "main.py": "from helper import double\nprint(double(21))\n",
        },
        command=["python3", "main.py"],
        timeout_s=10,
    )
    result = _sandbox().run(request)

    assert result.succeeded
    assert "42" in result.stdout


def test_wall_clock_timeout_is_reported():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "import time\ntime.sleep(30)\n"},
        command=["python3", "main.py"],
        timeout_s=1,
        cpu_limit_seconds=30,  # isolate: we want the *wall clock* timeout to fire, not the CPU limit
    )
    result = _sandbox().run(request)

    assert result.timed_out is True
    assert not result.succeeded


def test_cpu_limit_or_timeout_kills_a_busy_loop():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "x = 0\nwhile True:\n    x += 1\n"},
        command=["python3", "main.py"],
        timeout_s=5,
        cpu_limit_seconds=1,
    )
    result = _sandbox().run(request)

    assert not result.succeeded
    assert result.timed_out or result.resource_limit_exceeded


def test_network_env_vars_are_scrubbed_when_network_disabled(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:8080")
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "import os\nprint(repr(os.environ.get('HTTPS_PROXY')))"},
        command=["python3", "main.py"],
        timeout_s=10,
        network_disabled=True,
    )
    result = _sandbox().run(request)

    assert result.succeeded
    assert "None" in result.stdout


def test_explicit_env_vars_are_passed_through():
    request = ExecutionRequest(
        language=Language.PYTHON,
        files={"main.py": "import os\nprint(os.environ.get('MY_FLAG'))"},
        command=["python3", "main.py"],
        timeout_s=10,
        env={"MY_FLAG": "enabled"},
    )
    result = _sandbox().run(request)

    assert "enabled" in result.stdout
