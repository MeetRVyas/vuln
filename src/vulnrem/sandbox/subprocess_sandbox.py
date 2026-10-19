"""Local-development fallback sandbox.

Runs the requested command as a plain OS subprocess, inside a throwaway
temp directory, under best-effort CPU-time and address-space limits
(``resource.setrlimit`` via ``preexec_fn``) and a wall-clock timeout.

This is **not** a real security boundary: ``preexec_fn``-based rlimits do
not stop a process from reading the rest of the host filesystem, and
``network_disabled`` here only scrubs proxy-related environment variables
-- it does not actually block socket connections. It exists purely so the
pipeline is fully runnable and testable on a laptop with no Docker daemon
(exactly this build environment). Real isolation is
:class:`~vulnrem.sandbox.docker_sandbox.DockerSandbox`, which is what
``configs/prod.yaml`` selects.
"""

from __future__ import annotations

import contextlib
import resource
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from vulnrem.sandbox.base import Sandbox
from vulnrem.sandbox.schemas import ExecutionRequest, ExecutionResult

_RESOURCE_KILL_SIGNALS = {signal.SIGXCPU, signal.SIGKILL, signal.SIGSEGV}

# Environment variables that would let a "network disabled" run still reach
# the network via an ambient proxy configuration.
_NETWORK_ENV_VARS_TO_SCRUB = {
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
}

_MINIMAL_PASSTHROUGH_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH")


class SubprocessSandbox(Sandbox):
    @property
    def backend_name(self) -> str:
        return "subprocess"

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="vulnrem-sandbox-") as workdir_str:
            workdir = Path(workdir_str)
            for relpath, content in request.files.items():
                target = workdir / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            env = self._build_env(request)
            preexec_fn = self._build_preexec_fn(request) if hasattr(resource, "setrlimit") else None

            start = time.perf_counter()
            timed_out = False
            try:
                completed = subprocess.run(  # noqa: S603 - args are argv lists, not shell strings
                    request.command,
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    preexec_fn=preexec_fn,
                )
                exit_code = completed.returncode
                stdout, stderr = completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = None
                stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            duration = time.perf_counter() - start

            resource_limit_exceeded = (
                exit_code is not None and exit_code < 0 and abs(exit_code) in _RESOURCE_KILL_SIGNALS
            )

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                resource_limit_exceeded=resource_limit_exceeded,
                duration_seconds=duration,
            )

    def health_check(self) -> bool:
        return True  # always available -- it's just the local Python process

    @staticmethod
    def _build_env(request: ExecutionRequest) -> dict[str, str]:
        import os

        base_env = {k: v for k, v in os.environ.items() if k in _MINIMAL_PASSTHROUGH_ENV_VARS}
        if request.network_disabled:
            for var in _NETWORK_ENV_VARS_TO_SCRUB:
                base_env.pop(var, None)
        base_env.update(request.env)
        return base_env

    @staticmethod
    def _build_preexec_fn(request: ExecutionRequest):
        cpu_seconds = int(request.cpu_limit_seconds)
        memory_bytes = request.memory_limit_mb * 1024 * 1024

        def _apply_limits() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            with contextlib.suppress(ValueError, OSError):
                # Some platforms/containers don't allow RLIMIT_AS; degrade gracefully.
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        return _apply_limits
