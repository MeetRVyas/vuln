"""Docker-based sandbox -- the primary implementation for deployed
execution (``configs/prod.yaml`` selects this backend).

Each call to :meth:`run` starts one throwaway, network-disabled container
per execution: the request's files are bind-mounted in, the command is run
as the container's entrypoint, and the container is removed afterwards
regardless of outcome. This is real isolation (a separate PID/mount/network
namespace and cgroup), unlike
:class:`~vulnrem.sandbox.subprocess_sandbox.SubprocessSandbox`.

Note on resource limits: Docker/cgroups do not expose a direct "total
CPU-seconds consumed" limit the way POSIX ``RLIMIT_CPU`` does. We cap the
CPU *rate* (``nano_cpus``) for fairness and rely on the wall-clock
``timeout_s`` (enforced via ``container.wait(timeout=...)``) as the
effective runaway-process bound, which is what actually matters for an
infinite-loop PoC or a hung test. Memory is a real hard limit
(``mem_limit``), and an OOM kill is detected and reported via
``resource_limit_exceeded``.

The concrete Docker client can be injected (``client=``) so the request/
response mapping and error handling can be unit-tested with a fake client,
without a real Docker daemon -- exactly the pattern used for the Google
provider (:class:`~vulnrem.llm.providers.google.GoogleGeminiProvider`).
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from pathlib import Path
from typing import Any

from vulnrem.core.enums import Language
from vulnrem.core.errors import SandboxUnavailableError
from vulnrem.sandbox.base import Sandbox
from vulnrem.sandbox.schemas import ExecutionRequest, ExecutionResult

_WORKSPACE_MOUNT = "/workspace"


class DockerSandbox(Sandbox):
    def __init__(self, image_by_language: dict[Language, str], client: Any | None = None):
        self._image_by_language = image_by_language
        self._injected_client = client

    @property
    def backend_name(self) -> str:
        return "docker"

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        try:
            import docker

            return docker.from_env()
        except Exception as exc:  # noqa: BLE001 - docker-py raises varied errors when no daemon
            raise SandboxUnavailableError(
                f"Docker daemon is not reachable: {exc}. Is Docker running and is this "
                "process allowed to talk to its socket? See docs/SETUP_DEPLOYMENT.md."
            ) from exc

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:  # noqa: BLE001
            return False

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        image = self._image_by_language.get(request.language)
        if image is None:
            raise SandboxUnavailableError(
                f"No Docker image configured for language={request.language.value!r}. "
                "Set sandbox.docker_images in configs/{env}.yaml."
            )
        client = self._get_client()

        with tempfile.TemporaryDirectory(prefix="vulnrem-docker-sandbox-") as workdir_str:
            workdir = Path(workdir_str)
            for relpath, content in request.files.items():
                target = workdir / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            container = client.containers.run(
                image=image,
                command=request.command,
                working_dir=_WORKSPACE_MOUNT,
                volumes={str(workdir): {"bind": _WORKSPACE_MOUNT, "mode": "rw"}},
                environment=request.env,
                network_disabled=request.network_disabled,
                mem_limit=f"{request.memory_limit_mb}m",
                nano_cpus=1_000_000_000,  # rate cap (~1 vCPU); see module docstring
                detach=True,
            )
            return self._await_and_collect(client, container, request)

    def _await_and_collect(self, client: Any, container: Any, request: ExecutionRequest) -> ExecutionResult:
        start = time.perf_counter()
        timed_out = False
        exit_code: int | None = None
        try:
            wait_result = container.wait(timeout=request.timeout_s)
            exit_code = wait_result.get("StatusCode") if isinstance(wait_result, dict) else None
        except Exception:  # noqa: BLE001 - docker-py raises requests.ReadTimeout et al. on timeout
            timed_out = True
            with contextlib.suppress(Exception):
                container.kill()
        duration = time.perf_counter() - start

        stdout = self._safe_logs(container, stdout=True, stderr=False)
        stderr = self._safe_logs(container, stdout=False, stderr=True)
        resource_limit_exceeded = self._was_oom_killed(client, container)

        with contextlib.suppress(Exception):
            container.remove(force=True)

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            resource_limit_exceeded=resource_limit_exceeded,
            duration_seconds=duration,
        )

    @staticmethod
    def _safe_logs(container: Any, *, stdout: bool, stderr: bool) -> str:
        try:
            raw = container.logs(stdout=stdout, stderr=stderr)
            return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _was_oom_killed(client: Any, container: Any) -> bool:
        try:
            state = client.api.inspect_container(container.id).get("State", {})
            return bool(state.get("OOMKilled", False))
        except Exception:  # noqa: BLE001
            return False
