"""CI-testable sandbox backend: isolated subprocess with rlimits and default-deny network.

Firecracker / gVisor / Kata are production backends for the same Sandbox protocol.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from agentsphere.sandbox.capabilities import SandboxCapabilities, probe
from agentsphere.sandbox.limits import SandboxResult, SandboxSpec
from agentsphere.sandbox.linux import isolation_preexec


class Sandbox(Protocol):
    async def run(
        self,
        argv: Sequence[str],
        spec: SandboxSpec,
        *,
        stdin_data: bytes | None = None,
        extra_fds: Mapping[int, int] | None = None,
    ) -> SandboxResult: ...

    def capabilities(self) -> SandboxCapabilities: ...


class SubprocessSandbox:
    """Isolated subprocess. Secrets must never be placed in ``spec.env``."""

    def __init__(self) -> None:
        self._capabilities = probe()

    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities

    async def run(
        self,
        argv: Sequence[str],
        spec: SandboxSpec,
        *,
        stdin_data: bytes | None = None,
        extra_fds: Mapping[int, int] | None = None,
    ) -> SandboxResult:
        if not argv:
            raise ValueError("sandbox command must not be empty")
        cwd = spec.working_directory
        workspace = Path(cwd) if cwd else None
        env = _sanitize_env(spec.env)
        linux = sys.platform == "linux"
        status_r = status_w = -1
        fd_map = dict(extra_fds or {})
        placeholders: list[int] = []
        if linux:
            # Occupy destination FDs (especially 3) in the parent so Popen's
            # errpipe and the isolation status pipe cannot land on them. After
            # fork, CPython closes errpipe_write; if that were FD 3 it would
            # drop the secrets pipe the worker expects.
            for dest in fd_map:
                occupied = _occupy_fd(dest)
                if occupied is not None:
                    placeholders.append(occupied)
            status_r, status_w = os.pipe()
            os.set_inheritable(status_r, False)
            os.set_inheritable(status_w, True)
            for src in fd_map.values():
                try:
                    os.set_inheritable(src, True)
                except OSError:
                    pass
            for dest in fd_map:
                try:
                    os.set_inheritable(dest, True)
                except OSError:
                    pass

        def preexec() -> None:
            dests = set(fd_map)
            if status_r >= 0 and status_r not in dests:
                try:
                    os.close(status_r)
                except OSError:
                    pass
            status_write = status_w
            if status_w >= 0 and status_w in dests:
                status_write = os.dup(status_w)
                try:
                    os.set_inheritable(status_write, True)
                except OSError:
                    pass
            for dest, src in fd_map.items():
                if src != dest:
                    os.dup2(src, dest)
                try:
                    os.set_inheritable(dest, True)
                except OSError:
                    pass
            for dest, src in fd_map.items():
                if src != dest and src not in dests and src != status_write:
                    try:
                        os.close(src)
                    except OSError:
                        pass
            net, fs = isolation_preexec(spec, workspace=workspace)
            if status_write >= 0:
                try:
                    os.write(status_write, bytes((1 if net else 0, 1 if fs else 0)))
                finally:
                    try:
                        os.close(status_write)
                    except OSError:
                        pass
                    if status_w >= 0 and status_w != status_write and status_w not in dests:
                        try:
                            os.close(status_w)
                        except OSError:
                            pass

        inherited = set()
        if linux:
            inherited.add(status_w)
        if fd_map:
            inherited.update(fd_map.values())
            inherited.update(fd_map.keys())
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                preexec_fn=preexec if linux else None,
                pass_fds=tuple(fd for fd in inherited if fd >= 0),
            )
        except OSError as exc:
            if status_r >= 0:
                os.close(status_r)
            if status_w >= 0:
                os.close(status_w)
            for fd in placeholders:
                try:
                    os.close(fd)
                except OSError:
                    pass
            return SandboxResult(exit_code=127, stderr=str(exc), timed_out=False)
        for fd in placeholders:
            try:
                os.close(fd)
            except OSError:
                pass
        isolated_network = False
        isolated_filesystem = False
        if linux:
            os.close(status_w)
            try:
                status = os.read(status_r, 2)
            except OSError:
                status = b""
            finally:
                os.close(status_r)
            isolated_network = len(status) >= 1 and status[0] == 1
            isolated_filesystem = len(status) >= 2 and status[1] == 1

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(stdin_data),
                timeout=spec.limits.wall_time_seconds,
            )
        except TimeoutError:
            timed_out = True
            process.kill()
            try:
                stdout_b, stderr_b = await process.communicate()
            except Exception:
                stdout_b, stderr_b = b"", b""
        exit_code = process.returncode if process.returncode is not None else -1
        if timed_out:
            exit_code = -9
        return SandboxResult(
            exit_code=exit_code,
            stdout=_decode(stdout_b),
            stderr=_decode(stderr_b),
            timed_out=timed_out,
            isolated_network=isolated_network,
            isolated_filesystem=isolated_filesystem,
        )


def _occupy_fd(dest: int) -> int | None:
    """If ``dest`` is closed, occupy it so later ``os.pipe()`` calls skip it.

    Returns the fd the caller should close in the parent after spawn, or
    ``None`` if ``dest`` was already open.
    """
    try:
        os.fstat(dest)
        return None
    except OSError:
        pass
    fd = os.open("/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    if fd == dest:
        try:
            os.set_inheritable(dest, True)
        except OSError:
            pass
        return dest
    try:
        os.dup2(fd, dest)
        os.set_inheritable(dest, True)
    finally:
        os.close(fd)
    return dest


def _sanitize_env(extra: Mapping[str, str]) -> dict[str, str]:
    """Minimal env for the child. Never copies the parent environment (where secrets might live)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    else:
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    virtual = os.environ.get("VIRTUAL_ENV")
    if virtual:
        env["VIRTUAL_ENV"] = virtual
    sys_prefix_bin = str(Path(sys.prefix) / "bin")
    if sys_prefix_bin not in env["PATH"]:
        env["PATH"] = f"{sys_prefix_bin}:{env['PATH']}"
    for key, value in extra.items():
        env[str(key)] = str(value)
    return env


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")
