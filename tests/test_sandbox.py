from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentsphere.sandbox import ResourceLimits, SandboxSpec, SubprocessSandbox, probe


@pytest.mark.asyncio
async def test_sandbox_wall_time_limit() -> None:
    sandbox = SubprocessSandbox()
    spec = SandboxSpec(limits=ResourceLimits(wall_time_seconds=0.2, cpu_seconds=1, memory_bytes=256 * 1024 * 1024))
    result = await sandbox.run([sys.executable, "-c", "import time; time.sleep(5)"], spec)
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_sandbox_env_does_not_inherit_parent_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_SECRET", "super-secret-value-xyz")
    sandbox = SubprocessSandbox()
    spec = SandboxSpec(env={"PUBLIC": "ok"})
    result = await sandbox.run(
        [sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        spec,
    )
    assert result.exit_code == 0
    assert "PUBLIC" in result.stdout
    assert "ok" in result.stdout
    assert "super-secret-value-xyz" not in result.stdout
    assert "VAULT_SECRET" not in result.stdout


@pytest.mark.asyncio
async def test_sandbox_default_deny_network() -> None:
    caps = probe()
    sandbox = SubprocessSandbox()
    spec = SandboxSpec()
    result = await sandbox.run(
        [
            sys.executable,
            "-c",
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
            "    print('CONNECTED')\n"
            "except Exception:\n"
            "    print('BLOCKED')\n",
        ],
        spec,
    )
    if not result.isolated_network:
        pytest.skip(
            "network namespace isolation is unavailable: "
            f"user_ns={caps.user_namespace} net_ns={caps.net_namespace} reason={caps.reason or 'unshare failed at runtime'}"
        )
    assert "BLOCKED" in result.stdout
    assert "CONNECTED" not in result.stdout


@pytest.mark.asyncio
async def test_sandbox_workspace_rw_other_paths_not_visible(tmp_path: Path) -> None:
    caps = probe()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("workspace-ok")
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("host-secret")
    sandbox = SubprocessSandbox()
    spec = SandboxSpec(
        working_directory=str(workspace),
        restrict_filesystem=True,
        extra_readable_paths=[sys.prefix, str(Path(sys.executable).resolve().parent)],
        limits=ResourceLimits(wall_time_seconds=8),
    )
    result = await sandbox.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path\n"
            "print(Path('note.txt').read_text())\n"
            "try:\n"
            f"    print(Path({str(secret)!r}).read_text())\n"
            "except Exception:\n"
            "    print('HIDDEN')\n",
        ],
        spec,
    )
    if not result.isolated_filesystem:
        pytest.skip(
            "filesystem isolation is unavailable: "
            f"landlock={caps.landlock} user_ns={caps.user_namespace} mount_ns={caps.mount_namespace} "
            f"reason={caps.reason or 'landlock/chroot failed at runtime'}"
        )
    assert "workspace-ok" in result.stdout
    assert "host-secret" not in result.stdout
    assert "HIDDEN" in result.stdout
