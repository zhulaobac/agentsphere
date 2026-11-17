"""Sandbox resource limits, filesystem mounts, and default-deny network policy."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
    cpu_seconds: float | None = 2.0
    memory_bytes: int | None = 256 * 1024 * 1024
    wall_time_seconds: float = 5.0
    max_processes: int | None = 32
    max_open_files: int | None = 64


class FilesystemMount(BaseModel):
    host_path: str
    sandbox_path: str
    read_only: bool = True


class NetworkPolicy(BaseModel):
    default_deny: bool = True
    allow_hosts: list[str] = Field(default_factory=list)


class SandboxSpec(BaseModel):
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    mounts: list[FilesystemMount] = Field(default_factory=list)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    env: dict[str, str] = Field(default_factory=dict)
    working_directory: str | None = None
    extra_readable_paths: list[str] = Field(default_factory=list)
    restrict_filesystem: bool = False


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    isolated_network: bool = False
    isolated_filesystem: bool = False
