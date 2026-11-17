"""Control Plane talks to a Runtime Provider. The existing Managed runtime is the first provider.

This is not a second runtime: it records deployments and delegates Session/Turn/Event
operations to ``agentsphere.runtime.interfaces.RuntimeProvider``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.context import SessionState
from agentsphere.kernel.events import Event
from agentsphere.runtime.checkpoint import Checkpoint
from agentsphere.runtime.entities import Session, Turn
from agentsphere.runtime.interfaces import RuntimeProvider
from agentsphere.runtime.lease import WorkerLease
from agentsphere.controlplane.release import DeploymentRecord
from agentsphere.controlplane.types import ReleaseChannel


class DeploymentBinding(BaseModel):
    deployment_id: str
    agent_name: str
    agent_version: str
    tenant: str
    channel: ReleaseChannel = ReleaseChannel.CANARY
    weight: float = 0.05
    system_prompt: str = ""
    tool_pins: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlRuntimeProvider(Protocol):
    """What the Control Plane binds releases to. Implemented by ManagedRuntimeProvider."""

    async def bind_deployment(self, binding: DeploymentBinding) -> None: ...

    async def unbind_deployment(self, deployment_id: str) -> None: ...

    def get_binding(self, deployment_id: str) -> DeploymentBinding | None: ...

    async def create_session(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        tool_pins: dict[str, str] | None = None,
        deployment_id: str | None = None,
    ) -> Session: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    def run_turn(self, session_id: str, user_input: str) -> AsyncIterator[Event]: ...

    def resume(self, session_id: str, *, worker_id: str | None = None) -> AsyncIterator[Event]: ...

    async def list_events(self, session_id: str) -> list[Event]: ...

    async def list_turns(self, session_id: str) -> list[Turn]: ...

    async def rebuild_state(self, session_id: str) -> SessionState: ...

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None: ...

    async def acquire_lease(self, session_id: str, worker_id: str) -> WorkerLease: ...


class ManagedRuntimeProvider:
    """First Runtime Provider: the in-tree Managed Runtime (SessionRuntime)."""

    def __init__(self, runtime: RuntimeProvider) -> None:
        self.runtime = runtime
        self._bindings: dict[str, DeploymentBinding] = {}

    async def bind_deployment(self, binding: DeploymentBinding) -> None:
        self._bindings[binding.deployment_id] = binding.model_copy(deep=True)

    async def unbind_deployment(self, deployment_id: str) -> None:
        self._bindings.pop(deployment_id, None)

    def get_binding(self, deployment_id: str) -> DeploymentBinding | None:
        item = self._bindings.get(deployment_id)
        return item.model_copy(deep=True) if item else None

    def bind_from_record(self, record: DeploymentRecord, *, system_prompt: str = "") -> DeploymentBinding:
        binding = DeploymentBinding(
            deployment_id=record.deployment_id,
            agent_name=record.agent_name,
            agent_version=record.agent_version,
            tenant=record.tenant,
            channel=record.channel,
            weight=record.weight,
            system_prompt=system_prompt,
            metadata=dict(record.metadata),
        )
        self._bindings[binding.deployment_id] = binding
        return binding.model_copy(deep=True)

    async def create_session(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        tool_pins: dict[str, str] | None = None,
        deployment_id: str | None = None,
    ) -> Session:
        meta = dict(metadata or {})
        prompt = system_prompt
        pins = tool_pins
        if deployment_id is not None:
            binding = self._bindings.get(deployment_id)
            if binding is None:
                raise KeyError(f"deployment not bound: {deployment_id}")
            meta.setdefault("deployment_id", binding.deployment_id)
            meta.setdefault("agent_name", binding.agent_name)
            meta.setdefault("agent_version", binding.agent_version)
            meta.setdefault("release_channel", binding.channel.value)
            meta.setdefault("tenant", binding.tenant)
            prompt = prompt or binding.system_prompt
            pins = pins if pins is not None else (binding.tool_pins or None)
        return await self.runtime.create_session(
            system_prompt=prompt,
            metadata=meta,
            tool_pins=pins,
        )

    async def get_session(self, session_id: str) -> Session | None:
        return await self.runtime.get_session(session_id)

    def run_turn(self, session_id: str, user_input: str) -> AsyncIterator[Event]:
        return self.runtime.run_turn(session_id, user_input)

    def resume(self, session_id: str, *, worker_id: str | None = None) -> AsyncIterator[Event]:
        return self.runtime.resume(session_id, worker_id=worker_id)

    async def pause_session(self, session_id: str) -> None:
        await self.runtime.pause_session(session_id)

    async def cancel_session(self, session_id: str) -> Session:
        return await self.runtime.cancel_session(session_id)

    async def list_events(self, session_id: str) -> list[Event]:
        return await self.runtime.list_events(session_id)

    async def list_turns(self, session_id: str) -> list[Turn]:
        return await self.runtime.list_turns(session_id)

    async def rebuild_state(self, session_id: str) -> SessionState:
        return await self.runtime.rebuild_state(session_id)

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        return await self.runtime.latest_checkpoint(session_id)

    async def acquire_lease(self, session_id: str, worker_id: str) -> WorkerLease:
        return await self.runtime.acquire_lease(session_id, worker_id)
