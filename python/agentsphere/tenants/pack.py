"""Tenant pack protocol: the onboarding path a later second tenant should follow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.controlplane.runtime import ManagedRuntimeProvider
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import Principal
from agentsphere.evaluation.dataset import DatasetService
from agentsphere.evaluation.evaluator import EvaluatorRegistry
from agentsphere.identity.agent import AgentIdentity
from agentsphere.identity.audit import InMemoryAuditLog
from agentsphere.identity.policy import StaticPolicyEngine
from agentsphere.identity.vault import InMemoryVault
from agentsphere.runtime.service import SessionRuntime
from agentsphere.sandbox.subprocess import SubprocessSandbox
from agentsphere.tenants.registry import TenantRegistry
from agentsphere.tenants.types import TenantRecord
from agentsphere.tool_registry.registry import ToolRegistry
from agentsphere.trace.tracer import TraceRecorder


@dataclass
class OnboardedTenant:
    """Result of :meth:`TenantPack.onboard`. Holds isolated Control Plane + runtime handles."""

    record: TenantRecord
    principal: Principal
    control: ControlPlane
    tenants: TenantRegistry
    artifact: AgentArtifact
    tools: ToolRegistry
    evaluators: EvaluatorRegistry
    datasets: DatasetService
    identity: AgentIdentity
    vault: InMemoryVault | None = None
    sandbox: SubprocessSandbox | None = None
    audit: InMemoryAuditLog | None = None
    tracer: TraceRecorder | None = None
    policy: StaticPolicyEngine | None = None
    runtime: SessionRuntime | None = None
    provider: ManagedRuntimeProvider | None = None
    secret_name: str = ""
    secret_value: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        return self.artifact.metadata.name

    @property
    def agent_version(self) -> str:
        return self.artifact.version

    @property
    def tenant_id(self) -> str:
        return self.record.tenant_id


class TenantPack(Protocol):
    """Implement this for a new Anchor Tenant. Do not share state across packs."""

    def tenant_record(self) -> TenantRecord: ...

    def owner_principal(self) -> Principal: ...

    async def onboard(
        self,
        *,
        control: ControlPlane | None = None,
        tenants: TenantRegistry | None = None,
    ) -> OnboardedTenant: ...
