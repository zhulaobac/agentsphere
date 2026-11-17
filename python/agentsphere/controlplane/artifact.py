"""Unified Agent Artifact (apiVersion agentsphere/v1, kind Agent)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from agentsphere.controlplane.types import (
    API_VERSION,
    AssetKind,
    AssetMetadata,
    AssetRef,
    DeploymentStatus,
    GateThresholds,
    LifecyclePhase,
    ReleaseChannel,
    require_semver,
)


class HarnessSpec(BaseModel):
    strategy: str = "react"
    version: str = "1.0.0"
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        return require_semver(value)


class ModelSpec(BaseModel):
    primary: str = "mock"
    fallbacks: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class RuntimeSpec(BaseModel):
    resources: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = 1
    timeout: float | None = None


class AgentEvalSpec(BaseModel):
    datasets: list[AssetRef] = Field(default_factory=list)
    gates: list["AgentGateSpec"] = Field(default_factory=list)


class AgentGateSpec(BaseModel):
    name: str = "default"
    thresholds: GateThresholds = Field(default_factory=GateThresholds)
    baseline_version: str | None = None


class AgentSpec(BaseModel):
    type: Literal["managed", "byo"] = "managed"
    harness: HarnessSpec = Field(default_factory=HarnessSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    prompts: list[AssetRef] = Field(default_factory=list)
    context: AssetRef | None = None
    memory: AssetRef | None = None
    tools: list[AssetRef] = Field(default_factory=list)
    policies: AssetRef | None = None
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    eval: AgentEvalSpec = Field(default_factory=AgentEvalSpec)

    def dependency_refs(self) -> list[AssetRef]:
        refs: list[AssetRef] = []
        refs.extend(self.prompts)
        if self.context is not None:
            refs.append(self.context)
        if self.memory is not None:
            refs.append(self.memory)
        refs.extend(self.tools)
        if self.policies is not None:
            refs.append(self.policies)
        refs.extend(self.eval.datasets)
        return refs


class DeployedVersion(BaseModel):
    version: str
    environment: str = "default"
    channel: ReleaseChannel = ReleaseChannel.CANARY
    weight: float = 0.05
    deployment_id: str
    status: DeploymentStatus = DeploymentStatus.ACTIVE


class AgentStatus(BaseModel):
    phase: LifecyclePhase = LifecyclePhase.DRAFT
    deployedVersions: list[DeployedVersion] = Field(default_factory=list)
    lastExperimentRunId: str | None = None
    lastGateResultId: str | None = None
    releaseChannel: ReleaseChannel | None = None

    def active_deployments(self) -> list[DeployedVersion]:
        return [d for d in self.deployedVersions if d.status is DeploymentStatus.ACTIVE]


class AgentArtifact(BaseModel):
    """Everything-as-Artifact document. JSON-serializable counterpart of the design YAML."""

    apiVersion: Literal["agentsphere/v1"] = API_VERSION
    kind: Literal["Agent"] = AssetKind.AGENT.value
    metadata: AssetMetadata
    spec: AgentSpec = Field(default_factory=AgentSpec)
    status: AgentStatus = Field(default_factory=AgentStatus)
    version: str = "0.1.0"

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        return require_semver(value)

    def resource_key(self) -> str:
        return f"{self.metadata.tenant}/{self.metadata.name}@{self.version}"

    def to_document(self) -> dict[str, Any]:
        """Dict matching the design YAML schema (apiVersion / kind / metadata / spec / status)."""
        return {
            "apiVersion": self.apiVersion,
            "kind": self.kind,
            "metadata": self.metadata.model_dump(mode="json"),
            "spec": self.spec.model_dump(mode="json"),
            "status": self.status.model_dump(mode="json"),
            "version": self.version,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> AgentArtifact:
        return cls.model_validate(document)
