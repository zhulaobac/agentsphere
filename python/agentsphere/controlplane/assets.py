"""Uniform asset model: version, owner, labels, dependency graph / lineage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentsphere.controlplane.types import AssetKind, AssetMetadata, AssetRef, require_semver
from agentsphere.kernel.ids import new_id


class AssetRecord(BaseModel):
    """Versioned registry entry for Agent, Prompt, Tool, Skill, Dataset, Evaluator, Policy, stubs."""

    kind: AssetKind
    metadata: AssetMetadata
    version: str
    spec: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    immutable: bool = False
    stub: bool = False

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        return require_semver(value)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def owner(self) -> str:
        return self.metadata.owner

    @property
    def tenant(self) -> str:
        return self.metadata.tenant

    @property
    def labels(self) -> dict[str, str]:
        return self.metadata.labels

    def ref(self) -> AssetRef:
        return AssetRef(kind=self.kind, name=self.name, version=self.version, tenant=self.tenant)


class DependencyEdge(BaseModel):
    """Lineage edge: consumer → dependency. Also recorded as a ControlEvent."""

    edge_id: str = Field(default_factory=lambda: new_id("edge"))
    source: AssetRef
    target: AssetRef
    relation: str = "uses"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str | None = None


STUB_KINDS = frozenset({AssetKind.MCP_SERVER, AssetKind.MEMORY_STORE})
