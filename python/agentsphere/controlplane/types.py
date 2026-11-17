"""Shared Control Plane types: identity, asset kinds, lifecycle, gate contract."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, field_validator

from agentsphere.kernel.ids import new_id


API_VERSION = "agentsphere/v1"


class AssetKind(str, Enum):
    AGENT = "Agent"
    PROMPT = "Prompt"
    TOOL = "Tool"
    SKILL = "Skill"
    DATASET = "Dataset"
    EVALUATOR = "Evaluator"
    POLICY = "Policy"
    MCP_SERVER = "MCPServer"
    MEMORY_STORE = "MemoryStore"
    CONTEXT = "Context"


class LifecyclePhase(str, Enum):
    DRAFT = "Draft"
    BUILT = "Built"
    EVALUATED = "Evaluated"
    APPROVED = "Approved"
    RELEASED = "Released"
    DEPRECATED = "Deprecated"
    RETIRED = "Retired"


class ReleaseChannel(str, Enum):
    CANARY = "Canary"
    FULL = "Full"


class DeploymentStatus(str, Enum):
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class Principal(BaseModel):
    """Caller identity for tenant/owner scoping. Not production RBAC/ABAC."""

    name: str
    tenant: str
    owner: str
    team: str = ""

    @classmethod
    def from_owner(cls, owner: str, tenant: str, *, team: str = "") -> Principal:
        return cls(name=owner, tenant=tenant, owner=owner, team=team)


class AssetMetadata(BaseModel):
    name: str
    owner: str
    team: str = ""
    tenant: str
    labels: dict[str, str] = Field(default_factory=dict)


class AssetRef(BaseModel):
    """Pinned dependency: kind/name@version. Optional tenant defaults to the consumer's."""

    kind: AssetKind
    name: str
    version: str
    tenant: str | None = None

    def key(self, default_tenant: str = "") -> tuple[str, str, str, str]:
        return (self.kind.value, self.tenant or default_tenant, self.name, self.version)

    def display(self) -> str:
        return f"{self.kind.value}/{self.name}@{self.version}"


class GateCheck(BaseModel):
    """One Release Gate criterion. Skipped checks do not fail the gate."""

    name: str
    passed: bool
    skipped: bool = False
    observed: Any = None
    threshold: Any = None
    reason: str = ""


class GateThresholds(BaseModel):
    """Configurable Release Gate. Cost/latency apply only when those metrics exist."""

    min_quality: float = 0.8
    max_regression: float = 0.05
    require_safety_pass: bool = True
    max_cost: float | None = None
    max_latency_ms: float | None = None


class GateDecision(BaseModel):
    """Contract the Control Plane requires before Approved → Released."""

    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)
    reason: str = ""
    experiment_run_id: str | None = None
    baseline_run_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def failed_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.skipped and not c.passed]

    def derived_passed(self) -> bool:
        material = [c for c in self.checks if not c.skipped]
        if not material:
            return False
        return all(c.passed for c in material)


class SLOSignal(BaseModel):
    """Recorded (not live-scraped) SLO/gate signal that may trigger auto-rollback."""

    signal_id: str = Field(default_factory=lambda: new_id("slo"))
    deployment_id: str
    metric: str
    value: float
    threshold: float
    direction: Literal["below_is_bad", "above_is_bad"] = "below_is_bad"
    source: str = "recorded"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def breached(self) -> bool:
        if self.direction == "below_is_bad":
            return self.value < self.threshold
        return self.value > self.threshold


def require_semver(value: str) -> str:
    try:
        Version(value)
    except InvalidVersion as exc:
        raise ValueError(f"version must be SemVer, got {value!r}") from exc
    return value


class SemVerModel(BaseModel):
    version: str

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        return require_semver(value)
