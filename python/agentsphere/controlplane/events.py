"""Append-only Control Plane events. Lifecycle and lineage are reconstructed from this log."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentsphere.controlplane.types import Principal
from agentsphere.kernel.ids import new_id


class ControlEventType(str, Enum):
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"
    ASSET_CREATED = "asset.created"
    ASSET_UPDATED = "asset.updated"
    LIFECYCLE_TRANSITION = "lifecycle.transition"
    RELEASE_CANARY = "release.canary"
    RELEASE_FULL = "release.full"
    RELEASE_BLOCKED = "release.blocked"
    RELEASE_ROLLBACK = "release.rollback"
    RELEASE_AUTO_ROLLBACK = "release.auto_rollback"
    GATE_RECORDED = "gate.recorded"
    SLO_RECORDED = "slo.recorded"
    LINEAGE_EDGE = "lineage.edge"
    EVALUATION_RECORDED = "evaluation.recorded"
    APPROVAL_RECORDED = "approval.recorded"


class ControlEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("cpe"))
    type: ControlEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: Principal | None = None
    tenant: str = ""
    resource: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
