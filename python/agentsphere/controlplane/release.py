"""In-process / store-backed deployments: canary → full, rollback, auto-rollback from signals."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentsphere.controlplane.types import DeploymentStatus, ReleaseChannel
from agentsphere.kernel.ids import new_id


class DeploymentRecord(BaseModel):
    """A version of an Agent in an environment. Not a Kubernetes object."""

    deployment_id: str = Field(default_factory=lambda: new_id("dep"))
    tenant: str
    agent_name: str
    agent_version: str
    environment: str = "default"
    channel: ReleaseChannel = ReleaseChannel.CANARY
    weight: float = 0.05
    status: DeploymentStatus = DeploymentStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rolled_back_from: str | None = None
    rolled_back_to: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


CANARY_WEIGHT = 0.05
FULL_WEIGHT = 1.0
