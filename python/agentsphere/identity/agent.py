"""Agent Identity: a workload principal distinct from any human developer account."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentsphere.kernel.ids import new_id


class AgentIdentity(BaseModel):
    """SPIFFE-shaped identity. Production can mint real SPIFFE IDs; this field is a stable handle."""

    agent_id: str = Field(default_factory=lambda: new_id("agent"))
    name: str = "default-agent"
    tenant_id: str = "default"
    spiffe_id: str | None = None
    principal_type: Literal["agent"] = "agent"
    labels: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if not self.spiffe_id:
            self.spiffe_id = f"spiffe://agentsphere/{self.tenant_id}/agent/{self.agent_id}"

    @property
    def is_developer_account(self) -> bool:
        return False
