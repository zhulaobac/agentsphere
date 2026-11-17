"""Tenant identity records. Isolation key is ``tenant_id`` (Control Plane tenant)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TenantRecord(BaseModel):
    """Registered tenant. ``production=False`` is for test-only fixtures such as other-team."""

    tenant_id: str
    display_name: str
    owner: str
    team: str
    labels: dict[str, str] = Field(default_factory=dict)
    production: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
