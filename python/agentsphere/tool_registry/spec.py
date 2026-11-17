"""Tool artifact metadata for Registry v1."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolSpec(BaseModel):
    """Registered tool: JSON schema, SemVer, owner, risk, idempotency. No MCP/sandbox/vault in P0."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    version: str
    owner: str
    description: str = ""
    json_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": True}
    )
    risk_level: RiskLevel = RiskLevel.LOW
    idempotent: bool = False
    idempotency_key_fields: list[str] = Field(default_factory=list)
    handler: Callable[[Mapping[str, Any]], Any] | None = None

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError(f"tool version must be SemVer, got {value!r}") from exc
        return value

    @property
    def parsed_version(self) -> Version:
        return Version(self.version)
