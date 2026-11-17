"""Append-only audit log for tool calls and authorization decisions. Payloads must be redacted."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentsphere.identity.redact import redact
from agentsphere.kernel.ids import new_id


class AuditRecord(BaseModel):
    audit_id: str = Field(default_factory=lambda: new_id("aud"))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    action: str
    resource: str
    decision: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLog(Protocol):
    async def record(self, record: AuditRecord) -> AuditRecord: ...

    async def list(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        action: str | None = None,
    ) -> list[AuditRecord]: ...


class InMemoryAuditLog:
    def __init__(self, extra_secrets: set[str] | None = None) -> None:
        self._records: list[AuditRecord] = []
        self._extra_secrets = set(extra_secrets or ())

    def add_secret(self, secret: str) -> None:
        if secret:
            self._extra_secrets.add(secret)

    async def record(self, record: AuditRecord) -> AuditRecord:
        stored = record.model_copy(deep=True)
        stored.details = redact(stored.details, extra_secrets=self._extra_secrets)
        self._records.append(stored)
        return stored.model_copy(deep=True)

    async def list(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        action: str | None = None,
    ) -> list[AuditRecord]:
        items = self._records
        if session_id is not None:
            items = [r for r in items if r.session_id == session_id]
        if agent_id is not None:
            items = [r for r in items if r.agent_id == agent_id]
        if action is not None:
            items = [r for r in items if r.action == action]
        return [r.model_copy(deep=True) for r in items]
