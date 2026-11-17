"""Durable idempotency result cache. A completed tool side effect is stored before the event is yielded."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field


class CachedResult(BaseModel):
    idempotency_key: str
    session_id: str
    turn_id: str | None = None
    step_index: int | None = None
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    phase: str = "execute"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultCache(Protocol):
    async def put_result(self, record: CachedResult) -> None: ...

    async def get_result(self, idempotency_key: str) -> CachedResult | None: ...
