"""Checkpoint snapshots. Recovery loads the latest checkpoint, then replays the event tail."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.context import SessionState
from agentsphere.kernel.ids import new_id


class Checkpoint(BaseModel):
    """Durable snapshot of context/plan plus an optional workspace snapshot reference."""

    checkpoint_id: str = Field(default_factory=lambda: new_id("ckpt"))
    session_id: str
    seq: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = "turn_completed"
    turn_id: str | None = None
    turn_index: int | None = None
    session_state: SessionState
    workspace_snapshot_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointStore(Protocol):
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint: ...

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None: ...

    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None: ...

    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]: ...
