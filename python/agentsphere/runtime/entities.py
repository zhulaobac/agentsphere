"""P0 Runtime entities: Session, Turn. Events live in the Kernel (shared schema)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentsphere.kernel.ids import new_id


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HANDED_OFF = "handed_off"


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("ses"))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = ""
    tool_pins: dict[str, str] = Field(default_factory=dict)


class Turn(BaseModel):
    turn_id: str
    session_id: str
    turn_index: int
    user_input: str
    status: TurnStatus = TurnStatus.RUNNING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    output: str | None = None
    reason: str | None = None
