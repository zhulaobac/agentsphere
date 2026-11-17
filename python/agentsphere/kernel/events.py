"""Standardized Kernel events. Shared with Runtime, Trace, and (later) Eval.

Event payloads are JSON-serializable so a future Go control-plane or runtime
can consume the same log without depending on Python objects.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.ids import new_id


class EventType(str, Enum):
    SESSION_CREATED = "session.created"
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    MODEL_CALL_STARTED = "model.call.started"
    MODEL_CALL_DELTA = "model.call.delta"
    MODEL_CALL_COMPLETED = "model.call.completed"
    MODEL_CALL_FAILED = "model.call.failed"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    WORKFLOW_TRANSITION = "workflow.transition"
    OUTPUT = "output"
    HANDOFF = "handoff"
    ERROR = "error"


class Event(BaseModel):
    """Append-only event. OpenTelemetry-compatible via trace_id / span_id."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    type: EventType
    session_id: str
    turn_id: str | None = None
    step_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    seq: int | None = None
    parent_event_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


class EventListener(Protocol):
    async def on_event(self, event: Event) -> None: ...


ListenerFn = Callable[[Event], Awaitable[None] | None]


class EventEmitter:
    """Fan-out emitter. Kernel yields events; Runtime persists them; Trace records them."""

    def __init__(self, listeners: list[EventListener | ListenerFn] | None = None) -> None:
        self._listeners: list[EventListener | ListenerFn] = list(listeners or [])

    def add_listener(self, listener: EventListener | ListenerFn) -> None:
        self._listeners.append(listener)

    async def emit(self, event: Event) -> Event:
        for listener in self._listeners:
            on_event = getattr(listener, "on_event", listener)
            result = on_event(event)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        return event

    def make(
        self,
        type: EventType,
        *,
        session_id: str,
        turn_id: str | None = None,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Event:
        return Event(
            type=type,
            session_id=session_id,
            turn_id=turn_id,
            step_id=step_id,
            payload=payload or {},
            parent_event_id=parent_event_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
