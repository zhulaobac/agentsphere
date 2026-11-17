"""Load the latest checkpoint, replay remaining events, and rebuild a TurnResume if a turn is open."""

from __future__ import annotations

from collections.abc import Iterable

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.context import SessionState
from agentsphere.kernel.events import Event, EventType
from agentsphere.model_adapter.types import ModelResponse, ToolCall
from agentsphere.runtime.checkpoint import Checkpoint
from agentsphere.kernel.control import TurnResume
from agentsphere.runtime.replay import empty_state, replay


def recover_state(
    events: Iterable[Event],
    *,
    session_id: str,
    checkpoint: Checkpoint | None = None,
) -> SessionState:
    """Fold the event tail onto a checkpoint. Event sourcing remains the source of truth."""
    event_list = list(events)
    if checkpoint is None:
        return replay(event_list, session_id=session_id)
    tail = [event for event in event_list if (event.seq or 0) > checkpoint.seq]
    return replay(tail, session_id=session_id, initial=checkpoint.session_state)


def in_progress_turn(events: Iterable[Event]) -> tuple[str, str, int] | None:
    """Return (turn_id, user_input, turn_index) if a turn started and has not completed."""
    open_turn: tuple[str, str, int] | None = None
    for event in events:
        if event.type is EventType.TURN_STARTED:
            open_turn = (
                event.turn_id or "",
                str(event.payload.get("user_input", "")),
                int(event.payload.get("turn_index", 0)),
            )
        elif event.type in {EventType.TURN_COMPLETED, EventType.TURN_CANCELLED}:
            open_turn = None
    return open_turn


def reconstruct_resume(events: Iterable[Event], *, turn_id: str) -> TurnResume:
    """Rebuild Kernel resume context from the events of an in-progress turn."""
    turn_events = [event for event in events if event.turn_id == turn_id]
    user_input = ""
    turn_index = 0
    start_step_index = 0
    last_model: ModelResponse | None = None
    last_tools: list[dict[str, object]] | None = None
    pending: list[ToolCallRequest] | None = None
    open_step: int | None = None

    for event in turn_events:
        if event.type is EventType.TURN_STARTED:
            user_input = str(event.payload.get("user_input", ""))
            turn_index = int(event.payload.get("turn_index", turn_index))
        elif event.type is EventType.STEP_STARTED:
            open_step = int(event.payload.get("index", 0))
        elif event.type is EventType.STEP_COMPLETED:
            completed_index = int(event.payload.get("index", open_step or 0))
            start_step_index = completed_index + 1
            open_step = None
        elif event.type is EventType.MODEL_CALL_COMPLETED:
            raw_calls = event.payload.get("tool_calls") or []
            tool_calls = [ToolCall.model_validate(c) for c in raw_calls]
            last_model = ModelResponse(
                content=event.payload.get("content"),
                tool_calls=tool_calls,
                provider=str(event.payload.get("provider", "")),
            )
            pending = [
                ToolCallRequest(id=call.id, name=call.name, arguments=call.arguments)
                for call in tool_calls
            ] or None
            last_tools = None
        elif event.type is EventType.TOOL_CALL_COMPLETED:
            result = {
                "call_id": event.payload.get("call_id"),
                "name": event.payload.get("name"),
                "output": event.payload.get("output"),
                "error": None,
                "idempotency_key": event.payload.get("idempotency_key", ""),
            }
            last_tools = (last_tools or []) + [result]
            pending = None
            last_model = None
        elif event.type is EventType.TOOL_CALL_FAILED:
            result = {
                "call_id": event.payload.get("call_id"),
                "name": event.payload.get("name"),
                "output": "",
                "error": event.payload.get("error"),
                "idempotency_key": event.payload.get("idempotency_key", ""),
            }
            last_tools = (last_tools or []) + [result]
            pending = None
            last_model = None

    if open_step is not None:
        start_step_index = open_step

    return TurnResume(
        turn_id=turn_id,
        turn_index=turn_index,
        start_step_index=start_step_index,
        user_input=user_input,
        last_model_response=last_model,
        last_tool_results=last_tools,
        pending_tool_calls=pending,
    )


def started_tool_keys(events: Iterable[Event], *, turn_id: str | None = None) -> set[str]:
    keys: set[str] = set()
    for event in events:
        if turn_id is not None and event.turn_id != turn_id:
            continue
        if event.type is EventType.TOOL_CALL_STARTED:
            key = str(event.payload.get("idempotency_key") or "")
            if key:
                keys.add(key)
    return keys


def completed_tool_keys(events: Iterable[Event], *, turn_id: str | None = None) -> set[str]:
    keys: set[str] = set()
    for event in events:
        if turn_id is not None and event.turn_id != turn_id:
            continue
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
            key = str(event.payload.get("idempotency_key") or "")
            if key:
                keys.add(key)
    return keys


def in_flight_tool_keys(events: Iterable[Event], *, turn_id: str | None = None) -> set[str]:
    return started_tool_keys(events, turn_id=turn_id) - completed_tool_keys(events, turn_id=turn_id)


def empty_session_state(session_id: str) -> SessionState:
    return empty_state(session_id)
