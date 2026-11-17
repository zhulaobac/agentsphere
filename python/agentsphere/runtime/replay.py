"""Event-sourcing lite: SessionState is the fold of the event log."""

from __future__ import annotations

from collections.abc import Iterable

from agentsphere.kernel.context import SessionState
from agentsphere.kernel.events import Event, EventType
from agentsphere.model_adapter.types import Message, MessageRole, ToolCall


def empty_state(session_id: str) -> SessionState:
    return SessionState(session_id=session_id)


def apply_event(state: SessionState, event: Event) -> SessionState:
    payload = event.payload
    if event.type is EventType.SESSION_CREATED:
        state.system_prompt = payload.get("system_prompt", state.system_prompt)
        state.metadata = payload.get("metadata", state.metadata)
        state.tool_pins = payload.get("tool_pins", state.tool_pins)
        return state

    if event.type is EventType.TURN_STARTED:
        state.turn_index = int(payload.get("turn_index", state.turn_index + 1))
        state.messages.append(Message(role=MessageRole.USER, content=str(payload.get("user_input", ""))))
        return state

    if event.type is EventType.MODEL_CALL_COMPLETED:
        raw_calls = payload.get("tool_calls") or []
        tool_calls = [ToolCall.model_validate(c) for c in raw_calls] or None
        state.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=payload.get("content"),
                tool_calls=tool_calls,
            )
        )
        return state

    if event.type is EventType.TOOL_CALL_COMPLETED:
        state.messages.append(
            Message(
                role=MessageRole.TOOL,
                name=payload.get("name"),
                tool_call_id=payload.get("call_id"),
                content=str(payload.get("output", "")),
            )
        )
        return state

    if event.type is EventType.TOOL_CALL_FAILED:
        state.messages.append(
            Message(
                role=MessageRole.TOOL,
                name=payload.get("name"),
                tool_call_id=payload.get("call_id"),
                content=f"ERROR: {payload.get('error', '')}",
            )
        )
        return state

    if event.type is EventType.WORKFLOW_TRANSITION:
        state.workflow_cursor = payload.get("to", state.workflow_cursor)
        return state

    if event.type is EventType.TURN_COMPLETED:
        if "workflow_cursor" in payload:
            state.workflow_cursor = payload.get("workflow_cursor")
        if "workflow_node_outputs" in payload and payload["workflow_node_outputs"] is not None:
            state.workflow_node_outputs = dict(payload["workflow_node_outputs"])
        return state

    return state


def replay(events: Iterable[Event], session_id: str | None = None) -> SessionState:
    events = list(events)
    sid = session_id
    if sid is None:
        if not events:
            raise ValueError("replay() requires events or an explicit session_id")
        sid = events[0].session_id
    state = empty_state(sid)
    for event in events:
        state = apply_event(state, event)
    return state
