from __future__ import annotations

import pytest

from agentsphere.kernel.events import Event, EventType
from agentsphere.trace import TraceRecorder


async def _run(recorder: TraceRecorder, session_id: str = "s1") -> None:
    events = [
        Event(type=EventType.SESSION_CREATED, session_id=session_id),
        Event(type=EventType.TURN_STARTED, session_id=session_id, turn_id="t1"),
        Event(type=EventType.STEP_STARTED, session_id=session_id, turn_id="t1", step_id="st1"),
        Event(
            type=EventType.MODEL_CALL_STARTED,
            session_id=session_id,
            turn_id="t1",
            step_id="st1",
        ),
        Event(
            type=EventType.MODEL_CALL_COMPLETED,
            session_id=session_id,
            turn_id="t1",
            step_id="st1",
            payload={"content": "x"},
        ),
        Event(type=EventType.STEP_COMPLETED, session_id=session_id, turn_id="t1", step_id="st1"),
        Event(
            type=EventType.STEP_STARTED,
            session_id=session_id,
            turn_id="t1",
            step_id="st2",
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session_id,
            turn_id="t1",
            step_id="st2",
            payload={"call_id": "c1", "name": "echo"},
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=session_id,
            turn_id="t1",
            step_id="st2",
            payload={"call_id": "c1", "output": "ok"},
        ),
        Event(type=EventType.STEP_COMPLETED, session_id=session_id, turn_id="t1", step_id="st2"),
        Event(type=EventType.TURN_COMPLETED, session_id=session_id, turn_id="t1", payload={"reason": "completed"}),
    ]
    for event in events:
        await recorder.on_event(event)


@pytest.mark.asyncio
async def test_trace_hierarchy_session_turn_step_calls() -> None:
    recorder = TraceRecorder()
    await _run(recorder)
    tree = recorder.inspect("s1")
    assert tree.root is not None
    assert tree.root.kind == "session"
    turn = tree.root.children[0]
    assert turn.kind == "turn"
    steps = turn.children
    assert [s.kind for s in steps] == ["step", "step"]
    assert steps[0].children[0].kind == "model.call"
    assert steps[1].children[0].kind == "tool.call"


@pytest.mark.asyncio
async def test_trace_exports_opentelemetry_spans() -> None:
    recorder = TraceRecorder()
    await _run(recorder)
    recorder.force_flush()
    spans = recorder.exported_spans()
    names = {span.name for span in spans}
    assert "agentsphere.turn" in names
    assert "agentsphere.step" in names
    assert "agentsphere.model.call" in names
    assert "agentsphere.tool.call" in names
    # Session span is still open (session has not ended).
    kinds = {span.attributes.get("agentsphere.kind") for span in spans}
    assert "turn" in kinds
