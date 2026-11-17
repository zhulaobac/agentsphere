from __future__ import annotations

from agentsphere.kernel.events import EventType
from agentsphere.kernel.loop import ReActStrategy
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text, assistant_tool_response
from agentsphere.runtime import InMemoryEventStore, build_runtime, replay
from agentsphere.trace import TraceRecorder


async def test_react_turn_with_mock_model_registered_tool_events_and_trace(registry) -> None:
    provider = MockModelProvider(
        script=[
            assistant_tool_response("echo", {"text": "hi"}),
            assistant_text("echoed: hi"),
        ]
    )
    tracer = TraceRecorder()
    runtime = build_runtime(
        adapter=ModelAdapter(provider),
        registry=registry,
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
        tracer=tracer,
    )

    session = await runtime.create_session(system_prompt="You are a test agent.")
    events = [event async for event in runtime.run_turn(session.session_id, "please echo hi")]
    types = [event.type for event in events]

    assert EventType.TURN_STARTED in types
    assert EventType.MODEL_CALL_STARTED in types
    assert EventType.MODEL_CALL_COMPLETED in types
    assert EventType.TOOL_CALL_STARTED in types
    assert EventType.TOOL_CALL_COMPLETED in types
    assert EventType.OUTPUT in types
    assert EventType.TURN_COMPLETED in types

    tool_done = next(e for e in events if e.type is EventType.TOOL_CALL_COMPLETED)
    assert tool_done.payload["output"] == "hi"
    assert tool_done.payload["name"] == "echo"

    output = next(e for e in events if e.type is EventType.OUTPUT)
    assert output.payload["text"] == "echoed: hi"

    stored = await runtime.list_events(session.session_id)
    assert stored[0].type is EventType.SESSION_CREATED
    assert stored[0].seq == 1
    rebuilt = replay(stored)
    roles = [m.role.value for m in rebuilt.messages]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles
    assert rebuilt.system_prompt == "You are a test agent."

    tree = tracer.inspect(session.session_id)
    assert tree.root is not None
    assert tree.root.kind == "session"
    turn = tree.root.children[0]
    assert turn.kind == "turn"
    step_kinds = [child.kind for child in turn.children]
    assert step_kinds.count("step") >= 2
    nested = [grandchild.kind for step in turn.children for grandchild in step.children]
    assert "model.call" in nested
    assert "tool.call" in nested

    tracer.force_flush()
    otel_names = {span.name for span in tracer.exported_spans()}
    assert "agentsphere.turn" in otel_names
    assert "agentsphere.model.call" in otel_names
    assert "agentsphere.tool.call" in otel_names

    turns = await runtime.list_turns(session.session_id)
    assert len(turns) == 1
    assert turns[0].output == "echoed: hi"
