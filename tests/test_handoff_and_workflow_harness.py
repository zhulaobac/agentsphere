from __future__ import annotations

from agentsphere.kernel.actions import Handoff
from agentsphere.kernel.context import TurnContext
from agentsphere.kernel.events import EventType
from agentsphere.kernel.loop import WorkflowDefinition, WorkflowEdge, WorkflowNode, WorkflowStrategy
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text
from agentsphere.runtime import InMemoryEventStore, build_runtime
from agentsphere.tool_registry import ToolRegistry


class HandoffStrategy:
    name = "handoff-test"

    def step(self, ctx: TurnContext) -> Handoff:
        return Handoff(target="specialist-b", payload={"task": ctx.user_input})


async def test_harness_emits_handoff_and_ends_turn(registry: ToolRegistry) -> None:
    runtime = build_runtime(
        adapter=ModelAdapter(MockModelProvider()),
        registry=registry,
        strategy=HandoffStrategy(),
        store=InMemoryEventStore(),
    )
    session = await runtime.create_session()
    events = [e async for e in runtime.run_turn(session.session_id, "route me")]
    types = [e.type for e in events]
    assert EventType.HANDOFF in types
    handoff = next(e for e in events if e.type is EventType.HANDOFF)
    assert handoff.payload["target"] == "specialist-b"
    completed = next(e for e in events if e.type is EventType.TURN_COMPLETED)
    assert completed.payload["reason"] == "handoff"


async def test_workflow_turn_via_runtime(registry: ToolRegistry) -> None:
    definition = WorkflowDefinition(
        start="ask",
        nodes=[
            WorkflowNode(id="ask", type="model", prompt=None),
            WorkflowNode(id="echo", type="tool", tool_name="echo", arguments={"text": "{{user_input}}"}),
            WorkflowNode(id="done", type="end", output="ok"),
        ],
        edges=[
            WorkflowEdge(source="ask", target="echo"),
            WorkflowEdge(source="echo", target="done"),
        ],
    )
    runtime = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("go")])),
        registry=registry,
        strategy=WorkflowStrategy(definition),
        store=InMemoryEventStore(),
    )
    session = await runtime.create_session()
    events = [e async for e in runtime.run_turn(session.session_id, "ping")]
    types = [e.type for e in events]
    assert EventType.MODEL_CALL_COMPLETED in types
    assert EventType.TOOL_CALL_COMPLETED in types
    tool = next(e for e in events if e.type is EventType.TOOL_CALL_COMPLETED)
    assert tool.payload["output"] == "ping"
    completed = next(e for e in events if e.type is EventType.TURN_COMPLETED)
    assert completed.payload["reason"] == "workflow_end"
    assert completed.payload["output"] == "ok"
