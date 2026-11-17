from __future__ import annotations

from agentsphere.kernel.actions import CallModel, CallTool, Finish
from agentsphere.kernel.context import SessionState, TurnContext
from agentsphere.kernel.loop import WorkflowDefinition, WorkflowEdge, WorkflowNode, WorkflowStrategy


def _definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        start="ask",
        nodes=[
            WorkflowNode(id="ask", type="model", prompt="say hello"),
            WorkflowNode(id="echo", type="tool", tool_name="echo", arguments={"text": "hello"}),
            WorkflowNode(id="done", type="end", output="finished"),
        ],
        edges=[
            WorkflowEdge(source="ask", target="echo"),
            WorkflowEdge(source="echo", target="done"),
        ],
    )


def _ctx(state: SessionState, step: int = 0) -> TurnContext:
    return TurnContext(
        session_state=state,
        user_input="go",
        turn_id="t1",
        step_index=step,
    )


def test_workflow_walks_dag_deterministically() -> None:
    strategy = WorkflowStrategy(_definition())
    state = SessionState(session_id="s1")

    first = strategy.step(_ctx(state))
    assert isinstance(first, CallModel)
    assert state.workflow_cursor == "ask"

    state.workflow_node_outputs["ask"] = {"content": "hello"}
    second = strategy.step(_ctx(state, step=1))
    assert isinstance(second, CallTool)
    assert second.calls[0].name == "echo"
    assert state.workflow_cursor == "echo"

    state.workflow_node_outputs["echo"] = {"output": "hello"}
    third = strategy.step(_ctx(state, step=2))
    assert isinstance(third, Finish)
    assert third.output == "finished"
    assert third.reason == "workflow_end"


def test_workflow_conditional_edge() -> None:
    definition = WorkflowDefinition(
        start="branch",
        nodes=[
            WorkflowNode(id="branch", type="model", prompt="pick"),
            WorkflowNode(id="left", type="end", output="L"),
            WorkflowNode(id="right", type="end", output="R"),
        ],
        edges=[
            WorkflowEdge(source="branch", target="left", when="LEFT"),
            WorkflowEdge(source="branch", target="right", when="RIGHT"),
        ],
    )
    strategy = WorkflowStrategy(definition)
    state = SessionState(session_id="s1")
    strategy.step(_ctx(state))
    state.workflow_node_outputs["branch"] = {"content": "go RIGHT please"}
    action = strategy.step(_ctx(state, step=1))
    assert isinstance(action, Finish)
    assert action.output == "R"
