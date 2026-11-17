from __future__ import annotations

import pytest

from agentsphere.kernel.actions import CallModel, CallTool, Finish, ToolCallRequest
from agentsphere.kernel.context import AssembledContext, SessionState, TurnContext
from agentsphere.kernel.loop import ReActStrategy, UnimplementedStrategyError, get_strategy_class
from agentsphere.model_adapter.types import Message, MessageRole, ModelResponse, ToolCall


def _ctx(**kwargs: object) -> TurnContext:
    assembled = AssembledContext(
        messages=[Message(role=MessageRole.USER, content="hi")],
        tools=[],
    )
    values: dict[str, object] = {
        "session_state": SessionState(session_id="s1"),
        "user_input": "hi",
        "turn_id": "t1",
        "step_index": 0,
        "assembled": assembled,
    }
    values.update(kwargs)
    return TurnContext.model_validate(values)


def test_react_starts_with_call_model() -> None:
    action = ReActStrategy().step(_ctx())
    assert isinstance(action, CallModel)
    assert action.messages[0].content == "hi"


def test_react_calls_pending_tools() -> None:
    pending = [ToolCallRequest(id="c1", name="echo", arguments={"text": "x"})]
    action = ReActStrategy().step(_ctx(pending_tool_calls=pending))
    assert isinstance(action, CallTool)
    assert action.calls[0].name == "echo"


def test_react_finishes_when_model_returns_text() -> None:
    action = ReActStrategy().step(
        _ctx(last_model_response=ModelResponse(content="done", tool_calls=[]))
    )
    assert isinstance(action, Finish)
    assert action.output == "done"


def test_react_does_not_finish_when_tool_calls_present() -> None:
    response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
    )
    action = ReActStrategy().step(_ctx(last_model_response=response))
    assert isinstance(action, CallModel)


def test_react_max_steps() -> None:
    action = ReActStrategy(max_steps=2).step(_ctx(step_index=2))
    assert isinstance(action, Finish)
    assert action.reason == "max_steps"


def test_planner_executor_is_explicitly_unimplemented() -> None:
    with pytest.raises(UnimplementedStrategyError, match="P2"):
        get_strategy_class("planner-executor")


def test_multi_agent_is_explicitly_unimplemented() -> None:
    with pytest.raises(UnimplementedStrategyError, match="P2"):
        get_strategy_class("multi-agent")


def test_react_is_registered() -> None:
    assert get_strategy_class("react") is ReActStrategy
