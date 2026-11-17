from __future__ import annotations

from agentsphere.kernel.actions import CallModel
from agentsphere.kernel.context import SessionState, TurnContext
from agentsphere.kernel.events import EventType
from agentsphere.kernel.harness import AgentHarness
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text
from agentsphere.model_adapter.types import InferenceParams, Message, MessageRole


class AlwaysCallModel:
    name = "always-model"

    def step(self, ctx: TurnContext) -> CallModel:
        messages = list(ctx.assembled.messages) if ctx.assembled is not None else [
            Message(role=MessageRole.USER, content=ctx.user_input)
        ]
        return CallModel(messages=messages, params=InferenceParams())


async def test_max_steps_is_not_off_by_one() -> None:
    harness = AgentHarness(
        AlwaysCallModel(),
        ModelAdapter(MockModelProvider(script=[assistant_text("ok")])),
        max_steps=3,
    )
    events = [event async for event in harness.run_turn(SessionState(session_id="s1"), "loop")]
    completed = next(event for event in events if event.type is EventType.TURN_COMPLETED)
    assert completed.payload["reason"] == "max_steps"
    assert len([e for e in events if e.type is EventType.STEP_STARTED]) == 3
    assert len([e for e in events if e.type is EventType.MODEL_CALL_COMPLETED]) == 3
