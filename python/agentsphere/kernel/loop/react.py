"""ReAct: single-loop reason / act until the model produces a final answer."""

from __future__ import annotations

from agentsphere.kernel.actions import CallModel, CallTool, Finish, ToolCallRequest
from agentsphere.kernel.context import TurnContext
from agentsphere.kernel.loop.strategy import register_strategy
from agentsphere.model_adapter.types import InferenceParams


class ReActStrategy:
    """Classic ReAct: CallModel → (CallTool → CallModel)* → Finish."""

    name = "react"

    def __init__(self, max_steps: int = 16, params: InferenceParams | None = None) -> None:
        self.max_steps = max_steps
        self.params = params or InferenceParams()

    def step(self, ctx: TurnContext) -> CallModel | CallTool | Finish:
        if ctx.pending_tool_calls:
            return CallTool(calls=list(ctx.pending_tool_calls))

        if ctx.last_model_response is not None and not ctx.last_model_response.has_tool_calls:
            return Finish(
                output=ctx.last_model_response.content or "",
                reason="completed",
            )

        if ctx.step_index >= self.max_steps:
            output = ""
            if ctx.last_model_response is not None:
                output = ctx.last_model_response.content or ""
            return Finish(output=output, reason="max_steps")

        assembled = ctx.assembled
        if assembled is None:
            raise RuntimeError("ReActStrategy.step requires assembled context")
        return CallModel(
            messages=list(assembled.messages),
            tools=list(assembled.tools),
            params=self.params,
        )


register_strategy(ReActStrategy.name, ReActStrategy)
