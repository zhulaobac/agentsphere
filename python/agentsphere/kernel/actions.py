"""Actions returned by LoopStrategy.step: the Kernel's next-step vocabulary."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from agentsphere.model_adapter.types import InferenceParams, Message, ToolDefinition


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    version: str | None = None


class CallModel(BaseModel):
    type: Literal["call_model"] = "call_model"
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    params: InferenceParams = Field(default_factory=InferenceParams)


class CallTool(BaseModel):
    type: Literal["call_tool"] = "call_tool"
    calls: list[ToolCallRequest]


class Finish(BaseModel):
    type: Literal["finish"] = "finish"
    output: str = ""
    reason: str = "completed"


class Handoff(BaseModel):
    """Extension point for Multi-Agent (P2). P0 emits the event and ends the turn."""

    type: Literal["handoff"] = "handoff"
    target: str
    payload: dict[str, Any] = Field(default_factory=dict)


Action = Annotated[Union[CallModel, CallTool, Finish, Handoff], Field(discriminator="type")]
