"""Pure Kernel state. Rebuilt by Runtime from the event log; never stored in-process."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.model_adapter.types import Message, ModelResponse, ToolDefinition


class SessionState(BaseModel):
    """Snapshot the Kernel consumes. Runtime owns durability; Kernel is (state -> action)."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    turn_index: int = 0
    system_prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_pins: dict[str, str] = Field(default_factory=dict)
    workflow_cursor: str | None = None
    workflow_node_outputs: dict[str, Any] = Field(default_factory=dict)


class AssembledContext(BaseModel):
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    truncated: dict[str, bool] = Field(default_factory=dict)


class TurnContext(BaseModel):
    """Per-step view handed to LoopStrategy.step."""

    session_state: SessionState
    user_input: str
    turn_id: str
    step_index: int = 0
    assembled: AssembledContext | None = None
    last_model_response: ModelResponse | None = None
    last_tool_results: list[dict[str, Any]] | None = None
    pending_tool_calls: list[ToolCallRequest] | None = None
    last_action_result: dict[str, Any] | None = None
