"""Canonical model types shared by the adapter and the Harness Kernel."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDefinition(BaseModel):
    """Provider-agnostic tool schema presented to a model."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    untrusted: bool = False


class InferenceParams(BaseModel):
    """Unified inference parameters. Extra provider-specific keys go in `extra`."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelCapabilities(BaseModel):
    supports_tools: bool = False
    supports_parallel_tools: bool = False
    supports_streaming: bool = False
    supports_thinking: bool = False
    context_window: int = 0
    max_output_tokens: int = 0


class ModelRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    params: InferenceParams = Field(default_factory=InferenceParams)
    model: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    provider: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ModelChunk(BaseModel):
    """Streaming delta. The final chunk may carry the completed `response`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delta: str = ""
    done: bool = False
    tool_call_delta: ToolCall | None = None
    response: ModelResponse | None = None
