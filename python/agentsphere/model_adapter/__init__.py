"""Model Adapter module.

Unify message format, tool calling, streaming, and inference params.
Providers (including the in-process mock) plug in here without changing the Kernel.
"""

from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.model_adapter.mock import (
    MockModelProvider,
    assistant_text,
    assistant_tool_response,
    tool_call,
)
from agentsphere.model_adapter.provider import FallbackHook, ModelAdapterError, ModelProvider
from agentsphere.model_adapter.types import (
    InferenceParams,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "FallbackHook",
    "InferenceParams",
    "Message",
    "MessageRole",
    "MockModelProvider",
    "ModelAdapter",
    "ModelAdapterError",
    "ModelCapabilities",
    "ModelChunk",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "assistant_text",
    "assistant_tool_response",
    "tool_call",
]
