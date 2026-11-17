"""Model Adapter module.

Unify message format, tool calling, streaming, and inference params.
Providers (mock and OpenAI-compatible Chat Completions) plug in here without
changing the Kernel. The live vendor path is opt-in and off by default.
"""

from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.model_adapter.mock import (
    MockModelProvider,
    assistant_text,
    assistant_tool_response,
    tool_call,
)
from agentsphere.model_adapter.openai_compat import (
    OpenAICompatConfig,
    OpenAICompatError,
    OpenAICompatHttpError,
    OpenAICompatNotConfigured,
    OpenAICompatibleProvider,
    ScriptedHttpTransport,
    StdlibHttpTransport,
    resolve_model_provider,
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
    "OpenAICompatConfig",
    "OpenAICompatError",
    "OpenAICompatHttpError",
    "OpenAICompatNotConfigured",
    "OpenAICompatibleProvider",
    "ScriptedHttpTransport",
    "StdlibHttpTransport",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "assistant_text",
    "assistant_tool_response",
    "resolve_model_provider",
    "tool_call",
]
