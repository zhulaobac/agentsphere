"""Fake provider for tests and local runs. No network, no API keys."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from agentsphere.model_adapter.types import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    ToolCall,
)

ScriptItem = ModelResponse | BaseException | Callable[[ModelRequest], ModelResponse]


class MockModelProvider:
    """Scripted model. Responses are served in order; callables receive the request."""

    name = "mock"

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        capabilities: ModelCapabilities | None = None,
        default_content: str = "ok",
    ) -> None:
        self._script: list[ScriptItem] = list(script or [])
        self._cursor = 0
        self.requests: list[ModelRequest] = []
        self._capabilities = capabilities or ModelCapabilities(
            supports_tools=True,
            supports_parallel_tools=True,
            supports_streaming=True,
            supports_thinking=False,
            context_window=128_000,
            max_output_tokens=4_096,
        )
        self._default_content = default_content

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def reset(self, script: Sequence[ScriptItem] | None = None) -> None:
        if script is not None:
            self._script = list(script)
        self._cursor = 0
        self.requests.clear()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, ModelResponse):
            item = item(request)
        if not item.provider:
            item = item.model_copy(update={"provider": self.name})
        return item

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        response = await self.complete(request)
        text = response.content or ""
        if text:
            mid = max(1, len(text) // 2)
            yield ModelChunk(delta=text[:mid], done=False)
            if text[mid:]:
                yield ModelChunk(delta=text[mid:], done=False)
        yield ModelChunk(delta="", done=True, response=response)

    def _next(self) -> ScriptItem:
        if self._cursor >= len(self._script):
            return ModelResponse(content=self._default_content, provider=self.name)
        item = self._script[self._cursor]
        self._cursor += 1
        return item


def tool_call(name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolCall:
    return ToolCall(id=call_id or f"call_{uuid.uuid4().hex}", name=name, arguments=arguments)


def assistant_tool_response(name: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(content=None, tool_calls=[tool_call(name, arguments)], provider="mock")


def assistant_text(content: str) -> ModelResponse:
    return ModelResponse(content=content, provider="mock")
