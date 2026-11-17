from __future__ import annotations

import pytest

from agentsphere.model_adapter import (
    MockModelProvider,
    ModelAdapter,
    ModelAdapterError,
    assistant_text,
    assistant_tool_response,
)
from agentsphere.model_adapter.types import (
    InferenceParams,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelRequest,
    ToolDefinition,
)


def _request(*, tools: bool = False, stream: bool = False) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=MessageRole.USER, content="hello")],
        tools=[ToolDefinition(name="echo", description="echo")] if tools else [],
        params=InferenceParams(stream=stream),
    )


@pytest.mark.asyncio
async def test_mock_provider_serves_script_without_network() -> None:
    provider = MockModelProvider(script=[assistant_text("hi"), assistant_tool_response("echo", {"text": "x"})])
    first = await provider.complete(_request())
    assert first.content == "hi"
    second = await provider.complete(_request())
    assert second.has_tool_calls
    assert second.tool_calls[0].name == "echo"


@pytest.mark.asyncio
async def test_mock_stream_yields_deltas() -> None:
    provider = MockModelProvider(script=[assistant_text("abcd")])
    chunks = [c async for c in provider.stream(_request())]
    assert "".join(c.delta for c in chunks) == "abcd"
    assert chunks[-1].done is True
    assert chunks[-1].response is not None


@pytest.mark.asyncio
async def test_adapter_falls_back_when_primary_raises() -> None:
    primary = MockModelProvider(script=[RuntimeError("primary down")])
    fallback = MockModelProvider(script=[assistant_text("from fallback")])
    adapter = ModelAdapter(primary, fallbacks=[fallback])
    response = await adapter.complete(_request())
    assert response.content == "from fallback"
    assert response.provider == "mock"


@pytest.mark.asyncio
async def test_adapter_skips_provider_missing_tool_capability() -> None:
    no_tools = MockModelProvider(
        script=[assistant_text("should not run")],
        capabilities=ModelCapabilities(supports_tools=False, context_window=1024, max_output_tokens=128),
    )
    with_tools = MockModelProvider(script=[assistant_text("tools ok")])
    adapter = ModelAdapter(no_tools, fallbacks=[with_tools])
    response = await adapter.complete(_request(tools=True))
    assert response.content == "tools ok"
    assert len(no_tools.requests) == 0
    assert len(with_tools.requests) == 1


@pytest.mark.asyncio
async def test_adapter_degrades_by_dropping_tools_when_nothing_supports_them() -> None:
    no_tools = MockModelProvider(
        script=[assistant_text("degraded")],
        capabilities=ModelCapabilities(supports_tools=False, context_window=1024, max_output_tokens=128),
    )
    adapter = ModelAdapter(no_tools)
    response = await adapter.complete(_request(tools=True))
    assert response.content == "degraded"
    assert no_tools.requests[0].tools == []


@pytest.mark.asyncio
async def test_adapter_raises_when_all_fail() -> None:
    adapter = ModelAdapter(
        MockModelProvider(script=[RuntimeError("a")]),
        fallbacks=[MockModelProvider(script=[RuntimeError("b")])],
    )
    with pytest.raises(ModelAdapterError):
        await adapter.complete(_request())


def test_capability_detection_unions_chain() -> None:
    primary = MockModelProvider(
        capabilities=ModelCapabilities(
            supports_tools=False,
            supports_streaming=True,
            context_window=100,
            max_output_tokens=10,
        )
    )
    fallback = MockModelProvider(
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_streaming=False,
            context_window=8000,
            max_output_tokens=256,
        )
    )
    caps = ModelAdapter(primary, fallbacks=[fallback]).capabilities()
    assert caps.supports_tools is True
    assert caps.supports_streaming is True
    assert caps.context_window == 8000
