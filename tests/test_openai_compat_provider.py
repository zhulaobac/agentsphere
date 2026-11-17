from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from agentsphere.identity import InMemoryVault
from agentsphere.identity.agent import AgentIdentity
from agentsphere.kernel.context import SessionState
from agentsphere.kernel.events import EventType
from agentsphere.kernel.harness import AgentHarness
from agentsphere.kernel.loop.react import ReActStrategy
from agentsphere.model_adapter import (
    MockModelProvider,
    ModelAdapter,
    ModelAdapterError,
    OpenAICompatibleProvider,
    ScriptedHttpTransport,
    StdlibHttpTransport,
    assistant_text,
    resolve_model_provider,
)
from agentsphere.model_adapter.openai_compat import (
    ENV_AGENT_ID,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_SECRET_NAME,
    OpenAICompatHttpError,
    OpenAICompatNotConfigured,
    chat_completions_url,
    encode_chat_request,
    openai_chat_response,
    openai_sse_text,
    openai_sse_tool_call,
    openai_tool_call,
)
from agentsphere.model_adapter.types import (
    InferenceParams,
    Message,
    MessageRole,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)
from agentsphere.trace.tracer import TraceRecorder

SECRET = "test-offline-openai-token-not-a-real-key"
SECRET_NAME = "openai_api_key"
BASE_URL = "http://127.0.0.1:9/v1"
MODEL = "test-compat-model"


class _CapturingTransport:
    """Test wrapper: prove the bearer token reached HTTP without storing it on the provider."""

    def __init__(self, inner: ScriptedHttpTransport) -> None:
        self.inner = inner
        self.authorization: list[str] = []

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        stream: bool,
    ) -> Any:
        self.authorization.append(str(headers.get("Authorization", "")))
        return await self.inner.post(
            url,
            headers=headers,
            body=body,
            timeout_seconds=timeout_seconds,
            stream=stream,
        )


def _request(**kwargs: Any) -> ModelRequest:
    messages = kwargs.pop(
        "messages",
        [Message(role=MessageRole.USER, content="hello")],
    )
    return ModelRequest(messages=messages, **kwargs)


async def _vault(agent_id: str, secret: str = SECRET) -> InMemoryVault:
    vault = InMemoryVault()
    await vault.put(SECRET_NAME, secret, agent_id=agent_id, ttl_seconds=60)
    return vault


def _provider(
    vault: InMemoryVault,
    transport: Any,
    *,
    agent_id: str,
    base_url: str = BASE_URL,
    model: str = MODEL,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider.from_settings(
        base_url=base_url,
        model=model,
        secret_name=SECRET_NAME,
        agent_id=agent_id,
        vault=vault,
        transport=transport,
    )


def _assert_secret_absent(*values: Any) -> None:
    for value in values:
        blob = value if isinstance(value, str) else json.dumps(value, default=str)
        assert SECRET not in blob


async def test_resolve_model_provider_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    monkeypatch.delenv(ENV_SECRET_NAME, raising=False)
    provider = resolve_model_provider()
    assert isinstance(provider, MockModelProvider)
    assert provider.name == "mock"


async def test_live_provider_fails_closed_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    monkeypatch.delenv(ENV_SECRET_NAME, raising=False)
    monkeypatch.setenv(ENV_PROVIDER, "openai-compat")
    with pytest.raises(OpenAICompatNotConfigured):
        resolve_model_provider()
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    with pytest.raises(OpenAICompatNotConfigured, match="not configured"):
        resolve_model_provider(vault=vault, agent_id=identity.agent_id)
    monkeypatch.setenv(ENV_BASE_URL, "https://api.openai.com/v1")
    monkeypatch.setenv(ENV_MODEL, MODEL)
    monkeypatch.setenv(ENV_SECRET_NAME, SECRET_NAME)
    with pytest.raises(OpenAICompatNotConfigured, match="opt-in"):
        resolve_model_provider(vault=vault, agent_id=identity.agent_id, allow_network=False)


async def test_missing_or_empty_vault_secret_fails_closed() -> None:
    identity = AgentIdentity(name="compat-agent")
    empty = InMemoryVault()
    transport = ScriptedHttpTransport(script=[openai_chat_response(content="nope")])
    provider = _provider(empty, transport, agent_id=identity.agent_id)
    with pytest.raises(OpenAICompatNotConfigured, match="unavailable"):
        await provider.complete(_request())
    assert transport.calls == []

    vault = await _vault(identity.agent_id, secret="")
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    with pytest.raises(OpenAICompatNotConfigured, match="empty"):
        await provider.complete(_request())
    assert transport.calls == []


async def test_request_shaping_maps_messages_tools_and_params() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    transport = ScriptedHttpTransport(script=[openai_chat_response(content="ok")])
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    request = _request(
        messages=[
            Message(role=MessageRole.SYSTEM, content="system"),
            Message(role=MessageRole.USER, content="hello"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})],
            ),
            Message(role=MessageRole.TOOL, name="echo", tool_call_id="call_1", content="hi"),
        ],
        tools=[ToolDefinition(name="echo", description="echo text", parameters={"type": "object"})],
        params=InferenceParams(
            temperature=0.2,
            max_tokens=32,
            top_p=0.9,
            stop=["END"],
            extra={"tool_choice": "auto", "api_key": SECRET, "authorization": SECRET},
        ),
        model="override-model",
    )
    response = await provider.complete(request)
    assert response.content == "ok"
    assert response.provider == "openai-compat"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.url == "http://127.0.0.1:9/v1/chat/completions"
    assert call.stream is False
    assert call.had_authorization is True
    assert call.body["model"] == "override-model"
    assert call.body["stream"] is False
    assert call.body["temperature"] == 0.2
    assert call.body["max_tokens"] == 32
    assert call.body["top_p"] == 0.9
    assert call.body["stop"] == ["END"]
    assert call.body["tool_choice"] == "auto"
    assert "api_key" not in call.body
    assert "authorization" not in call.body
    assert call.body["messages"][2]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(call.body["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"text": "hi"}
    assert call.body["messages"][3]["role"] == "tool"
    assert call.body["messages"][3]["tool_call_id"] == "call_1"
    assert call.body["tools"][0]["type"] == "function"
    assert call.body["tools"][0]["function"]["name"] == "echo"
    _assert_secret_absent(call.model_dump(), call.headers, response.raw, response.model_dump())


async def test_tool_call_response_mapping() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    transport = ScriptedHttpTransport(
        script=[
            openai_chat_response(
                content=None,
                tool_calls=[openai_tool_call("echo", {"text": "x"}, call_id="call_echo")],
            )
        ]
    )
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    response = await provider.complete(_request(tools=[ToolDefinition(name="echo")]))
    assert response.has_tool_calls
    assert response.content is None
    assert response.tool_calls[0].id == "call_echo"
    assert response.tool_calls[0].name == "echo"
    assert response.tool_calls[0].arguments == {"text": "x"}


async def test_streaming_text_and_tool_calls() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    transport = ScriptedHttpTransport(
        script=[
            openai_sse_text(["ab", "cd"], usage={"prompt_tokens": 3, "completion_tokens": 2}),
            openai_sse_tool_call("echo", {"text": "stream"}),
        ]
    )
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    text_chunks = [chunk async for chunk in provider.stream(_request(params=InferenceParams(stream=True)))]
    assert "".join(chunk.delta for chunk in text_chunks) == "abcd"
    assert text_chunks[-1].done is True
    assert text_chunks[-1].response is not None
    assert text_chunks[-1].response.content == "abcd"
    assert text_chunks[-1].response.usage.input_tokens == 3
    assert transport.calls[0].stream is True
    assert transport.calls[0].body["stream"] is True
    assert transport.calls[0].body["stream_options"] == {"include_usage": True}

    tool_chunks = [chunk async for chunk in provider.stream(_request(tools=[ToolDefinition(name="echo")]))]
    final = tool_chunks[-1].response
    assert final is not None
    assert final.has_tool_calls
    assert final.tool_calls[0].name == "echo"
    assert final.tool_calls[0].arguments == {"text": "stream"}
    assert any(chunk.tool_call_delta and chunk.tool_call_delta.name == "echo" for chunk in tool_chunks)


async def test_adapter_fallback_on_vendor_http_error() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    from agentsphere.model_adapter.openai_compat import HttpResponse

    failing = ScriptedHttpTransport(
        script=[
            HttpResponse(
                status=500,
                body=json.dumps({"error": {"message": f"backend down {SECRET}"}}).encode(),
            )
        ]
    )
    primary = _provider(vault, failing, agent_id=identity.agent_id)
    fallback = MockModelProvider(script=[assistant_text("from fallback"), assistant_text("stream fallback")])
    adapter = ModelAdapter(primary, fallbacks=[fallback])
    response = await adapter.complete(_request())
    assert response.content == "from fallback"
    assert response.provider == "mock"
    _assert_secret_absent(response.model_dump())

    failing.reset(
        script=[
            HttpResponse(
                status=503,
                body=json.dumps({"error": {"message": f"stream down {SECRET}"}}).encode(),
            )
        ]
    )
    chunks = [chunk async for chunk in adapter.stream(_request(params=InferenceParams(stream=True)))]
    assert "".join(chunk.delta for chunk in chunks) == "stream fallback"
    assert chunks[-1].done is True


async def test_adapter_raises_when_openai_compat_and_fallback_fail() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    from agentsphere.model_adapter.openai_compat import HttpResponse

    transport = ScriptedHttpTransport(
        script=[HttpResponse(status=401, body=json.dumps({"error": {"message": "nope"}}).encode())]
    )
    adapter = ModelAdapter(
        _provider(vault, transport, agent_id=identity.agent_id),
        fallbacks=[MockModelProvider(script=[RuntimeError("fallback down")])],
    )
    with pytest.raises(ModelAdapterError):
        await adapter.complete(_request())


async def test_http_error_message_redacts_secret() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    from agentsphere.model_adapter.openai_compat import HttpResponse

    transport = ScriptedHttpTransport(
        script=[
            HttpResponse(
                status=401,
                body=json.dumps({"error": {"message": f"invalid key {SECRET}"}}).encode(),
            )
        ]
    )
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    with pytest.raises(OpenAICompatHttpError, match="invalid key") as info:
        await provider.complete(_request())
    assert SECRET not in str(info.value)
    assert SECRET not in repr(info.value)


async def test_secret_never_enters_provider_state_response_or_recorded_calls() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    inner = ScriptedHttpTransport(script=[openai_chat_response(content="hello")])
    transport = _CapturingTransport(inner)
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    response = await provider.complete(_request())
    assert transport.authorization == [f"Bearer {SECRET}"]
    assert inner.calls[0].headers["Authorization"] == "[redacted]"
    assert SECRET not in repr(provider)
    for key, value in provider.__dict__.items():
        if key == "_vault":
            continue
        _assert_secret_absent({key: str(value)})
    _assert_secret_absent(
        response.model_dump(),
        response.raw,
        inner.calls[0].model_dump(),
        inner.calls[0].headers,
    )


async def test_secret_never_enters_harness_events_or_traces() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    transport = ScriptedHttpTransport(script=[openai_chat_response(content="hello from vendor")])
    provider = _provider(vault, transport, agent_id=identity.agent_id)
    tracer = TraceRecorder()
    harness = AgentHarness(ReActStrategy(), ModelAdapter(provider), max_steps=4)
    harness.event_emitter.add_listener(tracer)
    events = [event async for event in harness.run_turn(SessionState(session_id="s-openai"), "hello")]
    blob = " ".join(event.model_dump_json() for event in events)
    assert any(event.type is EventType.MODEL_CALL_COMPLETED for event in events)
    assert SECRET not in blob
    tree = tracer.inspect("s-openai")
    assert SECRET not in tree.model_dump_json()
    messages_blob = " ".join(
        json.dumps(event.payload, default=str) for event in events if event.type is EventType.MODEL_CALL_COMPLETED
    )
    assert "hello from vendor" in messages_blob
    _assert_secret_absent(messages_blob)


async def test_transport_exception_redacts_secret() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)

    class Boom:
        async def post(self, url: str, *, headers: Mapping[str, str], body: bytes, timeout_seconds: float, stream: bool):
            del url, body, timeout_seconds, stream
            raise RuntimeError(f"socket failed for {headers.get('Authorization')}")

    provider = _provider(vault, Boom(), agent_id=identity.agent_id)
    with pytest.raises(OpenAICompatHttpError) as info:
        await provider.complete(_request())
    assert SECRET not in str(info.value)
    assert "Bearer [redacted]" in str(info.value) or "[redacted]" in str(info.value)


async def test_stdlib_transport_refuses_public_hosts_without_opt_in() -> None:
    transport = StdlibHttpTransport()
    with pytest.raises(OpenAICompatNotConfigured, match="allow_public_network"):
        await transport.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=1,
            stream=False,
        )


async def test_chat_completions_url_joins_and_rejects_bad_schemes() -> None:
    assert chat_completions_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert (
        chat_completions_url("https://api.openai.com/v1/chat/completions")
        == "https://api.openai.com/v1/chat/completions"
    )
    with pytest.raises(OpenAICompatNotConfigured):
        chat_completions_url("file:///tmp/secrets")


async def test_encode_chat_request_omits_empty_tools() -> None:
    from agentsphere.model_adapter.openai_compat import OpenAICompatConfig

    body = encode_chat_request(
        _request(),
        OpenAICompatConfig(base_url=BASE_URL, model=MODEL, secret_name=SECRET_NAME, agent_id="agt"),
        stream=False,
    )
    assert "tools" not in body
    assert body["model"] == MODEL


class _FakeVendor(ThreadingHTTPServer):
    secret = SECRET
    mode = "complete"
    captured_auth: list[str]


def _start_fake_vendor(*, mode: str) -> tuple[_FakeVendor, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            server: _FakeVendor = self.server  # type: ignore[assignment]
            server.captured_auth.append(self.headers.get("Authorization") or "")
            body = json.loads(raw.decode("utf-8")) if raw else {}
            assert "api_key" not in body
            if server.mode == "stream":
                payload_lines = openai_sse_text(["lo", "cal"])
                blob = "\n".join(payload_lines) + "\n"
                data = blob.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            payload = json.dumps(openai_chat_response(content="from-loopback")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

    server = _FakeVendor(("127.0.0.1", 0), Handler)
    server.mode = mode
    server.captured_auth = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/v1"


async def test_stdlib_transport_against_local_fake_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    server, base_url = _start_fake_vendor(mode="complete")
    try:
        provider = _provider(
            vault,
            StdlibHttpTransport(honor_proxy=False),
            agent_id=identity.agent_id,
            base_url=base_url,
        )
        response = await provider.complete(_request())
        assert response.content == "from-loopback"
        assert server.captured_auth == [f"Bearer {SECRET}"]
        _assert_secret_absent(response.model_dump(), response.raw, repr(provider))
    finally:
        server.shutdown()
        server.server_close()


async def test_stdlib_stream_against_local_fake_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    server, base_url = _start_fake_vendor(mode="stream")
    try:
        provider = _provider(
            vault,
            StdlibHttpTransport(honor_proxy=False),
            agent_id=identity.agent_id,
            base_url=base_url,
        )
        chunks = [chunk async for chunk in provider.stream(_request(params=InferenceParams(stream=True)))]
        assert "".join(chunk.delta for chunk in chunks) == "local"
        assert chunks[-1].done is True
        assert server.captured_auth == [f"Bearer {SECRET}"]
        _assert_secret_absent(chunks[-1].response.model_dump() if chunks[-1].response else {})
    finally:
        server.shutdown()
        server.server_close()


async def test_resolve_model_provider_live_with_injected_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    monkeypatch.setenv(ENV_PROVIDER, "openai-compat")
    monkeypatch.setenv(ENV_BASE_URL, BASE_URL)
    monkeypatch.setenv(ENV_MODEL, MODEL)
    monkeypatch.setenv(ENV_SECRET_NAME, SECRET_NAME)
    monkeypatch.setenv(ENV_AGENT_ID, identity.agent_id)
    transport = ScriptedHttpTransport(script=[openai_chat_response(content="via-factory")])
    provider = resolve_model_provider(vault=vault, transport=transport)
    assert isinstance(provider, OpenAICompatibleProvider)
    response = await provider.complete(_request())
    assert response.content == "via-factory"


async def test_construction_requires_transport_and_vault() -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    with pytest.raises(OpenAICompatNotConfigured, match="HTTP transport"):
        OpenAICompatibleProvider.from_settings(
            base_url=BASE_URL,
            model=MODEL,
            secret_name=SECRET_NAME,
            agent_id=identity.agent_id,
            vault=vault,
            transport=None,  # type: ignore[arg-type]
        )
    with pytest.raises(OpenAICompatNotConfigured, match="Credential Vault"):
        OpenAICompatibleProvider.from_settings(
            base_url=BASE_URL,
            model=MODEL,
            secret_name=SECRET_NAME,
            agent_id=identity.agent_id,
            vault=None,  # type: ignore[arg-type]
            transport=ScriptedHttpTransport(),
        )


async def test_factory_live_transport_is_opt_in_and_does_not_call_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = AgentIdentity(name="compat-agent")
    vault = await _vault(identity.agent_id)
    monkeypatch.setenv(ENV_PROVIDER, "openai-compat")
    monkeypatch.setenv(ENV_BASE_URL, "https://api.openai.com/v1")
    monkeypatch.setenv(ENV_MODEL, MODEL)
    monkeypatch.setenv(ENV_SECRET_NAME, SECRET_NAME)
    provider = resolve_model_provider(vault=vault, agent_id=identity.agent_id, allow_network=True)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert isinstance(provider._transport, StdlibHttpTransport)
    assert provider._transport.allow_public_network is True
