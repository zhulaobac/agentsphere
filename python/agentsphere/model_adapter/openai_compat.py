"""OpenAI-compatible Chat Completions provider behind ``ModelProvider``.

Default is **off**. Official tests inject ``ScriptedHttpTransport`` or talk to a
loopback fake server. A live vendor is opt-in: pass ``StdlibHttpTransport`` with
``allow_public_network=True`` and load the API key from the Credential Vault at
call time. Keys never belong in env, model context, events, traces, or git.

Live smoke (not run in CI)::

    AGENTSPHERE_MODEL_PROVIDER=openai-compat
    AGENTSPHERE_OPENAI_BASE_URL=https://api.openai.com/v1
    AGENTSPHERE_OPENAI_MODEL=gpt-4o-mini
    AGENTSPHERE_OPENAI_SECRET_NAME=openai_api_key

    vault.put("openai_api_key", key, agent_id=agent.agent_id)
    provider = resolve_model_provider(vault=vault, agent_id=agent.agent_id, allow_network=True)
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from agentsphere.model_adapter.mock import MockModelProvider
from agentsphere.model_adapter.provider import ModelProvider
from agentsphere.model_adapter.types import (
    Message,
    MessageRole,
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)

if TYPE_CHECKING:
    from agentsphere.identity.vault import Vault

ENV_PROVIDER = "AGENTSPHERE_MODEL_PROVIDER"
ENV_BASE_URL = "AGENTSPHERE_OPENAI_BASE_URL"
ENV_MODEL = "AGENTSPHERE_OPENAI_MODEL"
ENV_SECRET_NAME = "AGENTSPHERE_OPENAI_SECRET_NAME"
ENV_AGENT_ID = "AGENTSPHERE_OPENAI_AGENT_ID"

DEFAULT_CHAT_PATH = "/chat/completions"
_BLOCKED_EXTRA_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "bearer",
    "token",
    "access_token",
    "secret",
    "password",
    "credential",
    "private_key",
    "openai_api_key",
}


class OpenAICompatError(RuntimeError):
    """Vendor or mapping failure. Messages must never contain vault secrets."""

    retryable: bool = False


class OpenAICompatNotConfigured(OpenAICompatError):
    """Missing vault, secret, base URL, model, or HTTP transport. Fail closed."""


class OpenAICompatHttpError(OpenAICompatError):
    def __init__(self, message: str, *, status: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class OpenAICompatConfig(BaseModel):
    """Non-secret vendor settings. The API key is a vault secret name, not a value."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    secret_name: str
    agent_id: str
    timeout_seconds: float = 60.0
    chat_completions_path: str = DEFAULT_CHAT_PATH
    context_window: int = 128_000
    max_output_tokens: int = 16_384
    provider_name: str = "openai-compat"


class HttpResponse:
    """Transport result. ``headers`` must already be safe to inspect (redact auth)."""

    def __init__(
        self,
        *,
        status: int,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
        lines: Sequence[bytes] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.headers = dict(headers or {})
        self.lines = list(lines or [])

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if not self.body:
            return {}
        return json.loads(self.body)


class HttpTransport(Protocol):
    """Injected HTTP seam. Tests bind a scripted transport; live binds stdlib."""

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        stream: bool,
    ) -> HttpResponse: ...


class RecordedHttpCall(BaseModel):
    """Safe request snapshot. Authorization is present/absent, never the secret."""

    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    had_authorization: bool = False


class ScriptedHttpTransport:
    """Offline Chat Completions stand-in. No sockets, no real keys."""

    def __init__(
        self,
        script: Sequence[HttpResponse | BaseException | dict[str, Any] | list[str]] | None = None,
    ) -> None:
        self._script: list[HttpResponse | BaseException | dict[str, Any] | list[str]] = list(script or [])
        self._cursor = 0
        self.calls: list[RecordedHttpCall] = []

    def reset(
        self,
        script: Sequence[HttpResponse | BaseException | dict[str, Any] | list[str]] | None = None,
    ) -> None:
        if script is not None:
            self._script = list(script)
        self._cursor = 0
        self.calls.clear()

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        stream: bool,
    ) -> HttpResponse:
        del timeout_seconds
        parsed: dict[str, Any]
        try:
            parsed = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            parsed = {"_raw": body.decode("utf-8", errors="replace")}
        self.calls.append(
            RecordedHttpCall(
                url=url,
                headers=_public_headers(headers),
                body=parsed,
                stream=stream,
                had_authorization=_has_authorization(headers),
            )
        )
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, HttpResponse):
            return item
        if isinstance(item, list):
            encoded = [line.encode("utf-8") if isinstance(line, str) else line for line in item]
            return HttpResponse(status=200, lines=encoded, headers={"content-type": "text/event-stream"})
        payload = json.dumps(item).encode("utf-8")
        return HttpResponse(status=200, body=payload, headers={"content-type": "application/json"})

    def _next(self) -> HttpResponse | BaseException | dict[str, Any] | list[str]:
        if self._cursor >= len(self._script):
            raise OpenAICompatNotConfigured("scripted HTTP transport has no remaining responses")
        item = self._script[self._cursor]
        self._cursor += 1
        return item


class StdlibHttpTransport:
    """stdlib ``urllib`` transport. Loopback only unless ``allow_public_network``."""

    def __init__(self, *, allow_public_network: bool = False, honor_proxy: bool = True) -> None:
        self.allow_public_network = allow_public_network
        self.honor_proxy = honor_proxy

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        stream: bool,
    ) -> HttpResponse:
        self._assert_url_allowed(url)
        if stream:
            return await asyncio.to_thread(
                self._post_stream_sync, url, dict(headers), body, timeout_seconds
            )
        return await asyncio.to_thread(
            self._post_sync, url, dict(headers), body, timeout_seconds
        )

    def _assert_url_allowed(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise OpenAICompatNotConfigured("OpenAI-compatible URL must be http or https")
        if self.allow_public_network:
            return
        if not _is_loopback_host(parsed.hostname):
            raise OpenAICompatNotConfigured(
                "stdlib transport refuses non-loopback hosts unless allow_public_network=True"
            )

    def _opener(self) -> urllib.request.OpenerDirector:
        if self.honor_proxy:
            return urllib.request.build_opener()
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, url: str, headers: dict[str, str], body: bytes) -> urllib.request.Request:
        request = urllib.request.Request(url, data=body, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        return request

    def _post_sync(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> HttpResponse:
        opener = self._opener()
        try:
            with opener.open(self._request(url, headers, body), timeout=timeout_seconds) as response:
                return HttpResponse(
                    status=getattr(response, "status", 200),
                    body=response.read(),
                    headers=_public_headers(dict(response.headers)),
                )
        except urllib.error.HTTPError as exc:
            payload = exc.read() if exc.fp is not None else b""
            return HttpResponse(
                status=int(exc.code),
                body=payload,
                headers=_public_headers(dict(exc.headers or {})),
            )

    def _post_stream_sync(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> HttpResponse:
        opener = self._opener()
        try:
            with opener.open(self._request(url, headers, body), timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    return HttpResponse(
                        status=status,
                        body=response.read(),
                        headers=_public_headers(dict(response.headers)),
                    )
                lines = [line.rstrip(b"\r\n") for line in response]
                return HttpResponse(
                    status=status,
                    lines=lines,
                    headers=_public_headers(dict(response.headers)),
                )
        except urllib.error.HTTPError as exc:
            payload = exc.read() if exc.fp is not None else b""
            return HttpResponse(
                status=int(exc.code),
                body=payload,
                headers=_public_headers(dict(exc.headers or {})),
            )


class OpenAICompatibleProvider:
    """Chat Completions backend. Kernel still talks only to ``ModelAdapter``."""

    def __init__(
        self,
        config: OpenAICompatConfig,
        *,
        vault: Vault,
        transport: HttpTransport,
    ) -> None:
        _validate_config(config)
        if vault is None:
            raise OpenAICompatNotConfigured("OpenAI-compatible provider requires a Credential Vault")
        if transport is None:
            raise OpenAICompatNotConfigured(
                "OpenAI-compatible provider requires an HTTP transport; live HTTP is opt-in"
            )
        self._config = config
        self._vault = vault
        self._transport = transport
        self.name = config.provider_name
        self._capabilities = ModelCapabilities(
            supports_tools=True,
            supports_parallel_tools=True,
            supports_streaming=True,
            supports_thinking=False,
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
        )

    @classmethod
    def from_settings(
        cls,
        *,
        base_url: str,
        model: str,
        secret_name: str,
        agent_id: str,
        vault: Vault,
        transport: HttpTransport,
        timeout_seconds: float = 60.0,
        provider_name: str = "openai-compat",
    ) -> OpenAICompatibleProvider:
        return cls(
            OpenAICompatConfig(
                base_url=base_url,
                model=model,
                secret_name=secret_name,
                agent_id=agent_id,
                timeout_seconds=timeout_seconds,
                provider_name=provider_name,
            ),
            vault=vault,
            transport=transport,
        )

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleProvider(name={self.name!r}, base_url={self._config.base_url!r}, "
            f"model={self._config.model!r}, secret_name={self._config.secret_name!r})"
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        body = encode_chat_request(request, self._config, stream=False)
        response = await self._send(body, stream=False)
        return decode_chat_response(response, provider=self.name)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        body = encode_chat_request(request, self._config, stream=True)
        response = await self._send(body, stream=True)
        async for chunk in decode_chat_stream(response, provider=self.name):
            yield chunk

    async def _send(self, body: dict[str, Any], *, stream: bool) -> HttpResponse:
        url = chat_completions_url(self._config.base_url, self._config.chat_completions_path)
        payload = json.dumps(body).encode("utf-8")
        secret: str | None = None
        headers: dict[str, str] = {}
        try:
            secret = await self._reveal_secret()
            headers = {
                "Authorization": f"Bearer {secret}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "User-Agent": "agentsphere-openai-compat/0.1",
            }
            result = await self._transport.post(
                url,
                headers=headers,
                body=payload,
                timeout_seconds=self._config.timeout_seconds,
                stream=stream,
            )
            if result.status >= 400:
                raise _http_error(result)
            return result
        except OpenAICompatError as exc:
            raise _redact_exception(exc, secret) from None
        except Exception as exc:  # noqa: BLE001 — never leak the bearer token
            raise OpenAICompatHttpError(_safe_text(str(exc), secret), retryable=True) from None
        finally:
            headers.clear()
            secret = None

    async def _reveal_secret(self) -> str:
        from agentsphere.identity.vault import VaultError

        try:
            issued = await self._vault.issue(
                self._config.secret_name,
                agent_id=self._config.agent_id,
                ttl_seconds=60,
            )
            secret = await self._vault.reveal(issued.token)
        except VaultError as exc:
            raise OpenAICompatNotConfigured(
                f"credential {self._config.secret_name!r} is unavailable"
            ) from exc
        if not secret:
            raise OpenAICompatNotConfigured(f"credential {self._config.secret_name!r} is empty")
        return secret


def resolve_model_provider(
    kind: str | None = None,
    *,
    vault: Vault | None = None,
    agent_id: str | None = None,
    transport: HttpTransport | None = None,
    allow_network: bool = False,
    mock: ModelProvider | None = None,
) -> ModelProvider:
    """Default is the in-process mock. ``openai-compat`` fails closed if unconfigured."""

    selected = (kind or os.environ.get(ENV_PROVIDER) or "mock").strip().lower()
    if selected in {"mock", "offline", ""}:
        return mock if mock is not None else MockModelProvider()
    if selected in {"openai", "openai-compat", "openai_compat"}:
        return _live_openai_compat(
            vault=vault,
            agent_id=agent_id,
            transport=transport,
            allow_network=allow_network,
        )
    raise ValueError(f"unknown model provider {selected!r}; use 'mock' or 'openai-compat'")


def config_from_env(*, agent_id: str | None = None) -> OpenAICompatConfig:
    base_url = (os.environ.get(ENV_BASE_URL) or "").strip()
    model = (os.environ.get(ENV_MODEL) or "").strip()
    secret_name = (os.environ.get(ENV_SECRET_NAME) or "").strip()
    resolved_agent = (agent_id or os.environ.get(ENV_AGENT_ID) or "").strip()
    missing = [
        name
        for name, value in (
            (ENV_BASE_URL, base_url),
            (ENV_MODEL, model),
            (ENV_SECRET_NAME, secret_name),
        )
        if not value
    ]
    if not resolved_agent:
        missing.append("agent_id")
    if missing:
        raise OpenAICompatNotConfigured(
            "openai-compat is not configured; missing " + ", ".join(missing)
        )
    return OpenAICompatConfig(
        base_url=base_url,
        model=model,
        secret_name=secret_name,
        agent_id=resolved_agent,
    )


def chat_completions_url(base_url: str, path: str = DEFAULT_CHAT_PATH) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise OpenAICompatNotConfigured("base URL is empty")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"}:
        raise OpenAICompatNotConfigured("base URL must be http or https")
    suffix = path if path.startswith("/") else f"/{path}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


def encode_chat_request(
    request: ModelRequest,
    config: OpenAICompatConfig,
    *,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model or config.model,
        "messages": [_encode_message(message) for message in request.messages],
        "stream": stream,
    }
    if request.tools:
        body["tools"] = [_encode_tool(tool) for tool in request.tools]
    params = request.params
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if params.max_tokens is not None:
        body["max_tokens"] = params.max_tokens
    if params.top_p is not None:
        body["top_p"] = params.top_p
    if params.stop:
        body["stop"] = list(params.stop)
    if stream:
        extra_stream = params.extra.get("stream_options") if isinstance(params.extra, dict) else None
        body["stream_options"] = extra_stream or {"include_usage": True}
    for key, value in (params.extra or {}).items():
        if key in body or value is None:
            continue
        if key == "stream_options" and stream:
            continue
        if _is_blocked_extra(key):
            continue
        body[key] = value
    return body


def decode_chat_response(response: HttpResponse, *, provider: str) -> ModelResponse:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise OpenAICompatError("vendor returned non-JSON Chat Completions body") from exc
    if not isinstance(payload, dict):
        raise OpenAICompatError("vendor returned a non-object Chat Completions body")
    error = payload.get("error")
    if error:
        raise OpenAICompatHttpError(_error_message(error, response.status), status=response.status)
    choices = payload.get("choices") or []
    if not choices:
        raise OpenAICompatError("vendor returned no choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    tool_calls = [_decode_tool_call(item) for item in message.get("tool_calls") or []]
    usage = _decode_usage(payload.get("usage") or {})
    return ModelResponse(
        content=content if isinstance(content, str) else (None if tool_calls else ""),
        tool_calls=tool_calls,
        usage=usage,
        provider=provider,
        raw=_safe_raw(payload),
    )


async def decode_chat_stream(
    response: HttpResponse,
    *,
    provider: str,
) -> AsyncIterator[ModelChunk]:
    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    usage = TokenUsage()
    raw_chunks: list[dict[str, Any]] = []
    finish_reason: str | None = None
    iterator: Iterator[bytes]
    if response.lines:
        iterator = iter(response.lines)
    else:
        iterator = iter(response.body.splitlines())
    for data in _iter_sse_data(iterator):
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OpenAICompatError("vendor returned non-JSON SSE chunk") from exc
        if not isinstance(payload, dict):
            continue
        raw_chunks.append(_safe_raw(payload))
        if payload.get("error"):
            raise OpenAICompatHttpError(_error_message(payload.get("error"), response.status), status=response.status)
        usage_payload = payload.get("usage")
        if usage_payload:
            usage = _decode_usage(usage_payload)
        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or choice.get("message") or {}
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            content_parts.append(piece)
            yield ModelChunk(delta=piece, done=False)
        for item in delta.get("tool_calls") or []:
            _accumulate_tool_delta(tool_acc, item)
    tool_calls = [
        ToolCall(
            id=slot.get("id") or f"call_{index}",
            name=slot.get("name") or "",
            arguments=_parse_arguments(slot.get("arguments") or ""),
        )
        for index, slot in sorted(tool_acc.items())
    ]
    content = "".join(content_parts)
    final = ModelResponse(
        content=content if content else (None if tool_calls else ""),
        tool_calls=tool_calls,
        usage=usage,
        provider=provider,
        raw={"chunks": raw_chunks, "finish_reason": finish_reason},
    )
    for call in tool_calls:
        yield ModelChunk(delta="", done=False, tool_call_delta=call)
    yield ModelChunk(delta="", done=True, response=final)


def openai_chat_response(
    *,
    content: str | None = None,
    tool_calls: Sequence[dict[str, Any]] | None = None,
    usage: Mapping[str, int] | None = None,
    model: str = "test-model",
) -> dict[str, Any]:
    """Build a Chat Completions JSON body for the scripted transport."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
        if content is None:
            message["content"] = None
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": int((usage or {}).get("prompt_tokens", 1)),
            "completion_tokens": int((usage or {}).get("completion_tokens", 1)),
        },
    }


def openai_tool_call(name: str, arguments: Mapping[str, Any], call_id: str = "call_test") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def openai_sse_text(parts: Sequence[str], *, usage: Mapping[str, int] | None = None) -> list[str]:
    lines = [
        _sse_line({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
    ]
    for part in parts:
        lines.append(_sse_line({"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}))
    lines.append(_sse_line({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    if usage:
        lines.append(
            _sse_line(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                    },
                }
            )
        )
    lines.append("data: [DONE]")
    return lines


def openai_sse_tool_call(
    name: str,
    arguments: Mapping[str, Any],
    call_id: str = "call_stream",
) -> list[str]:
    raw = json.dumps(dict(arguments))
    mid = max(1, len(raw) // 2)
    lines = [
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[:mid]}}]},
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[mid:]}}]},
                    }
                ]
            }
        ),
        _sse_line({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    return lines


def _live_openai_compat(
    *,
    vault: Vault | None,
    agent_id: str | None,
    transport: HttpTransport | None,
    allow_network: bool,
) -> OpenAICompatibleProvider:
    if vault is None:
        raise OpenAICompatNotConfigured("openai-compat requires a Credential Vault")
    config = config_from_env(agent_id=agent_id)
    bound = transport
    if bound is None:
        if not allow_network:
            raise OpenAICompatNotConfigured(
                "openai-compat live HTTP is opt-in; pass a transport or allow_network=True"
            )
        bound = StdlibHttpTransport(allow_public_network=True)
    return OpenAICompatibleProvider(config, vault=vault, transport=bound)


def _validate_config(config: OpenAICompatConfig) -> None:
    if not (config.base_url or "").strip():
        raise OpenAICompatNotConfigured("base URL is empty")
    if not (config.model or "").strip():
        raise OpenAICompatNotConfigured("model name is empty")
    if not (config.secret_name or "").strip():
        raise OpenAICompatNotConfigured("vault secret name is empty")
    if not (config.agent_id or "").strip():
        raise OpenAICompatNotConfigured("agent_id is empty")
    chat_completions_url(config.base_url, config.chat_completions_path)


def _encode_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value}
    if message.role is MessageRole.TOOL:
        payload["content"] = message.content or ""
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
        return payload
    if message.tool_calls:
        payload["content"] = message.content
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
        return payload
    payload["content"] = message.content or ""
    return payload


def _encode_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }


def _decode_tool_call(item: Mapping[str, Any]) -> ToolCall:
    function = item.get("function") or {}
    return ToolCall(
        id=str(item.get("id") or ""),
        name=str(function.get("name") or item.get("name") or ""),
        arguments=_parse_arguments(function.get("arguments") or item.get("arguments") or {}),
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    if not isinstance(raw, str):
        return {"_raw": raw}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def _decode_usage(payload: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0),
        output_tokens=int(payload.get("completion_tokens") or payload.get("output_tokens") or 0),
    )


def _accumulate_tool_delta(acc: dict[int, dict[str, str]], item: Mapping[str, Any]) -> None:
    index = int(item.get("index") or 0)
    slot = acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if item.get("id"):
        slot["id"] = str(item["id"])
    function = item.get("function") or {}
    if function.get("name"):
        slot["name"] += str(function["name"])
    if function.get("arguments"):
        slot["arguments"] += str(function["arguments"])


def _iter_sse_data(lines: Iterator[bytes]) -> Iterator[str]:
    """Yield Chat Completions SSE payloads.

    OpenAI-compatible vendors send one JSON object per ``data:`` line. Blank
    lines are ignored so both scripted transports and loopback servers work.
    """

    pending: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.rstrip("\r")
        if line == "":
            data = "\n".join(pending).strip()
            pending = []
            if data:
                yield data
            continue
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
            continue
        if line.startswith("data:"):
            piece = line[5:].lstrip()
            if piece.startswith("{") or piece.startswith("[") or piece == "[DONE]":
                extra = "\n".join(pending).strip()
                pending = []
                if extra:
                    yield extra
                if piece:
                    yield piece
            elif piece:
                pending.append(piece)
    data = "\n".join(pending).strip()
    if data:
        yield data


def _sse_line(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(payload)


def _http_error(response: HttpResponse) -> OpenAICompatHttpError:
    message = f"vendor HTTP {response.status}"
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            message = _error_message(payload.get("error"), response.status)
    except json.JSONDecodeError:
        text = response.text().strip()
        if text:
            message = f"vendor HTTP {response.status}: {text[:200]}"
    retryable = response.status >= 500 or response.status == 429
    return OpenAICompatHttpError(message, status=response.status, retryable=retryable)


def _error_message(error: Any, status: int) -> str:
    if isinstance(error, dict):
        text = str(error.get("message") or error.get("code") or error)
    else:
        text = str(error)
    return f"vendor HTTP {status}: {text}" if status else text


def _safe_raw(payload: Mapping[str, Any]) -> dict[str, Any]:
    from agentsphere.identity.redact import redact

    keep = {key: payload[key] for key in ("id", "object", "model", "choices", "usage") if key in payload}
    redacted = redact(keep)
    return redacted if isinstance(redacted, dict) else {}


def _public_headers(headers: Mapping[str, str]) -> dict[str, str]:
    from agentsphere.identity.redact import REDACTED, is_secret_key

    out: dict[str, str] = {}
    for key, value in headers.items():
        if is_secret_key(str(key)) or str(key).lower() == "authorization":
            out[str(key)] = REDACTED
        else:
            out[str(key)] = value
    return out


def _has_authorization(headers: Mapping[str, str]) -> bool:
    return any(str(key).lower() == "authorization" and value for key, value in headers.items())


def _is_blocked_extra(key: str) -> bool:
    from agentsphere.identity.redact import is_secret_key

    lowered = key.lower().replace("-", "_")
    return lowered in _BLOCKED_EXTRA_KEYS or is_secret_key(key)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _safe_text(text: str, secret: str | None) -> str:
    from agentsphere.identity.redact import REDACTED

    if not text:
        return text
    redacted = text
    if secret:
        redacted = redacted.replace(secret, REDACTED)
        redacted = redacted.replace(f"Bearer {secret}", f"Bearer {REDACTED}")
    return redacted


def _redact_exception(exc: OpenAICompatError, secret: str | None) -> OpenAICompatError:
    message = _safe_text(str(exc), secret)
    if isinstance(exc, OpenAICompatNotConfigured):
        return OpenAICompatNotConfigured(message)
    if isinstance(exc, OpenAICompatHttpError):
        return OpenAICompatHttpError(message, status=exc.status, retryable=exc.retryable)
    wrapped = OpenAICompatError(message)
    wrapped.retryable = exc.retryable
    return wrapped
