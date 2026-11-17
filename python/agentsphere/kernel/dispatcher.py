"""Tool Dispatcher: schema validation, concurrency, timeout/retry, result truncation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, Field

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.middleware import Middleware, MiddlewareStage, run_chain


class ToolInvokeError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ExecutableTool(Protocol):
    name: str
    version: str
    parameter_schema: Mapping[str, Any]
    idempotent: bool

    def invoke(self, arguments: Mapping[str, Any]) -> Any: ...


class ToolResolver(Protocol):
    def resolve(self, name: str, version: str | None = None) -> ExecutableTool: ...


class ToolResult(BaseModel):
    call_id: str
    name: str
    version: str = ""
    output: str = ""
    truncated: bool = False
    error: str | None = None
    attempts: int = 1
    idempotency_key: str = ""


class DispatcherConfig(BaseModel):
    max_concurrency: int = 4
    timeout_seconds: float = 10.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.01
    max_result_chars: int = 8_192


class ToolDispatcher:
    def __init__(
        self,
        resolver: ToolResolver,
        config: DispatcherConfig | None = None,
        middlewares: Sequence[Middleware] | None = None,
    ) -> None:
        self.resolver = resolver
        self.config = config or DispatcherConfig()
        self.middlewares = list(middlewares or [])

    async def dispatch(
        self,
        calls: Sequence[ToolCallRequest],
        *,
        session_id: str,
        turn_id: str,
        step_id: str,
    ) -> list[ToolResult]:
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def one(call: ToolCallRequest, index: int) -> ToolResult:
            async with semaphore:
                return await self._invoke(call, session_id, turn_id, step_id, index)

        return list(await asyncio.gather(*[one(call, i) for i, call in enumerate(calls)]))

    async def _invoke(
        self,
        call: ToolCallRequest,
        session_id: str,
        turn_id: str,
        step_id: str,
        index: int,
    ) -> ToolResult:
        key = _idempotency_key(session_id, turn_id, step_id, index, call)
        try:
            tool = self.resolver.resolve(call.name, call.version)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.id,
                name=call.name,
                error=str(exc),
                idempotency_key=key,
            )

        try:
            _validate_params(call.arguments, tool.parameter_schema)
        except JSONSchemaValidationError as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                version=tool.version,
                error=f"parameter validation failed: {exc.message}",
                idempotency_key=key,
            )

        attempts = 0
        last_error: str | None = None
        max_tries = 1 + max(0, self.config.max_retries)
        raw: Any = None
        succeeded = False

        while attempts < max_tries:
            attempts += 1

            async def core(payload: ToolCallRequest) -> Any:
                return await _call_handler(tool.invoke, payload.arguments)

            try:
                raw = await asyncio.wait_for(
                    run_chain(
                        self.middlewares,
                        MiddlewareStage.BEFORE_TOOL,
                        call,
                        core,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                succeeded = True
                break
            except TimeoutError:
                last_error = f"tool timed out after {self.config.timeout_seconds}s"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if isinstance(exc, ToolInvokeError) and not exc.retryable:
                    break
            if attempts < max_tries:
                await asyncio.sleep(self.config.retry_backoff_seconds * attempts)

        if not succeeded:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                version=tool.version,
                error=last_error or "tool failed",
                attempts=attempts,
                idempotency_key=key,
            )

        # after_tool runs only after a successful invoke and must not re-enter the retry loop.
        try:
            after = await run_chain(
                self.middlewares,
                MiddlewareStage.AFTER_TOOL,
                raw,
                _identity,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.id,
                name=call.name,
                version=tool.version,
                error=f"after_tool middleware failed: {exc}",
                attempts=attempts,
                idempotency_key=key,
            )

        text, truncated = _truncate(_stringify(after), self.config.max_result_chars)
        return ToolResult(
            call_id=call.id,
            name=call.name,
            version=tool.version,
            output=text,
            truncated=truncated,
            attempts=attempts,
            idempotency_key=key,
        )


async def _identity(payload: Any) -> Any:
    return payload


async def _call_handler(handler: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    """Run async handlers on the loop; run sync handlers in a worker thread so timeouts apply."""
    args = dict(arguments)
    if inspect.iscoroutinefunction(handler):
        return await handler(args)

    def run_sync() -> Any:
        return handler(args)

    result = await asyncio.to_thread(run_sync)
    if inspect.isawaitable(result):
        return await result
    return result


def _validate_params(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if not schema:
        return
    Draft202012Validator(schema).validate(dict(arguments))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "...[truncated]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker, True


def _idempotency_key(
    session_id: str,
    turn_id: str,
    step_id: str,
    index: int,
    call: ToolCallRequest,
) -> str:
    material = json.dumps(
        {
            "session": session_id,
            "turn": turn_id,
            "step": step_id,
            "index": index,
            "name": call.name,
            "arguments": call.arguments,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
