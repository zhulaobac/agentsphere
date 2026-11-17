from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.dispatcher import DispatcherConfig, ToolDispatcher
from agentsphere.tool_registry import RiskLevel, ToolRegistry, ToolSpec


def _registry(handler, *, name: str = "echo") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            version="1.0.0",
            owner="platform",
            json_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            risk_level=RiskLevel.LOW,
            idempotent=True,
            handler=handler,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_dispatcher_validates_parameters() -> None:
    dispatcher = ToolDispatcher(_registry(lambda args: args["text"]))
    results = await dispatcher.dispatch(
        [ToolCallRequest(id="c1", name="echo", arguments={})],
        session_id="s",
        turn_id="t",
        step_id="st",
    )
    assert results[0].error is not None
    assert "parameter validation failed" in results[0].error


@pytest.mark.asyncio
async def test_dispatcher_retries_then_succeeds() -> None:
    attempts = {"n": 0}

    def flaky(arguments: Mapping[str, Any]) -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return str(arguments["text"])

    dispatcher = ToolDispatcher(
        _registry(flaky),
        config=DispatcherConfig(max_retries=2, retry_backoff_seconds=0.0, timeout_seconds=1.0),
    )
    results = await dispatcher.dispatch(
        [ToolCallRequest(id="c1", name="echo", arguments={"text": "ok"})],
        session_id="s",
        turn_id="t",
        step_id="st",
    )
    assert results[0].error is None
    assert results[0].output == "ok"
    assert results[0].attempts == 2
    assert results[0].idempotency_key


@pytest.mark.asyncio
async def test_dispatcher_timeout_and_truncation() -> None:
    async def slow(_arguments: Mapping[str, Any]) -> str:
        await asyncio.sleep(1)
        return "never"

    timed = ToolDispatcher(
        _registry(slow, name="slow"),
        config=DispatcherConfig(timeout_seconds=0.05, max_retries=0),
    )
    timed_result = await timed.dispatch(
        [ToolCallRequest(id="c1", name="slow", arguments={"text": "x"})],
        session_id="s",
        turn_id="t",
        step_id="st",
    )
    assert timed_result[0].error is not None
    assert "timed out" in timed_result[0].error

    trunc = ToolDispatcher(
        _registry(lambda args: "x" * 50),
        config=DispatcherConfig(max_result_chars=20),
    )
    truncated = await trunc.dispatch(
        [ToolCallRequest(id="c2", name="echo", arguments={"text": "x"})],
        session_id="s",
        turn_id="t",
        step_id="st",
    )
    assert truncated[0].truncated is True
    assert truncated[0].output.endswith("[truncated]")
