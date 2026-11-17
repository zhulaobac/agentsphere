from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agentsphere.identity import AgentIdentity, PolicyRule, StaticPolicyEngine
from agentsphere.kernel.events import EventType
from agentsphere.kernel.loop import ReActStrategy
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text, assistant_tool_response
from agentsphere.runtime import InMemoryEventStore, SessionStatus, TurnStatus, build_runtime
from agentsphere.tool_registry import RiskLevel, ToolRegistry, ToolSpec


def _echo_registry(handler=None, *, risk: RiskLevel = RiskLevel.LOW, name: str = "echo") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            version="1.0.0",
            owner="platform",
            json_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            risk_level=risk,
            idempotent=True,
            handler=handler or (lambda arguments: str(arguments["text"])),
        )
    )
    return registry


@pytest.mark.asyncio
async def test_pause_releases_and_resume_continues_from_checkpoint() -> None:
    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(
                script=[
                    assistant_tool_response("echo", {"text": "hi"}),
                    assistant_text("echoed: hi"),
                ]
            )
        ),
        registry=_echo_registry(),
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
    )
    session = await runtime.create_session()
    paused = []
    async for event in runtime.run_turn(session.session_id, "please echo hi"):
        paused.append(event)
        if event.type is EventType.STEP_COMPLETED:
            await runtime.pause_session(session.session_id)
    assert any(event.type is EventType.TURN_PAUSED for event in paused)
    loaded = await runtime.get_session(session.session_id)
    assert loaded is not None
    assert loaded.status is SessionStatus.PAUSED
    lease = await runtime.get_lease(session.session_id)
    assert lease is None or lease.released is True

    rest = [event async for event in runtime.resume(session.session_id)]
    assert any(event.type is EventType.TURN_RESUMED for event in rest)
    assert any(event.type is EventType.TURN_COMPLETED for event in rest)
    loaded = await runtime.get_session(session.session_id)
    assert loaded is not None
    assert loaded.status is SessionStatus.ACTIVE


@pytest.mark.asyncio
async def test_cancel_ends_the_turn() -> None:
    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(
                script=[
                    assistant_tool_response("echo", {"text": "hi"}),
                    assistant_text("echoed: hi"),
                ]
            )
        ),
        registry=_echo_registry(),
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
    )
    session = await runtime.create_session()
    events = []
    async for event in runtime.run_turn(session.session_id, "please echo hi"):
        events.append(event)
        if event.type is EventType.STEP_STARTED:
            await runtime.cancel_session(session.session_id)
    assert any(event.type is EventType.TURN_CANCELLED for event in events)
    completed = next(event for event in events if event.type is EventType.TURN_COMPLETED)
    assert completed.payload["reason"] == "cancelled"
    loaded = await runtime.get_session(session.session_id)
    assert loaded is not None
    assert loaded.status is SessionStatus.CANCELLED


@pytest.mark.asyncio
async def test_hitl_releases_lease_and_resume_after_approval() -> None:
    identity = AgentIdentity(name="payments-agent", tenant_id="acme")
    policy = StaticPolicyEngine(
        [PolicyRule(policy_id="allow-all", effect="allow")],
        hitl_at_or_above=RiskLevel.HIGH,
    )
    calls = {"n": 0}

    def transfer(arguments: Mapping[str, Any]) -> str:
        calls["n"] += 1
        return f"sent:{arguments['text']}"

    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(
                script=[
                    assistant_tool_response("echo", {"text": "100"}),
                    assistant_text("transferred"),
                ]
            )
        ),
        registry=_echo_registry(transfer, risk=RiskLevel.HIGH),
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
        identity=identity,
        policy=policy,
    )
    session = await runtime.create_session()
    events = [event async for event in runtime.run_turn(session.session_id, "send 100")]
    requested = next(event for event in events if event.type is EventType.APPROVAL_REQUESTED)
    assert calls["n"] == 0
    loaded = await runtime.get_session(session.session_id)
    assert loaded is not None
    assert loaded.status is SessionStatus.WAITING_APPROVAL
    lease = await runtime.get_lease(session.session_id)
    assert lease is None or lease.released is True
    turns = await runtime.list_turns(session.session_id)
    assert turns[0].status is TurnStatus.WAITING_APPROVAL

    other = await runtime.acquire_lease(session.session_id, "worker-hitl")
    await runtime.release_lease(session.session_id, "worker-hitl", other.token)

    await runtime.approve(session.session_id, str(requested.payload["approval_id"]))
    rest = [event async for event in runtime.resume(session.session_id)]
    assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in rest)
    assert calls["n"] == 1
    loaded = await runtime.get_session(session.session_id)
    assert loaded is not None
    assert loaded.status is SessionStatus.ACTIVE
