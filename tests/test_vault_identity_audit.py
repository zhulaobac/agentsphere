from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agentsphere.identity import (
    AgentIdentity,
    InMemoryAuditLog,
    InMemoryVault,
    PolicyRule,
    StaticPolicyEngine,
    contains_secret,
)
from agentsphere.identity.vault import current_issued_credential
from agentsphere.kernel.events import EventType
from agentsphere.kernel.loop import ReActStrategy
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text, assistant_tool_response
from agentsphere.model_adapter.types import MessageRole
from agentsphere.runtime import InMemoryEventStore, build_runtime, replay
from agentsphere.tool_registry import RiskLevel, ToolRegistry, ToolSpec


SECRET = "super-secret-value-xyz"


def _registry() -> ToolRegistry:
    def handler(arguments: Mapping[str, Any]) -> str:
        issued = current_issued_credential()
        assert issued is not None
        assert issued.token != SECRET
        return f"ok:{arguments['text']}"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            version="1.0.0",
            owner="platform",
            json_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            risk_level=RiskLevel.LOW,
            idempotent=True,
            required_credentials=["weather_api"],
            handler=handler,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_vault_secret_never_enters_model_context_events_or_traces() -> None:
    identity = AgentIdentity(name="weather-agent", tenant_id="acme")
    vault = InMemoryVault()
    await vault.put("weather_api", SECRET, agent_id=identity.agent_id, ttl_seconds=60)
    audit = InMemoryAuditLog()
    audit.add_secret(SECRET)
    policy = StaticPolicyEngine([PolicyRule(policy_id="allow", effect="allow")], hitl_at_or_above=RiskLevel.CRITICAL)
    from agentsphere.trace import TraceRecorder

    tracer = TraceRecorder()
    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(
                script=[
                    assistant_tool_response("echo", {"text": "hi"}),
                    assistant_text("echoed: hi"),
                ]
            )
        ),
        registry=_registry(),
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
        tracer=tracer,
        identity=identity,
        vault=vault,
        policy=policy,
        audit=audit,
    )
    runtime.add_secret_for_redaction(SECRET)
    session = await runtime.create_session(system_prompt="You are a test agent.")
    events = [event async for event in runtime.run_turn(session.session_id, "please echo hi")]
    assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in events)

    blob = " ".join(event.model_dump_json() for event in events)
    stored = await runtime.list_events(session.session_id)
    stored_blob = " ".join(event.model_dump_json() for event in stored)
    state = replay(stored)
    messages_blob = " ".join((m.content or "") for m in state.messages)
    assert SECRET not in blob
    assert SECRET not in stored_blob
    assert SECRET not in messages_blob
    assert not contains_secret(stored_blob, {SECRET})

    model_messages = [m for m in state.messages if m.role is not MessageRole.TOOL]
    for message in model_messages:
        assert SECRET not in (message.content or "")

    tool_messages = [m for m in state.messages if m.role is MessageRole.TOOL]
    assert tool_messages
    assert all(message.untrusted for message in tool_messages)

    tree = tracer.inspect(session.session_id)
    assert SECRET not in tree.model_dump_json()
    records = await audit.list()
    assert any(record.action == "tool.call" for record in records)
    assert any(record.action == "policy.decide" for record in records)
    assert SECRET not in " ".join(record.model_dump_json() for record in records)
    assert identity.is_developer_account is False
    assert identity.principal_type == "agent"


@pytest.mark.asyncio
async def test_default_deny_policy_blocks_unlisted_tools() -> None:
    identity = AgentIdentity(name="locked-down")
    policy = StaticPolicyEngine([])  # no allow rules
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            version="1.0.0",
            owner="platform",
            json_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            risk_level=RiskLevel.LOW,
            idempotent=True,
            handler=lambda arguments: arguments["text"],
        )
    )
    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(script=[assistant_tool_response("echo", {"text": "hi"}), assistant_text("x")])
        ),
        registry=registry,
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
        identity=identity,
        policy=policy,
    )
    session = await runtime.create_session()
    with pytest.raises(PermissionError, match="default-deny"):
        async for _event in runtime.run_turn(session.session_id, "please echo hi"):
            pass
