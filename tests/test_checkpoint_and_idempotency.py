from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agentsphere.kernel.events import Event, EventType
from agentsphere.kernel.loop import ReActStrategy
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text, assistant_tool_response
from agentsphere.runtime import (
    FakeClock,
    InMemoryEventStore,
    SqliteEventStore,
    WorkerCrash,
    build_runtime,
    recover_state,
    replay,
)
from agentsphere.tool_registry import RiskLevel, ToolRegistry, ToolSpec


class CrashOnCompleteStore(InMemoryEventStore):
    """Persist TOOL_CALL_STARTED, then die before TOOL_CALL_COMPLETED is durable."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        super().__init__(clock=clock)
        self.crash_on_complete = True

    async def append_event(
        self, event: Event, *, worker_id: str | None = None, token: int | None = None
    ) -> Event:
        if self.crash_on_complete and event.type is EventType.TOOL_CALL_COMPLETED:
            raise WorkerCrash("worker died after the tool ran but before the completed event was stored")
        return await super().append_event(event, worker_id=worker_id, token=token)


class CrashOnStartStore(InMemoryEventStore):
    """Die after TOOL_CALL_STARTED so the handler never returns a cached result."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        super().__init__(clock=clock)
        self.crash_on_start = True

    async def append_event(
        self, event: Event, *, worker_id: str | None = None, token: int | None = None
    ) -> Event:
        stored = await super().append_event(event, worker_id=worker_id, token=token)
        if self.crash_on_start and event.type is EventType.TOOL_CALL_STARTED:
            raise WorkerCrash("worker died mid-tool before the side effect completed")
        return stored


def _registry(handler, *, idempotent: bool = True, compensation_action: str | None = None, two_phase: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            version="1.0.0",
            owner="platform",
            json_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            risk_level=RiskLevel.LOW,
            idempotent=idempotent,
            compensation_action=compensation_action,
            two_phase=two_phase,
            handler=handler,
        )
    )
    return registry


def _adapter() -> ModelAdapter:
    return ModelAdapter(
        MockModelProvider(
            script=[
                assistant_tool_response("echo", {"text": "hi"}),
                assistant_text("echoed: hi"),
            ]
        )
    )


@pytest.mark.asyncio
async def test_checkpoint_at_turn_end_and_resume_from_snapshot(tmp_path) -> None:
    calls = {"n": 0}

    def echo(arguments: Mapping[str, Any]) -> str:
        calls["n"] += 1
        return str(arguments["text"])

    db = tmp_path / "runtime.sqlite"
    store = SqliteEventStore(db)
    runtime = build_runtime(adapter=_adapter(), registry=_registry(echo), store=store, strategy=ReActStrategy())
    session = await runtime.create_session(system_prompt="sys")
    events = [event async for event in runtime.run_turn(session.session_id, "please echo hi")]
    assert any(event.type is EventType.TURN_COMPLETED for event in events)
    checkpoint = await runtime.latest_checkpoint(session.session_id)
    assert checkpoint is not None
    assert checkpoint.reason == "turn_completed"
    assert checkpoint.session_state.turn_index == 1
    store.close()

    reopened = SqliteEventStore(db)
    runtime2 = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("second")])),
        registry=_registry(echo),
        store=reopened,
        strategy=ReActStrategy(),
    )
    rebuilt = await runtime2.rebuild_state(session.session_id)
    full = replay(await reopened.list_events(session.session_id), session_id=session.session_id)
    assert rebuilt.messages == full.messages
    assert rebuilt.turn_index == full.turn_index
    second = [event async for event in runtime2.run_turn(session.session_id, "next")]
    assert any(event.type is EventType.TURN_COMPLETED for event in second)
    assert calls["n"] == 1
    reopened.close()


@pytest.mark.asyncio
async def test_crash_mid_tool_recovers_without_duplicate_side_effects() -> None:
    calls = {"n": 0}

    def echo(arguments: Mapping[str, Any]) -> str:
        calls["n"] += 1
        return str(arguments["text"])

    clock = FakeClock()
    store = CrashOnCompleteStore(clock)
    runtime = build_runtime(
        adapter=_adapter(),
        registry=_registry(echo, idempotent=True),
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    session = await runtime.create_session()
    with pytest.raises(WorkerCrash):
        async for _event in runtime.run_turn(session.session_id, "please echo hi", worker_id="worker-a"):
            pass
    assert calls["n"] == 1
    events = await store.list_events(session.session_id)
    assert any(e.type is EventType.TOOL_CALL_STARTED for e in events)
    assert not any(e.type is EventType.TOOL_CALL_COMPLETED for e in events)

    store.crash_on_complete = False
    clock.advance(10)
    recovered = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("echoed: hi")])),
        registry=_registry(echo, idempotent=True),
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    follow = [event async for event in recovered.run_turn(session.session_id, "", worker_id="worker-b")]
    completed = [event for event in follow if event.type is EventType.TOOL_CALL_COMPLETED]
    assert completed
    assert completed[0].payload.get("from_cache") is True
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_non_idempotent_in_flight_tool_is_not_silently_reexecuted() -> None:
    calls = {"n": 0}

    def charge(arguments: Mapping[str, Any]) -> str:
        calls["n"] += 1
        raise WorkerCrash("died inside non-idempotent tool")

    clock = FakeClock()
    store = CrashOnStartStore(clock)
    registry = _registry(charge, idempotent=False, compensation_action="refund")
    runtime = build_runtime(
        adapter=_adapter(),
        registry=registry,
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    session = await runtime.create_session()
    with pytest.raises(WorkerCrash):
        async for _event in runtime.run_turn(session.session_id, "charge", worker_id="worker-a"):
            pass
    store.crash_on_start = False
    clock.advance(10)

    recovered = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("stopped")])),
        registry=registry,
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    follow = [event async for event in recovered.run_turn(session.session_id, "", worker_id="worker-b")]
    failed = [event for event in follow if event.type is EventType.TOOL_CALL_FAILED]
    assert failed
    assert failed[0].payload.get("skipped_non_idempotent") is True
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_lease_expiry_allows_another_worker_to_take_over() -> None:
    calls = {"n": 0}

    def echo(arguments: Mapping[str, Any]) -> str:
        calls["n"] += 1
        return str(arguments["text"])

    clock = FakeClock()
    store = CrashOnCompleteStore(clock)
    runtime = build_runtime(
        adapter=_adapter(),
        registry=_registry(echo),
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    session = await runtime.create_session()
    with pytest.raises(WorkerCrash):
        async for _event in runtime.run_turn(session.session_id, "please echo hi", worker_id="worker-a"):
            pass
    held = await store.get_lease(session.session_id)
    assert held is not None
    assert held.worker_id == "worker-a"
    assert held.released is False

    clock.advance(10)
    store.crash_on_complete = False
    recovered = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("echoed: hi")])),
        registry=_registry(echo),
        store=store,
        strategy=ReActStrategy(),
        lease_ttl_seconds=5,
    )
    follow = [event async for event in recovered.run_turn(session.session_id, "", worker_id="worker-b")]
    assert any(event.type is EventType.TURN_COMPLETED for event in follow)
    lease = await store.get_lease(session.session_id)
    assert lease is not None
    assert lease.token >= 2
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_recover_state_replays_only_the_event_tail() -> None:
    store = InMemoryEventStore()
    runtime = build_runtime(adapter=_adapter(), registry=_registry(lambda a: a["text"]), store=store)
    session = await runtime.create_session()
    events = [event async for event in runtime.run_turn(session.session_id, "please echo hi")]
    assert any(event.type is EventType.TURN_COMPLETED for event in events)
    checkpoint = await runtime.latest_checkpoint(session.session_id)
    assert checkpoint is not None
    all_events = await store.list_events(session.session_id)
    from_ckpt = recover_state(all_events, session_id=session.session_id, checkpoint=checkpoint)
    full = replay(all_events, session_id=session.session_id)
    assert from_ckpt.messages == full.messages
