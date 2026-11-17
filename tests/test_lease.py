from __future__ import annotations

import pytest

from agentsphere.kernel.events import Event, EventType
from agentsphere.runtime import FakeClock, InMemoryEventStore, LeaseHeldError, SqliteEventStore, StaleLeaseError
from agentsphere.runtime.entities import Session


@pytest.mark.asyncio
async def test_in_memory_lease_heartbeat_and_expiry_takeover() -> None:
    clock = FakeClock()
    store = InMemoryEventStore(clock=clock)
    first = await store.acquire_lease("ses", "worker-a", ttl_seconds=5)
    assert first.token == 1
    with pytest.raises(LeaseHeldError):
        await store.acquire_lease("ses", "worker-b", ttl_seconds=5)
    renewed = await store.heartbeat_lease("ses", "worker-a", first.token, ttl_seconds=5)
    assert renewed.token == 1
    clock.advance(10)
    with pytest.raises(StaleLeaseError):
        await store.heartbeat_lease("ses", "worker-a", first.token, ttl_seconds=5)
    second = await store.acquire_lease("ses", "worker-b", ttl_seconds=5)
    assert second.token == 2
    assert second.worker_id == "worker-b"
    with pytest.raises(StaleLeaseError):
        await store.heartbeat_lease("ses", "worker-a", first.token, ttl_seconds=5)


@pytest.mark.asyncio
async def test_sqlite_lease_release_and_fencing(tmp_path) -> None:
    clock = FakeClock()
    store = SqliteEventStore(tmp_path / "lease.sqlite", clock=clock)
    lease = await store.acquire_lease("ses", "worker-a", ttl_seconds=30)
    await store.release_lease("ses", "worker-a", lease.token)
    other = await store.acquire_lease("ses", "worker-b", ttl_seconds=30)
    assert other.token == lease.token + 1
    with pytest.raises(StaleLeaseError):
        await store.release_lease("ses", "worker-a", lease.token)
    store.close()


@pytest.mark.asyncio
async def test_stale_token_cannot_append_after_takeover() -> None:
    clock = FakeClock()
    store = InMemoryEventStore(clock=clock)
    await store.create_session(Session(session_id="ses"))
    first = await store.acquire_lease("ses", "worker-a", ttl_seconds=5)
    ok = await store.append_event(
        Event(type=EventType.SESSION_CREATED, session_id="ses"),
        worker_id="worker-a",
        token=first.token,
    )
    assert ok.seq == 1
    clock.advance(10)
    second = await store.acquire_lease("ses", "worker-b", ttl_seconds=5)
    assert second.token != first.token
    with pytest.raises(StaleLeaseError):
        await store.append_event(
            Event(type=EventType.TURN_STARTED, session_id="ses", payload={"user_input": "x"}),
            worker_id="worker-a",
            token=first.token,
        )
    with pytest.raises(StaleLeaseError):
        await store.append_event(Event(type=EventType.TURN_STARTED, session_id="ses"))
    allowed = await store.append_event(
        Event(type=EventType.TURN_STARTED, session_id="ses", payload={"user_input": "ok"}),
        worker_id="worker-b",
        token=second.token,
    )
    assert allowed.seq == 2


@pytest.mark.asyncio
async def test_sqlite_stale_token_cannot_append_after_takeover(tmp_path) -> None:
    clock = FakeClock()
    store = SqliteEventStore(tmp_path / "fence.sqlite", clock=clock)
    await store.create_session(Session(session_id="ses"))
    first = await store.acquire_lease("ses", "worker-a", ttl_seconds=5)
    clock.advance(10)
    second = await store.acquire_lease("ses", "worker-b", ttl_seconds=5)
    with pytest.raises(StaleLeaseError):
        await store.append_event(
            Event(type=EventType.TURN_STARTED, session_id="ses"),
            worker_id="worker-a",
            token=first.token,
        )
    stored = await store.append_event(
        Event(type=EventType.TURN_STARTED, session_id="ses"),
        worker_id="worker-b",
        token=second.token,
    )
    assert stored.seq == 1
    store.close()
