from __future__ import annotations

import pytest

from agentsphere.runtime import FakeClock, InMemoryEventStore, LeaseHeldError, SqliteEventStore, StaleLeaseError


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
