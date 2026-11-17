"""Worker lease + heartbeat. Expired leases can be taken over by another worker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.ids import new_id


class WorkerLease(BaseModel):
    session_id: str
    worker_id: str
    lease_id: str = Field(default_factory=lambda: new_id("lease"))
    token: int = 1
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    heartbeat_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    released: bool = False


class LeaseError(RuntimeError):
    pass


class LeaseHeldError(LeaseError):
    def __init__(self, lease: WorkerLease) -> None:
        super().__init__(
            f"session {lease.session_id} is leased to worker {lease.worker_id} until {lease.expires_at.isoformat()}"
        )
        self.lease = lease


class WorkerCrash(Exception):
    """Test/helper: the worker died without releasing its lease (kill -9 analogue)."""


class StaleLeaseError(LeaseError):
    def __init__(self, session_id: str, worker_id: str, token: int | None = None) -> None:
        super().__init__(
            f"stale or missing lease for session {session_id} worker {worker_id} token {token}"
        )
        self.session_id = session_id
        self.worker_id = worker_id
        self.token = token


class LeaseStore(Protocol):
    async def acquire_lease(
        self,
        session_id: str,
        worker_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    async def heartbeat_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    async def release_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        *,
        now: datetime | None = None,
    ) -> None: ...

    async def get_lease(self, session_id: str) -> WorkerLease | None: ...


def compute_expiry(now: datetime, ttl_seconds: float) -> datetime:
    return now + timedelta(seconds=ttl_seconds)
