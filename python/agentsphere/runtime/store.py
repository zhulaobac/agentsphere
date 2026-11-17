"""Durable event, checkpoint, lease, and result-cache stores. Kafka/Redis remain production backends."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agentsphere.kernel.events import Event
from agentsphere.runtime.cache import CachedResult
from agentsphere.runtime.checkpoint import Checkpoint
from agentsphere.runtime.clock import Clock, RealClock
from agentsphere.runtime.entities import Session, Turn
from agentsphere.runtime.lease import LeaseHeldError, StaleLeaseError, WorkerLease, compute_expiry


class EventStore(ABC):
    """Persistence port. A future Go runtime can implement the same contract over gRPC/HTTP."""

    @abstractmethod
    async def create_session(self, session: Session) -> None: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def update_session(self, session: Session) -> None: ...

    @abstractmethod
    async def append_event(self, event: Event) -> Event: ...

    @abstractmethod
    async def list_events(self, session_id: str, *, after_seq: int | None = None) -> list[Event]: ...

    @abstractmethod
    async def upsert_turn(self, turn: Turn) -> None: ...

    @abstractmethod
    async def list_turns(self, session_id: str) -> list[Turn]: ...

    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint: ...

    @abstractmethod
    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None: ...

    @abstractmethod
    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None: ...

    @abstractmethod
    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]: ...

    @abstractmethod
    async def put_result(self, record: CachedResult) -> None: ...

    @abstractmethod
    async def get_result(self, idempotency_key: str) -> CachedResult | None: ...

    @abstractmethod
    async def acquire_lease(
        self,
        session_id: str,
        worker_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    @abstractmethod
    async def heartbeat_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    @abstractmethod
    async def release_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        *,
        now: datetime | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_lease(self, session_id: str) -> WorkerLease | None: ...


class InMemoryEventStore(EventStore):
    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or RealClock()
        self._sessions: dict[str, Session] = {}
        self._events: dict[str, list[Event]] = defaultdict(list)
        self._turns: dict[str, dict[str, Turn]] = defaultdict(dict)
        self._checkpoints: dict[str, list[Checkpoint]] = defaultdict(list)
        self._checkpoint_by_id: dict[str, Checkpoint] = {}
        self._results: dict[str, CachedResult] = {}
        self._leases: dict[str, WorkerLease] = {}

    async def create_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session.model_copy(deep=True)

    async def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        return session.model_copy(deep=True) if session else None

    async def update_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session.model_copy(deep=True)

    async def append_event(self, event: Event) -> Event:
        seq = len(self._events[event.session_id]) + 1
        stored = event.model_copy(update={"seq": seq})
        self._events[event.session_id].append(stored)
        return stored

    async def list_events(self, session_id: str, *, after_seq: int | None = None) -> list[Event]:
        events = [e.model_copy(deep=True) for e in self._events.get(session_id, [])]
        if after_seq is not None:
            events = [e for e in events if (e.seq or 0) > after_seq]
        return events

    async def upsert_turn(self, turn: Turn) -> None:
        self._turns[turn.session_id][turn.turn_id] = turn.model_copy(deep=True)

    async def list_turns(self, session_id: str) -> list[Turn]:
        turns = list(self._turns.get(session_id, {}).values())
        return sorted(turns, key=lambda t: t.turn_index)

    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        stored = checkpoint.model_copy(deep=True)
        self._checkpoints[checkpoint.session_id].append(stored)
        self._checkpoint_by_id[stored.checkpoint_id] = stored
        return stored.model_copy(deep=True)

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        items = self._checkpoints.get(session_id) or []
        if not items:
            return None
        latest = max(items, key=lambda c: (c.seq, c.created_at))
        return latest.model_copy(deep=True)

    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        item = self._checkpoint_by_id.get(checkpoint_id)
        return item.model_copy(deep=True) if item else None

    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        items = [c.model_copy(deep=True) for c in self._checkpoints.get(session_id, [])]
        return sorted(items, key=lambda c: c.seq)

    async def put_result(self, record: CachedResult) -> None:
        self._results[record.idempotency_key] = record.model_copy(deep=True)

    async def get_result(self, idempotency_key: str) -> CachedResult | None:
        item = self._results.get(idempotency_key)
        return item.model_copy(deep=True) if item else None

    async def acquire_lease(
        self,
        session_id: str,
        worker_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or self._clock.now()
        current = self._leases.get(session_id)
        if current is not None and not current.released and current.expires_at > now:
            if current.worker_id == worker_id:
                current.heartbeat_at = now
                current.expires_at = compute_expiry(now, ttl_seconds)
                return current.model_copy(deep=True)
            raise LeaseHeldError(current)
        token = (current.token + 1) if current is not None else 1
        lease = WorkerLease(
            session_id=session_id,
            worker_id=worker_id,
            token=token,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=compute_expiry(now, ttl_seconds),
        )
        self._leases[session_id] = lease
        return lease.model_copy(deep=True)

    async def heartbeat_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or self._clock.now()
        current = self._leases.get(session_id)
        if (
            current is None
            or current.released
            or current.worker_id != worker_id
            or current.token != token
            or current.expires_at <= now
        ):
            raise StaleLeaseError(session_id, worker_id, token)
        current.heartbeat_at = now
        current.expires_at = compute_expiry(now, ttl_seconds)
        return current.model_copy(deep=True)

    async def release_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        *,
        now: datetime | None = None,
    ) -> None:
        del now
        current = self._leases.get(session_id)
        if current is None:
            return
        if current.worker_id != worker_id or current.token != token:
            raise StaleLeaseError(session_id, worker_id, token)
        current.released = True
        current.expires_at = datetime.fromtimestamp(0, tz=timezone.utc)

    async def get_lease(self, session_id: str) -> WorkerLease | None:
        lease = self._leases.get(session_id)
        return lease.model_copy(deep=True) if lease else None


class SqliteEventStore(EventStore):
    """SQLite-backed store suitable for local/dev. Schema is event-log + projections."""

    def __init__(self, path: str | Path = ":memory:", clock: Clock | None = None) -> None:
        self._path = str(path)
        self._clock = clock or RealClock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                system_prompt TEXT NOT NULL DEFAULT '',
                tool_pins_json TEXT NOT NULL DEFAULT '{}',
                agent_id TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                output TEXT,
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                body_json TEXT NOT NULL,
                UNIQUE(session_id, seq)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                body_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS result_cache (
                idempotency_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                body_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS leases (
                session_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                token INTEGER NOT NULL,
                acquired_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                released INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._ensure_column("sessions", "agent_id", "TEXT")
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = {row["name"] for row in rows}
        if column not in names:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    async def create_session(self, session: Session) -> None:
        payload = session.model_dump(mode="json")
        self._conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, status, metadata_json, system_prompt, tool_pins_json, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                payload["created_at"],
                payload["status"],
                json.dumps(payload["metadata"]),
                session.system_prompt,
                json.dumps(session.tool_pins),
                session.agent_id,
            ),
        )
        self._conn.commit()

    async def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        keys = row.keys()
        return Session(
            session_id=row["session_id"],
            created_at=row["created_at"],
            status=row["status"],
            metadata=json.loads(row["metadata_json"]),
            system_prompt=row["system_prompt"],
            tool_pins=json.loads(row["tool_pins_json"]),
            agent_id=row["agent_id"] if "agent_id" in keys else None,
        )

    async def update_session(self, session: Session) -> None:
        payload = session.model_dump(mode="json")
        self._conn.execute(
            """
            UPDATE sessions
            SET status = ?, metadata_json = ?, system_prompt = ?, tool_pins_json = ?, agent_id = ?
            WHERE session_id = ?
            """,
            (
                payload["status"],
                json.dumps(payload["metadata"]),
                session.system_prompt,
                json.dumps(session.tool_pins),
                session.agent_id,
                session.session_id,
            ),
        )
        self._conn.commit()

    async def append_event(self, event: Event) -> Event:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE session_id = ?",
            (event.session_id,),
        ).fetchone()
        seq = int(row["m"]) + 1
        stored = event.model_copy(update={"seq": seq})
        self._conn.execute(
            "INSERT INTO events (event_id, session_id, seq, body_json) VALUES (?, ?, ?, ?)",
            (stored.event_id, stored.session_id, seq, stored.model_dump_json()),
        )
        self._conn.commit()
        return stored

    async def list_events(self, session_id: str, *, after_seq: int | None = None) -> list[Event]:
        if after_seq is None:
            rows = self._conn.execute(
                "SELECT body_json FROM events WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body_json FROM events WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
                (session_id, after_seq),
            ).fetchall()
        return [Event.model_validate_json(row["body_json"]) for row in rows]

    async def upsert_turn(self, turn: Turn) -> None:
        payload = turn.model_dump(mode="json")
        self._conn.execute(
            """
            INSERT INTO turns (
                turn_id, session_id, turn_index, user_input, status,
                started_at, completed_at, output, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                status=excluded.status,
                completed_at=excluded.completed_at,
                output=excluded.output,
                reason=excluded.reason
            """,
            (
                turn.turn_id,
                turn.session_id,
                turn.turn_index,
                turn.user_input,
                payload["status"],
                payload["started_at"],
                payload["completed_at"],
                turn.output,
                turn.reason,
            ),
        )
        self._conn.commit()

    async def list_turns(self, session_id: str) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [
            Turn(
                turn_id=row["turn_id"],
                session_id=row["session_id"],
                turn_index=row["turn_index"],
                user_input=row["user_input"],
                status=row["status"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                output=row["output"],
                reason=row["reason"],
            )
            for row in rows
        ]

    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        stored = checkpoint.model_copy(deep=True)
        payload = stored.model_dump(mode="json")
        self._conn.execute(
            """
            INSERT INTO checkpoints (checkpoint_id, session_id, seq, created_at, body_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                stored.checkpoint_id,
                stored.session_id,
                stored.seq,
                payload["created_at"],
                stored.model_dump_json(),
            ),
        )
        self._conn.commit()
        return stored

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT body_json FROM checkpoints WHERE session_id = ? ORDER BY seq DESC, created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate_json(row["body_json"])

    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT body_json FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate_json(row["body_json"])

    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        rows = self._conn.execute(
            "SELECT body_json FROM checkpoints WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        return [Checkpoint.model_validate_json(row["body_json"]) for row in rows]

    async def put_result(self, record: CachedResult) -> None:
        self._conn.execute(
            """
            INSERT INTO result_cache (idempotency_key, session_id, body_json)
            VALUES (?, ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET body_json=excluded.body_json
            """,
            (record.idempotency_key, record.session_id, record.model_dump_json()),
        )
        self._conn.commit()

    async def get_result(self, idempotency_key: str) -> CachedResult | None:
        row = self._conn.execute(
            "SELECT body_json FROM result_cache WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        return CachedResult.model_validate_json(row["body_json"])

    async def acquire_lease(
        self,
        session_id: str,
        worker_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or self._clock.now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT * FROM leases WHERE session_id = ?", (session_id,)).fetchone()
            current = _lease_from_row(row) if row else None
            if current is not None and not current.released and current.expires_at > now:
                if current.worker_id == worker_id:
                    current.heartbeat_at = now
                    current.expires_at = compute_expiry(now, ttl_seconds)
                    _write_lease(self._conn, current)
                    self._conn.commit()
                    return current
                self._conn.rollback()
                raise LeaseHeldError(current)
            token = (current.token + 1) if current is not None else 1
            lease = WorkerLease(
                session_id=session_id,
                worker_id=worker_id,
                token=token,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=compute_expiry(now, ttl_seconds),
            )
            _write_lease(self._conn, lease)
            self._conn.commit()
            return lease
        except LeaseHeldError:
            raise
        except Exception:
            self._conn.rollback()
            raise

    async def heartbeat_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or self._clock.now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT * FROM leases WHERE session_id = ?", (session_id,)).fetchone()
            current = _lease_from_row(row) if row else None
            if (
                current is None
                or current.released
                or current.worker_id != worker_id
                or current.token != token
                or current.expires_at <= now
            ):
                self._conn.rollback()
                raise StaleLeaseError(session_id, worker_id, token)
            current.heartbeat_at = now
            current.expires_at = compute_expiry(now, ttl_seconds)
            _write_lease(self._conn, current)
            self._conn.commit()
            return current
        except StaleLeaseError:
            raise
        except Exception:
            self._conn.rollback()
            raise

    async def release_lease(
        self,
        session_id: str,
        worker_id: str,
        token: int,
        *,
        now: datetime | None = None,
    ) -> None:
        del now
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT * FROM leases WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                self._conn.commit()
                return
            current = _lease_from_row(row)
            if current.worker_id != worker_id or current.token != token:
                self._conn.rollback()
                raise StaleLeaseError(session_id, worker_id, token)
            current.released = True
            current.expires_at = datetime.fromtimestamp(0, tz=timezone.utc)
            _write_lease(self._conn, current)
            self._conn.commit()
        except StaleLeaseError:
            raise
        except Exception:
            self._conn.rollback()
            raise

    async def get_lease(self, session_id: str) -> WorkerLease | None:
        row = self._conn.execute("SELECT * FROM leases WHERE session_id = ?", (session_id,)).fetchone()
        return _lease_from_row(row) if row else None

    def close(self) -> None:
        self._conn.close()


def _lease_from_row(row: sqlite3.Row) -> WorkerLease:
    return WorkerLease(
        session_id=row["session_id"],
        worker_id=row["worker_id"],
        lease_id=row["lease_id"],
        token=int(row["token"]),
        acquired_at=row["acquired_at"],
        heartbeat_at=row["heartbeat_at"],
        expires_at=row["expires_at"],
        released=bool(row["released"]),
    )


def _write_lease(conn: sqlite3.Connection, lease: WorkerLease) -> None:
    payload = lease.model_dump(mode="json")
    conn.execute(
        """
        INSERT INTO leases (
            session_id, worker_id, lease_id, token, acquired_at, heartbeat_at, expires_at, released
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            worker_id=excluded.worker_id,
            lease_id=excluded.lease_id,
            token=excluded.token,
            acquired_at=excluded.acquired_at,
            heartbeat_at=excluded.heartbeat_at,
            expires_at=excluded.expires_at,
            released=excluded.released
        """,
        (
            lease.session_id,
            lease.worker_id,
            lease.lease_id,
            lease.token,
            payload["acquired_at"],
            payload["heartbeat_at"],
            payload["expires_at"],
            1 if lease.released else 0,
        ),
    )
