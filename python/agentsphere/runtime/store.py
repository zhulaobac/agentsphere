"""Durable event stores for local/dev and tests. No Kafka, Redis, or lease system in P0."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from agentsphere.kernel.events import Event
from agentsphere.runtime.entities import Session, Turn


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
    async def list_events(self, session_id: str) -> list[Event]: ...

    @abstractmethod
    async def upsert_turn(self, turn: Turn) -> None: ...

    @abstractmethod
    async def list_turns(self, session_id: str) -> list[Turn]: ...


class InMemoryEventStore(EventStore):
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._events: dict[str, list[Event]] = defaultdict(list)
        self._turns: dict[str, dict[str, Turn]] = defaultdict(dict)

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

    async def list_events(self, session_id: str) -> list[Event]:
        return [e.model_copy(deep=True) for e in self._events.get(session_id, [])]

    async def upsert_turn(self, turn: Turn) -> None:
        self._turns[turn.session_id][turn.turn_id] = turn.model_copy(deep=True)

    async def list_turns(self, session_id: str) -> list[Turn]:
        turns = list(self._turns.get(session_id, {}).values())
        return sorted(turns, key=lambda t: t.turn_index)


class SqliteEventStore(EventStore):
    """SQLite-backed store suitable for local/dev. Schema is event-log + projections."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
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
                tool_pins_json TEXT NOT NULL DEFAULT '{}'
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
            """
        )
        self._conn.commit()

    async def create_session(self, session: Session) -> None:
        payload = session.model_dump(mode="json")
        self._conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, status, metadata_json, system_prompt, tool_pins_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                payload["created_at"],
                payload["status"],
                json.dumps(payload["metadata"]),
                session.system_prompt,
                json.dumps(session.tool_pins),
            ),
        )
        self._conn.commit()

    async def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row["session_id"],
            created_at=row["created_at"],
            status=row["status"],
            metadata=json.loads(row["metadata_json"]),
            system_prompt=row["system_prompt"],
            tool_pins=json.loads(row["tool_pins_json"]),
        )

    async def update_session(self, session: Session) -> None:
        payload = session.model_dump(mode="json")
        self._conn.execute(
            """
            UPDATE sessions
            SET status = ?, metadata_json = ?, system_prompt = ?, tool_pins_json = ?
            WHERE session_id = ?
            """,
            (
                payload["status"],
                json.dumps(payload["metadata"]),
                session.system_prompt,
                json.dumps(session.tool_pins),
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

    async def list_events(self, session_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT body_json FROM events WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
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

    def close(self) -> None:
        self._conn.close()
