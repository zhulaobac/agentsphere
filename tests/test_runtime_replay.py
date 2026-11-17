from __future__ import annotations

from pathlib import Path

import pytest

from agentsphere.kernel.events import Event, EventType
from agentsphere.model_adapter.types import MessageRole
from agentsphere.runtime import InMemoryEventStore, SqliteEventStore, replay
from agentsphere.runtime.entities import Session, Turn, TurnStatus


def _events(session_id: str) -> list[Event]:
    return [
        Event(type=EventType.SESSION_CREATED, session_id=session_id, payload={"system_prompt": "sys"}),
        Event(
            type=EventType.TURN_STARTED,
            session_id=session_id,
            turn_id="t1",
            payload={"user_input": "hello", "turn_index": 1},
        ),
        Event(
            type=EventType.MODEL_CALL_COMPLETED,
            session_id=session_id,
            turn_id="t1",
            payload={"content": "hi", "tool_calls": []},
        ),
        Event(
            type=EventType.TURN_COMPLETED,
            session_id=session_id,
            turn_id="t1",
            payload={"output": "hi", "reason": "completed", "workflow_cursor": None},
        ),
    ]


def test_replay_rebuilds_session_state_from_events() -> None:
    state = replay(_events("s1"))
    assert state.session_id == "s1"
    assert state.system_prompt == "sys"
    assert state.turn_index == 1
    assert [m.role for m in state.messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert state.messages[0].content == "hello"
    assert state.messages[1].content == "hi"


def test_replay_includes_tool_messages() -> None:
    events = _events("s2")
    events.insert(
        3,
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="s2",
            turn_id="t1",
            payload={"call_id": "c1", "name": "echo", "output": "pong"},
        ),
    )
    state = replay(events)
    assert state.messages[-1].role is MessageRole.TOOL
    assert state.messages[-1].content == "pong"


@pytest.mark.asyncio
async def test_in_memory_store_assigns_monotonic_seq() -> None:
    store = InMemoryEventStore()
    session = Session(session_id="s3")
    await store.create_session(session)
    first = await store.append_event(Event(type=EventType.SESSION_CREATED, session_id="s3"))
    second = await store.append_event(
        Event(type=EventType.TURN_STARTED, session_id="s3", payload={"user_input": "x", "turn_index": 1})
    )
    assert first.seq == 1
    assert second.seq == 2
    listed = await store.list_events("s3")
    assert [e.seq for e in listed] == [1, 2]


@pytest.mark.asyncio
async def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "runtime.sqlite"
    store = SqliteEventStore(db)
    session = Session(session_id="s4", system_prompt="be brief")
    await store.create_session(session)
    for event in _events("s4"):
        await store.append_event(event)
    await store.upsert_turn(
        Turn(turn_id="t1", session_id="s4", turn_index=1, user_input="hello", status=TurnStatus.COMPLETED)
    )
    store.close()

    reopened = SqliteEventStore(db)
    loaded = await reopened.get_session("s4")
    assert loaded is not None
    assert loaded.system_prompt == "be brief"
    events = await reopened.list_events("s4")
    state = replay(events)
    assert state.messages[0].content == "hello"
    turns = await reopened.list_turns("s4")
    assert turns[0].turn_id == "t1"
    reopened.close()
