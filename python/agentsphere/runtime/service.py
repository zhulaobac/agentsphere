"""SessionRuntime: persist events, rebuild state, drive the Kernel. No in-process session cache as source of truth."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from typing import Any

from agentsphere.kernel.assembler import ContextAssembler
from agentsphere.kernel.context import SessionState
from agentsphere.kernel.dispatcher import DispatcherConfig, ToolDispatcher
from agentsphere.kernel.events import Event, EventEmitter, EventType
from agentsphere.kernel.harness import AgentHarness
from agentsphere.kernel.loop.react import ReActStrategy
from agentsphere.kernel.loop.strategy import LoopStrategy
from agentsphere.kernel.middleware import Middleware
from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.runtime.entities import Session, SessionStatus, Turn, TurnStatus
from agentsphere.runtime.replay import replay
from agentsphere.runtime.store import EventStore, InMemoryEventStore
from agentsphere.tool_registry.registry import ToolRegistry
from agentsphere.trace.tracer import TraceRecorder


class SessionRuntime:
    """Managed Runtime façade. Control Plane should depend on RuntimeProvider, not Kernel types."""

    def __init__(
        self,
        harness: AgentHarness,
        store: EventStore | None = None,
        tracer: TraceRecorder | None = None,
    ) -> None:
        self.harness = harness
        self.store = store or InMemoryEventStore()
        self.tracer = tracer
        if tracer is not None:
            self.harness.event_emitter.add_listener(tracer)

    async def create_session(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        tool_pins: dict[str, str] | None = None,
    ) -> Session:
        session = Session(
            system_prompt=system_prompt,
            metadata=metadata or {},
            tool_pins=tool_pins or {},
        )
        await self.store.create_session(session)
        event = Event(
            type=EventType.SESSION_CREATED,
            session_id=session.session_id,
            payload={
                "system_prompt": system_prompt,
                "metadata": session.metadata,
                "tool_pins": session.tool_pins,
            },
        )
        stored = await self.store.append_event(event)
        await self.harness.event_emitter.emit(stored)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return await self.store.get_session(session_id)

    async def list_events(self, session_id: str) -> list[Event]:
        return await self.store.list_events(session_id)

    async def list_turns(self, session_id: str) -> list[Turn]:
        return await self.store.list_turns(session_id)

    async def rebuild_state(self, session_id: str) -> SessionState:
        events = await self.store.list_events(session_id)
        return replay(events, session_id=session_id)

    async def run_turn(self, session_id: str, user_input: str) -> AsyncIterator[Event]:
        session = await self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        state = await self.rebuild_state(session_id)
        if not state.system_prompt:
            state.system_prompt = session.system_prompt
        if not state.tool_pins:
            state.tool_pins = session.tool_pins

        turn: Turn | None = None
        async for event in self.harness.run_turn(state, user_input):
            stored = await self.store.append_event(event)
            if stored.type is EventType.TURN_STARTED:
                turn = Turn(
                    turn_id=stored.turn_id or "",
                    session_id=session_id,
                    turn_index=int(stored.payload.get("turn_index", state.turn_index + 1)),
                    user_input=user_input,
                    status=TurnStatus.RUNNING,
                    started_at=stored.timestamp,
                )
                await self.store.upsert_turn(turn)
            elif stored.type is EventType.TURN_COMPLETED and turn is not None:
                reason = str(stored.payload.get("reason", "completed"))
                status = TurnStatus.COMPLETED
                if reason == "error":
                    status = TurnStatus.FAILED
                elif reason == "handoff":
                    status = TurnStatus.HANDED_OFF
                turn.status = status
                turn.output = stored.payload.get("output")
                turn.reason = reason
                turn.completed_at = stored.timestamp or datetime.now(timezone.utc)
                await self.store.upsert_turn(turn)
                if status is TurnStatus.FAILED:
                    session.status = SessionStatus.FAILED
                    await self.store.update_session(session)
            yield stored


def build_runtime(
    adapter: ModelAdapter,
    registry: ToolRegistry,
    *,
    strategy: LoopStrategy | None = None,
    store: EventStore | None = None,
    tracer: TraceRecorder | None = None,
    middlewares: Sequence[Middleware] | None = None,
    assembler: ContextAssembler | None = None,
    dispatcher_config: DispatcherConfig | None = None,
    max_steps: int = 20,
) -> SessionRuntime:
    emitter = EventEmitter()
    dispatcher = ToolDispatcher(
        registry,
        config=dispatcher_config,
        middlewares=list(middlewares or []),
    )
    harness = AgentHarness(
        strategy=strategy or ReActStrategy(),
        model_adapter=adapter,
        assembler=assembler,
        dispatcher=dispatcher,
        middlewares=list(middlewares or []),
        event_emitter=emitter,
        max_steps=max_steps,
        tool_definitions=registry.list_definitions(),
    )
    return SessionRuntime(harness, store=store, tracer=tracer)
