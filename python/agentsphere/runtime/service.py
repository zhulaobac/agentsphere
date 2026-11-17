"""SessionRuntime: persist events, rebuild state, drive the Kernel. No in-process session cache as source of truth."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentsphere.identity.agent import AgentIdentity
from agentsphere.identity.audit import AuditLog, InMemoryAuditLog
from agentsphere.identity.policy import ApprovalRecord, ApprovalStore, PolicyEngine, PolicyHitlMiddleware
from agentsphere.identity.redact import RedactingListener, contains_secret, redact
from agentsphere.identity.vault import Vault
from agentsphere.kernel.assembler import ContextAssembler
from agentsphere.kernel.context import SessionState
from agentsphere.kernel.control import TurnControl
from agentsphere.kernel.dispatcher import DispatcherConfig, ToolDispatcher
from agentsphere.kernel.events import Event, EventEmitter, EventType
from agentsphere.kernel.harness import AgentHarness
from agentsphere.kernel.ids import new_id
from agentsphere.kernel.loop.react import ReActStrategy
from agentsphere.kernel.loop.strategy import LoopStrategy
from agentsphere.kernel.middleware import Middleware
from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.runtime.checkpoint import Checkpoint
from agentsphere.runtime.entities import Session, SessionStatus, Turn, TurnStatus
from agentsphere.runtime.lease import StaleLeaseError, WorkerCrash, WorkerLease
from agentsphere.runtime.recovery import (
    in_flight_tool_keys,
    in_progress_turn,
    reconstruct_resume,
    recover_state,
)
from agentsphere.runtime.store import EventStore, InMemoryEventStore
from agentsphere.runtime.workspace import LocalWorkspaceStore, WorkspaceSnapshotStore
from agentsphere.sandbox.subprocess import Sandbox
from agentsphere.tool_registry.registry import ToolRegistry
from agentsphere.trace.tracer import TraceRecorder


class SessionRuntime:
    """Managed Runtime façade. Control Plane should depend on RuntimeProvider, not Kernel types."""

    def __init__(
        self,
        harness: AgentHarness,
        store: EventStore | None = None,
        tracer: TraceRecorder | None = None,
        *,
        identity: AgentIdentity | None = None,
        vault: Vault | None = None,
        sandbox: Sandbox | None = None,
        policy: PolicyEngine | None = None,
        audit: AuditLog | None = None,
        approvals: ApprovalStore | None = None,
        workspace: WorkspaceSnapshotStore | None = None,
        lease_ttl_seconds: float = 30.0,
        secrets_for_redaction: set[str] | None = None,
    ) -> None:
        self.harness = harness
        self.store = store or InMemoryEventStore()
        self.tracer = tracer
        self.identity = identity or AgentIdentity(name="default-agent")
        self.vault = vault
        self.sandbox = sandbox
        self.policy = policy
        self.audit = audit or InMemoryAuditLog()
        self.approvals = approvals or ApprovalStore()
        self.workspace = workspace
        self.lease_ttl_seconds = lease_ttl_seconds
        self._secrets = set(secrets_for_redaction or ())
        self._controls: dict[str, TurnControl] = {}
        if vault is not None:
            add_sink = getattr(vault, "add_sink", None)
            if callable(add_sink):
                add_sink(self.add_secret_for_redaction)
            known = getattr(vault, "known_plaintext_secrets", None)
            if callable(known):
                for secret in known():
                    self.add_secret_for_redaction(secret)
        if tracer is not None:
            self.harness.event_emitter.add_listener(RedactingListener(tracer, self._secret_set))

    def add_secret_for_redaction(self, secret: str) -> None:
        if secret:
            self._secrets.add(secret)
            add = getattr(self.audit, "add_secret", None)
            if callable(add):
                add(secret)

    def _secret_set(self) -> set[str]:
        secrets = set(self._secrets)
        if self.vault is not None:
            known = getattr(self.vault, "known_plaintext_secrets", None)
            if callable(known):
                secrets |= set(known() or ())
        return secrets

    async def create_session(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        tool_pins: dict[str, str] | None = None,
        agent_id: str | None = None,
    ) -> Session:
        session = Session(
            system_prompt=system_prompt,
            metadata=metadata or {},
            tool_pins=tool_pins or {},
            agent_id=agent_id or self.identity.agent_id,
        )
        await self.store.create_session(session)
        event = Event(
            type=EventType.SESSION_CREATED,
            session_id=session.session_id,
            payload={
                "system_prompt": system_prompt,
                "metadata": session.metadata,
                "tool_pins": session.tool_pins,
                "agent_id": session.agent_id,
            },
        )
        stored = await self._persist(event)
        await self.harness.event_emitter.emit(stored)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return await self.store.get_session(session_id)

    async def list_events(self, session_id: str, *, after_seq: int | None = None) -> list[Event]:
        return await self.store.list_events(session_id, after_seq=after_seq)

    async def list_turns(self, session_id: str) -> list[Turn]:
        return await self.store.list_turns(session_id)

    async def rebuild_state(self, session_id: str) -> SessionState:
        checkpoint = await self.store.latest_checkpoint(session_id)
        events = await self.store.list_events(session_id)
        return recover_state(events, session_id=session_id, checkpoint=checkpoint)

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        return await self.store.latest_checkpoint(session_id)

    async def acquire_lease(self, session_id: str, worker_id: str, ttl_seconds: float | None = None) -> WorkerLease:
        ttl = self.lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        lease = await self.store.acquire_lease(session_id, worker_id, ttl)
        await self._persist(
            Event(
                type=EventType.LEASE_ACQUIRED,
                session_id=session_id,
                payload={"worker_id": worker_id, "token": lease.token, "lease_id": lease.lease_id},
            ),
            lease=lease,
        )
        return lease

    async def heartbeat(self, session_id: str, worker_id: str, token: int) -> WorkerLease:
        return await self.store.heartbeat_lease(session_id, worker_id, token, self.lease_ttl_seconds)

    async def release_lease(self, session_id: str, worker_id: str, token: int) -> None:
        await self._persist(
            Event(
                type=EventType.LEASE_RELEASED,
                session_id=session_id,
                payload={"worker_id": worker_id, "token": token},
            ),
            worker_id=worker_id,
            token=token,
        )
        await self.store.release_lease(session_id, worker_id, token)

    async def get_lease(self, session_id: str) -> WorkerLease | None:
        return await self.store.get_lease(session_id)

    async def pause_session(self, session_id: str) -> None:
        control = self._controls.get(session_id)
        if control is not None:
            control.request_pause()
        else:
            session = await self._require_session(session_id)
            session.status = SessionStatus.PAUSED
            await self.store.update_session(session)
            await self._persist(Event(type=EventType.SESSION_PAUSED, session_id=session_id, payload={}))

    async def cancel_session(self, session_id: str) -> Session:
        control = self._controls.get(session_id)
        if control is not None:
            control.request_cancel()
            session = await self._require_session(session_id)
            return session
        session = await self._require_session(session_id)
        session.status = SessionStatus.CANCELLED
        await self.store.update_session(session)
        await self._persist(Event(type=EventType.SESSION_CANCELLED, session_id=session_id, payload={}))
        return session

    async def approve(self, session_id: str, approval_id: str) -> ApprovalRecord:
        events = await self.store.list_events(session_id)
        self.approvals.hydrate(events)
        record = self.approvals.decide(approval_id, "granted")
        await self._persist(
            Event(
                type=EventType.APPROVAL_GRANTED,
                session_id=session_id,
                turn_id=record.turn_id,
                payload={
                    "approval_id": approval_id,
                    "tool_name": record.tool_name,
                    "idempotency_key": record.idempotency_key,
                },
            )
        )
        return record

    async def deny(self, session_id: str, approval_id: str) -> ApprovalRecord:
        events = await self.store.list_events(session_id)
        self.approvals.hydrate(events)
        record = self.approvals.decide(approval_id, "denied")
        await self._persist(
            Event(
                type=EventType.APPROVAL_DENIED,
                session_id=session_id,
                turn_id=record.turn_id,
                payload={
                    "approval_id": approval_id,
                    "tool_name": record.tool_name,
                    "idempotency_key": record.idempotency_key,
                },
            )
        )
        return record

    async def run_turn(
        self,
        session_id: str,
        user_input: str,
        *,
        worker_id: str | None = None,
    ) -> AsyncIterator[Event]:
        async for event in self._run(session_id, user_input, worker_id=worker_id, resume=False):
            yield event

    async def resume(
        self,
        session_id: str,
        *,
        worker_id: str | None = None,
        user_input: str = "",
    ) -> AsyncIterator[Event]:
        async for event in self._run(session_id, user_input, worker_id=worker_id, resume=True):
            yield event

    async def _run(
        self,
        session_id: str,
        user_input: str,
        *,
        worker_id: str | None,
        resume: bool,
    ) -> AsyncIterator[Event]:
        session = await self._require_session(session_id)
        if session.status is SessionStatus.CANCELLED:
            raise RuntimeError(f"session {session_id} is cancelled")
        worker = worker_id or new_id("worker")
        lease = await self.acquire_lease(session_id, worker)
        control = TurnControl()
        self._controls[session_id] = control
        turn: Turn | None = None
        crash: BaseException | None = None
        try:
            state = await self.rebuild_state(session_id)
            if not state.system_prompt:
                state.system_prompt = session.system_prompt
            if not state.tool_pins:
                state.tool_pins = session.tool_pins

            events = await self.store.list_events(session_id)
            self.approvals.hydrate(events)
            open_turn = in_progress_turn(events)
            resume_spec = None
            in_flight = set()
            confirm_keys: set[str] = set()
            confirm_commit = False
            if resume or open_turn is not None:
                if open_turn is None:
                    raise RuntimeError(f"session {session_id} has no in-progress turn to resume")
                turn_id, stored_input, turn_index = open_turn
                user_input = user_input or stored_input
                resume_spec = reconstruct_resume(events, turn_id=turn_id)
                resume_spec.user_input = user_input
                resume_spec.turn_index = turn_index
                in_flight = in_flight_tool_keys(events, turn_id=turn_id)
                confirm_keys = self.approvals.granted_keys(session_id)
                # Per-key only: a grant for one idempotency_key must not confirm other tools.
                confirm_commit = False
                session.status = SessionStatus.ACTIVE
                await self.store.update_session(session)
                await self._persist(
                    Event(type=EventType.SESSION_RESUMED, session_id=session_id, payload={"worker_id": worker}),
                    lease=lease,
                )
            elif session.status is SessionStatus.PAUSED:
                session.status = SessionStatus.ACTIVE
                await self.store.update_session(session)

            recovery = resume_spec is not None
            if resume_spec is not None:
                turns = await self.store.list_turns(session_id)
                matching = [item for item in turns if item.turn_id == resume_spec.turn_id]
                turn = matching[-1] if matching else (turns[-1] if turns else None)
            if self.workspace is not None and self.harness.dispatcher is not None:
                self.harness.dispatcher.workspace_path = str(self.workspace.path_for(session_id))
            async for event in self.harness.run_turn(
                state,
                user_input,
                resume=resume_spec,
                control=control,
                recovery=recovery,
                in_flight_keys=in_flight,
                confirm_commit=confirm_commit,
                confirm_keys=confirm_keys,
                agent_id=session.agent_id or self.identity.agent_id,
            ):
                try:
                    await self.heartbeat(session_id, worker, lease.token)
                except StaleLeaseError:
                    raise
                stored = await self._persist(event, lease=lease)
                turn = await self._project_turn(session, stored, user_input, turn)
                if stored.type is EventType.APPROVAL_REQUESTED:
                    self.approvals.put(
                        ApprovalRecord(
                            approval_id=str(stored.payload.get("approval_id")),
                            session_id=session_id,
                            turn_id=stored.turn_id or "",
                            tool_name=str(stored.payload.get("name", "")),
                            idempotency_key=str(stored.payload.get("idempotency_key", "")),
                            reason=str(stored.payload.get("reason", "")),
                            arguments=dict(stored.payload.get("arguments") or {}),
                        )
                    )
                    session.status = SessionStatus.WAITING_APPROVAL
                    if turn is not None:
                        turn.status = TurnStatus.WAITING_APPROVAL
                        await self.store.upsert_turn(turn)
                    await self.store.update_session(session)
                    await self._checkpoint(session_id, stored, state_hint=None, reason="approval_requested", lease=lease)
                elif stored.type is EventType.TURN_PAUSED:
                    session.status = SessionStatus.PAUSED
                    if turn is not None:
                        turn.status = TurnStatus.PAUSED
                        await self.store.upsert_turn(turn)
                    await self.store.update_session(session)
                    await self._checkpoint(session_id, stored, reason="paused", lease=lease)
                    await self._persist(
                        Event(type=EventType.SESSION_PAUSED, session_id=session_id, payload={}),
                        lease=lease,
                    )
                elif stored.type is EventType.STEP_COMPLETED:
                    await self._checkpoint(session_id, stored, reason="step", lease=lease)
                elif stored.type is EventType.TURN_COMPLETED:
                    await self._checkpoint(session_id, stored, reason="turn_completed", lease=lease)
                yield stored
        except BaseException as exc:
            crash = exc
            raise
        finally:
            self._controls.pop(session_id, None)
            if not isinstance(crash, WorkerCrash):
                try:
                    await self.release_lease(session_id, worker, lease.token)
                except StaleLeaseError:
                    pass

    async def _project_turn(
        self,
        session: Session,
        stored: Event,
        user_input: str,
        turn: Turn | None,
    ) -> Turn | None:
        session_id = session.session_id
        if stored.type is EventType.TURN_STARTED:
            turn = Turn(
                turn_id=stored.turn_id or "",
                session_id=session_id,
                turn_index=int(stored.payload.get("turn_index", 1)),
                user_input=user_input,
                status=TurnStatus.RUNNING,
                started_at=stored.timestamp,
            )
            await self.store.upsert_turn(turn)
            return turn
        if stored.type is EventType.TURN_COMPLETED and turn is not None:
            reason = str(stored.payload.get("reason", "completed"))
            status = TurnStatus.COMPLETED
            if reason == "error":
                status = TurnStatus.FAILED
            elif reason == "handoff":
                status = TurnStatus.HANDED_OFF
            elif reason == "cancelled":
                status = TurnStatus.CANCELLED
            turn.status = status
            turn.output = stored.payload.get("output")
            turn.reason = reason
            turn.completed_at = stored.timestamp or datetime.now(timezone.utc)
            await self.store.upsert_turn(turn)
            if status is TurnStatus.FAILED:
                session.status = SessionStatus.FAILED
                await self.store.update_session(session)
            elif status is TurnStatus.CANCELLED:
                session.status = SessionStatus.CANCELLED
                await self.store.update_session(session)
        elif stored.type is EventType.TURN_COMPLETED:
            # Resume path: TURN_STARTED already persisted; load existing turn.
            turns = await self.store.list_turns(session_id)
            if turns:
                turn = turns[-1]
                return await self._project_turn(session, stored, user_input, turn)
        return turn

    async def _checkpoint(
        self,
        session_id: str,
        event: Event,
        *,
        reason: str,
        state_hint: SessionState | None = None,
        lease: WorkerLease | None = None,
    ) -> Checkpoint:
        events = await self.store.list_events(session_id)
        state = state_hint or recover_state(
            events,
            session_id=session_id,
            checkpoint=await self.store.latest_checkpoint(session_id),
        )
        workspace_ref = None
        if self.workspace is not None:
            root = self.workspace.path_for(session_id)
            workspace_ref = self.workspace.snapshot(session_id, root)
        checkpoint = Checkpoint(
            session_id=session_id,
            seq=event.seq or 0,
            reason=reason,
            turn_id=event.turn_id,
            turn_index=state.turn_index,
            session_state=state,
            workspace_snapshot_ref=workspace_ref,
        )
        stored = await self.store.save_checkpoint(checkpoint)
        await self._persist(
            Event(
                type=EventType.CHECKPOINT_SAVED,
                session_id=session_id,
                turn_id=event.turn_id,
                payload={
                    "checkpoint_id": stored.checkpoint_id,
                    "seq": stored.seq,
                    "reason": reason,
                    "workspace_snapshot_ref": workspace_ref,
                },
            ),
            lease=lease,
        )
        return stored

    async def _persist(
        self,
        event: Event,
        lease: WorkerLease | None = None,
        *,
        worker_id: str | None = None,
        token: int | None = None,
    ) -> Event:
        secrets = self._secret_set()
        payload = redact(event.payload, extra_secrets=secrets)
        if secrets and contains_secret(payload, secrets):
            payload = redact(payload, extra_secrets=secrets)
        event = event.model_copy(update={"payload": payload})
        return await self.store.append_event(
            event,
            worker_id=worker_id or (lease.worker_id if lease is not None else None),
            token=token if token is not None else (lease.token if lease is not None else None),
        )

    async def _require_session(self, session_id: str) -> Session:
        session = await self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return session


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
    identity: AgentIdentity | None = None,
    vault: Vault | None = None,
    sandbox: Sandbox | None = None,
    policy: PolicyEngine | None = None,
    audit: AuditLog | None = None,
    approvals: ApprovalStore | None = None,
    workspace: WorkspaceSnapshotStore | None = None,
    workspace_root: Path | str | None = None,
    lease_ttl_seconds: float = 30.0,
) -> SessionRuntime:
    identity = identity or AgentIdentity(name="default-agent")
    audit = audit or InMemoryAuditLog()
    approvals = approvals or ApprovalStore()
    store = store or InMemoryEventStore()
    if workspace is None and workspace_root is not None:
        workspace = LocalWorkspaceStore(Path(workspace_root))
    hooks = list(middlewares or [])
    if policy is not None:
        def risk_lookup(name: str, version: str | None) -> Any:
            spec = registry.get(name, version)
            return spec.risk_level

        hooks.append(
            PolicyHitlMiddleware(
                policy,
                identity,
                approvals,
                risk_lookup=risk_lookup,
                audit=audit,
            )
        )
    emitter = EventEmitter()
    workspace_path = None
    if workspace is not None:
        workspace_path = str(workspace.path_for("pending"))
    dispatcher = ToolDispatcher(
        registry,
        config=dispatcher_config,
        middlewares=hooks,
        result_cache=store,
        vault=vault,
        sandbox=sandbox,
        identity=identity,
        audit=audit,
        approvals=approvals,
        workspace_path=workspace_path,
    )
    harness = AgentHarness(
        strategy=strategy or ReActStrategy(),
        model_adapter=adapter,
        assembler=assembler,
        dispatcher=dispatcher,
        middlewares=hooks,
        event_emitter=emitter,
        max_steps=max_steps,
        tool_definitions=registry.list_definitions(),
    )
    return SessionRuntime(
        harness,
        store=store,
        tracer=tracer,
        identity=identity,
        vault=vault,
        sandbox=sandbox,
        policy=policy,
        audit=audit,
        approvals=approvals,
        workspace=workspace,
        lease_ttl_seconds=lease_ttl_seconds,
    )
