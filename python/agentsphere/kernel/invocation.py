"""Per-call context for dispatcher middleware (session, turn, idempotency_key, agent)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class InvocationContext:
    session_id: str = ""
    turn_id: str = ""
    step_id: str = ""
    step_index: int = 0
    call_index: int = 0
    agent_id: str = ""
    idempotency_key: str = ""
    recovery: bool = False
    in_flight: bool = False
    confirm_commit: bool = False


_current: ContextVar[InvocationContext | None] = ContextVar("agentsphere_invocation", default=None)


def bind_invocation(ctx: InvocationContext):
    return _current.set(ctx)


def reset_invocation(token: object) -> None:
    _current.reset(token)  # type: ignore[arg-type]


def current_invocation() -> InvocationContext | None:
    return _current.get()
