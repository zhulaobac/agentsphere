"""Runtime port the Control Plane (future Go services) should depend on.

P0 implements this in Python. A later Go runtime can speak the same operations
without rewriting the Harness Kernel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from agentsphere.kernel.context import SessionState
from agentsphere.kernel.events import Event
from agentsphere.runtime.checkpoint import Checkpoint
from agentsphere.runtime.entities import Session, Turn
from agentsphere.runtime.lease import WorkerLease


class RuntimeProvider(Protocol):
    async def create_session(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        tool_pins: dict[str, str] | None = None,
    ) -> Session: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    def run_turn(self, session_id: str, user_input: str) -> AsyncIterator[Event]: ...

    def resume(self, session_id: str, *, worker_id: str | None = None) -> AsyncIterator[Event]: ...

    async def pause_session(self, session_id: str) -> None: ...

    async def cancel_session(self, session_id: str) -> Session: ...

    async def list_events(self, session_id: str) -> list[Event]: ...

    async def list_turns(self, session_id: str) -> list[Turn]: ...

    async def rebuild_state(self, session_id: str) -> SessionState: ...

    async def latest_checkpoint(self, session_id: str) -> Checkpoint | None: ...

    async def acquire_lease(self, session_id: str, worker_id: str) -> WorkerLease: ...
