"""Pause / resume / cancel signals and turn-resume snapshots. Kernel-side, no Runtime imports."""

from __future__ import annotations

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.model_adapter.types import ModelResponse
from pydantic import BaseModel


class TurnPaused(Exception):
    """Raised inside the harness when pause is requested between steps."""


class TurnCancelled(Exception):
    """Raised inside the harness when cancel is requested."""


class TurnWaitingApproval(Exception):
    """The turn yielded control because a high-risk tool needs HITL approval."""

    def __init__(self, approval_id: str, message: str = "waiting for human approval") -> None:
        super().__init__(message)
        self.approval_id = approval_id


class TurnControl:
    """Shared flag object the Runtime mutates and the Kernel checks between steps."""

    def __init__(self) -> None:
        self.pause_requested: bool = False
        self.cancel_requested: bool = False

    def request_pause(self) -> None:
        self.pause_requested = True

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def check(self) -> None:
        if self.cancel_requested:
            raise TurnCancelled("session/turn cancelled")
        if self.pause_requested:
            raise TurnPaused("session/turn paused")


class TurnResume(BaseModel):
    """Enough context for the Kernel to continue an in-progress turn without re-starting it."""

    turn_id: str
    turn_index: int
    start_step_index: int = 0
    user_input: str = ""
    last_model_response: ModelResponse | None = None
    last_tool_results: list[dict[str, object]] | None = None
    pending_tool_calls: list[ToolCallRequest] | None = None
