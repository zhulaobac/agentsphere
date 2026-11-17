"""Human-in-the-loop interrupt. Raised from before_tool middleware / policy gates."""

from __future__ import annotations

from typing import Any

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.ids import new_id


class ApprovalRequired(Exception):
    """A high-risk tool call is gated on human approval. The worker should release its lease."""

    def __init__(
        self,
        tool_call: ToolCallRequest,
        *,
        reason: str,
        approval_id: str | None = None,
        decision: Any = None,
        session_id: str = "",
        turn_id: str = "",
        idempotency_key: str = "",
    ) -> None:
        super().__init__(reason)
        self.tool_call = tool_call
        self.reason = reason
        self.approval_id = approval_id or new_id("appr")
        self.decision = decision
        self.session_id = session_id
        self.turn_id = turn_id
        self.idempotency_key = idempotency_key
