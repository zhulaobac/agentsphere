"""Request-aware mock model for the research workflow. No network."""

from __future__ import annotations

from agentsphere.model_adapter.mock import MockModelProvider, assistant_text, assistant_tool_response
from agentsphere.model_adapter.types import MessageRole, ModelRequest, ModelResponse
from agentsphere.tenants.product_team.constants import (
    TOOL_ASSEMBLE,
    TOOL_COMPARE,
    TOOL_LIST,
    TOOL_LOOKUP,
)
from agentsphere.tenants.product_team.dataset import fixture_outputs


def _user_text(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if message.role is MessageRole.USER and message.content:
            return message.content
    return ""


def _tool_messages(request: ModelRequest) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for message in request.messages:
        if message.role is MessageRole.TOOL:
            found.append((message.name or "", message.content or ""))
    return found


def _is_safety(topic: str) -> bool:
    lowered = topic.lower()
    return any(
        token in lowered
        for token in ("vault", "token", "secret", "credential", "password", "dump")
    )


def plan_research_turn(request: ModelRequest) -> ModelResponse:
    topic = _user_text(request)
    if _is_safety(topic):
        return assistant_text(fixture_outputs()["safety-no-secret"])

    notes = dict(_tool_messages(request))
    names = {name for name, _ in _tool_messages(request)}
    if TOOL_LOOKUP not in names:
        return assistant_tool_response(TOOL_LOOKUP, {"topic": topic})
    if TOOL_LIST not in names:
        return assistant_tool_response(TOOL_LIST, {"category": topic, "topic": topic})
    if TOOL_COMPARE not in names:
        return assistant_tool_response(TOOL_COMPARE, {"topic": topic})
    if TOOL_ASSEMBLE not in names:
        return assistant_tool_response(TOOL_ASSEMBLE, {"topic": topic, "notes": notes})
    assembled = notes.get(TOOL_ASSEMBLE) or fixture_outputs().get("notebooks-easy", "{}")
    return assistant_text(assembled)


class ResearchMockProvider(MockModelProvider):
    """Always uses :func:`plan_research_turn` so experiments can create a new runtime per case."""

    name = "mock-research"

    def __init__(self) -> None:
        super().__init__(default_content="")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        item = plan_research_turn(request)
        if not item.provider:
            item = item.model_copy(update={"provider": self.name})
        return item
