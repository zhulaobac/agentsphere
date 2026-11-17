from __future__ import annotations

from agentsphere.kernel.assembler import ContextAssembler, TokenBudget
from agentsphere.model_adapter.types import Message, MessageRole, ToolDefinition


def test_assembler_respects_history_budget() -> None:
    assembler = ContextAssembler()
    history = [
        Message(role=MessageRole.USER, content="oldest" * 400),
        Message(role=MessageRole.ASSISTANT, content="mid" * 400),
        Message(role=MessageRole.USER, content="latest"),
    ]
    assembled = assembler.assemble(
        system_prompt="sys",
        history=history,
        budget=TokenBudget(total=200, system=20, tools=20, history=80),
    )
    assert assembled.messages[0].role is MessageRole.SYSTEM
    assert assembled.truncated["history"] is True
    contents = [m.content for m in assembled.messages if m.role is not MessageRole.SYSTEM]
    assert contents[-1] == "latest"
    assert all(c is not None and "oldest" not in c for c in contents)


def test_assembler_drops_tools_over_budget() -> None:
    assembler = ContextAssembler()
    tools = [
        ToolDefinition(name="a", description="alpha " * 50, parameters={"type": "object"}),
        ToolDefinition(name="b", description="beta " * 50, parameters={"type": "object"}),
    ]
    assembled = assembler.assemble(
        tools=tools,
        budget=TokenBudget(total=40, system=5, tools=15, history=20),
    )
    assert assembled.truncated["tools"] is True
    assert len(assembled.tools) < 2 or assembled.token_counts["tools"] <= 15
