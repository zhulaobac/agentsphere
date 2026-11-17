"""Context Assembler pipeline with token budgets.

P0 stages: system, tool definitions, history.
Memory / RAG stages are intentionally absent (P2).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.context import AssembledContext
from agentsphere.model_adapter.types import Message, MessageRole, ToolDefinition


class TokenBudget(BaseModel):
    total: int = 8_192
    system: int = 1_024
    tools: int = 2_048
    history: int = 5_120


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharHeuristicCounter:
    """Offline token estimate (~4 chars / token). No tokenizer download, no network."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)


class AssembleRequest(BaseModel):
    system_prompt: str = ""
    tools: list[ToolDefinition] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)


class StageOutput(BaseModel):
    name: str
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tokens: int = 0
    truncated: bool = False


class AssemblerStage(Protocol):
    name: str

    def run(self, request: AssembleRequest, budget: int, counter: TokenCounter) -> StageOutput: ...


def _clip_text(text: str, counter: TokenCounter, budget: int) -> tuple[str, bool]:
    if counter.count(text) <= budget:
        return text, False
    # binary-search a prefix that fits
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if counter.count(candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


class SystemStage:
    name = "system"

    def run(self, request: AssembleRequest, budget: int, counter: TokenCounter) -> StageOutput:
        if not request.system_prompt:
            return StageOutput(name=self.name, tokens=0)
        text, truncated = _clip_text(request.system_prompt, counter, budget)
        tokens = counter.count(text)
        message = Message(role=MessageRole.SYSTEM, content=text)
        return StageOutput(name=self.name, messages=[message], tokens=tokens, truncated=truncated)


class ToolDefinitionStage:
    name = "tools"

    def run(self, request: AssembleRequest, budget: int, counter: TokenCounter) -> StageOutput:
        kept: list[ToolDefinition] = []
        used = 0
        truncated = False
        for tool in request.tools:
            cost = counter.count(tool.model_dump_json())
            if used + cost > budget:
                truncated = True
                break
            kept.append(tool)
            used += cost
        return StageOutput(name=self.name, tools=kept, tokens=used, truncated=truncated)


class HistoryStage:
    name = "history"

    def run(self, request: AssembleRequest, budget: int, counter: TokenCounter) -> StageOutput:
        # Prefer recent messages; drop oldest first.
        selected: list[Message] = []
        used = 0
        truncated = False
        for message in reversed(request.history):
            cost = counter.count(message.model_dump_json())
            if used + cost > budget:
                truncated = True
                break
            selected.append(message)
            used += cost
        selected.reverse()
        return StageOutput(name=self.name, messages=selected, tokens=used, truncated=truncated)


class ContextAssembler:
    """Ordered pipeline. Extra stages (memory, retrieval) can be appended later without Kernel changes."""

    def __init__(
        self,
        stages: list[AssemblerStage] | None = None,
        counter: TokenCounter | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self.stages = stages or [SystemStage(), ToolDefinitionStage(), HistoryStage()]
        self.counter: TokenCounter = counter or CharHeuristicCounter()
        self.budget = budget or TokenBudget()

    def assemble(
        self,
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        history: list[Message] | None = None,
        budget: TokenBudget | None = None,
    ) -> AssembledContext:
        request = AssembleRequest(
            system_prompt=system_prompt,
            tools=list(tools or []),
            history=list(history or []),
        )
        budget = budget or self.budget
        per_stage = {
            "system": budget.system,
            "tools": budget.tools,
            "history": budget.history,
        }

        messages: list[Message] = []
        assembled_tools: list[ToolDefinition] = []
        token_counts: dict[str, int] = {}
        truncated: dict[str, bool] = {}
        total = 0

        for stage in self.stages:
            cap = per_stage.get(stage.name, max(0, budget.total - total))
            remaining = max(0, min(cap, budget.total - total))
            output = stage.run(request, remaining, self.counter)
            messages.extend(output.messages)
            if output.tools:
                assembled_tools = output.tools
            token_counts[output.name] = output.tokens
            truncated[output.name] = output.truncated
            total += output.tokens

        token_counts["total"] = total
        return AssembledContext(
            messages=messages,
            tools=assembled_tools,
            token_counts=token_counts,
            truncated=truncated,
        )

    def add_stage(self, stage: AssemblerStage, *, after: str | None = None) -> None:
        """Extension point: P2 memory/RAG stages register here."""
        if after is None:
            self.stages.append(stage)
            return
        for i, existing in enumerate(self.stages):
            if existing.name == after:
                self.stages.insert(i + 1, stage)
                return
        self.stages.append(stage)
