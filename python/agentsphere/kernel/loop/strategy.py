"""LoopStrategy protocol and registry.

P0 implements ReAct and Workflow. Planner-Executor and Multi-Agent are reserved
extension points (roadmap P2) — they are not implemented here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentsphere.kernel.actions import Action
from agentsphere.kernel.context import TurnContext

UNIMPLEMENTED_STRATEGIES: dict[str, str] = {
    "planner-executor": (
        "Planner-Executor (plan / execute / replan) is a P2 extension. "
        "Register a custom LoopStrategy via register_strategy()."
    ),
    "multi-agent": (
        "Multi-Agent (Orchestrator-Worker, Handoff, Peer) is a P2 extension. "
        "Handoff actions are accepted by the Kernel but not orchestrated in P0. "
        "Register a custom LoopStrategy via register_strategy()."
    ),
}


class UnimplementedStrategyError(NotImplementedError):
    pass


@runtime_checkable
class LoopStrategy(Protocol):
    """Pure (TurnContext -> Action) policy. Must not persist process-local session state."""

    name: str

    def step(self, ctx: TurnContext) -> Action: ...


_REGISTRY: dict[str, type] = {}


def register_strategy(name: str, cls: type) -> None:
    """Extension point for P2 strategies and tenant-specific loops."""
    key = name.strip().lower()
    if key in UNIMPLEMENTED_STRATEGIES and cls is None:
        raise UnimplementedStrategyError(UNIMPLEMENTED_STRATEGIES[key])
    _REGISTRY[key] = cls


def get_strategy_class(name: str) -> type:
    key = name.strip().lower()
    if key in UNIMPLEMENTED_STRATEGIES and key not in _REGISTRY:
        raise UnimplementedStrategyError(UNIMPLEMENTED_STRATEGIES[key])
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY) or ["(none registered)"])
        raise KeyError(f"unknown loop strategy '{name}'. registered: {known}")
    return _REGISTRY[key]


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
