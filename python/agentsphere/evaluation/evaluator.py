"""Pluggable evaluator protocol and in-process registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentsphere.evaluation.types import EvalResult, EvalSample


@runtime_checkable
class Evaluator(Protocol):
    name: str
    version: str
    kind: str

    async def evaluate(self, sample: EvalSample) -> EvalResult: ...


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Evaluator] = {}

    def register(self, evaluator: Evaluator, *, replace: bool = False) -> Evaluator:
        key = (evaluator.name, evaluator.version)
        if key in self._items and not replace:
            raise ValueError(f"evaluator {evaluator.name}@{evaluator.version} already registered")
        self._items[key] = evaluator
        return evaluator

    def get(self, name: str, version: str | None = None) -> Evaluator:
        if version is not None:
            item = self._items.get((name, version))
            if item is None:
                raise KeyError(f"evaluator {name}@{version} not found")
            return item
        versions = [e for (n, _), e in self._items.items() if n == name]
        if not versions:
            raise KeyError(f"evaluator {name} not found")
        return versions[-1]

    def list(self) -> list[Evaluator]:
        return list(self._items.values())
