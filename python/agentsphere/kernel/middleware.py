"""Onion-model middleware. Hook points: before_model / after_model / before_tool / after_tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

NextFn = Callable[..., Awaitable[Any]]


class MiddlewareStage(str, Enum):
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"


class Middleware(Protocol):
    async def wrap(self, stage: MiddlewareStage, payload: Any, nxt: NextFn) -> Any: ...


async def run_chain(
    middlewares: Sequence[Middleware],
    stage: MiddlewareStage,
    payload: Any,
    core: Callable[[Any], Awaitable[T]],
) -> T:
    """Run `middlewares[0]` as the outermost layer around `core`."""

    async def dispatch(index: int, current: Any) -> Any:
        if index >= len(middlewares):
            return await core(current)

        async def nxt(updated: Any = None) -> Any:
            nxt_payload = current if updated is None else updated
            return await dispatch(index + 1, nxt_payload)

        return await middlewares[index].wrap(stage, current, nxt)

    return await dispatch(0, payload)


class PassthroughMiddleware:
    """No-op middleware used as a default so the chain is never empty in tests."""

    async def wrap(self, stage: MiddlewareStage, payload: Any, nxt: NextFn) -> Any:
        del stage
        return await nxt(payload)
