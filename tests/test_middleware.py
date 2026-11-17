from __future__ import annotations

from typing import Any

from agentsphere.kernel.middleware import MiddlewareStage, run_chain


class RecordingMiddleware:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    async def wrap(self, stage: MiddlewareStage, payload: Any, nxt: Any) -> Any:
        self.log.append(f"{self.name}:enter:{stage.value}")
        result = await nxt(payload)
        self.log.append(f"{self.name}:exit:{stage.value}")
        return result


async def test_onion_order() -> None:
    log: list[str] = []
    outer = RecordingMiddleware("a", log)
    inner = RecordingMiddleware("b", log)

    async def core(payload: str) -> str:
        log.append(f"core:{payload}")
        return payload.upper()

    result = await run_chain([outer, inner], MiddlewareStage.BEFORE_MODEL, "hi", core)
    assert result == "HI"
    assert log == [
        "a:enter:before_model",
        "b:enter:before_model",
        "core:hi",
        "b:exit:before_model",
        "a:exit:before_model",
    ]
