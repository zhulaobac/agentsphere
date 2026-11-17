from __future__ import annotations

from typing import Any

from agentsphere.kernel.events import EventType
from agentsphere.kernel.loop import ReActStrategy
from agentsphere.kernel.middleware import MiddlewareStage, run_chain
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text, assistant_tool_response
from agentsphere.runtime import InMemoryEventStore, build_runtime


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


async def test_middleware_hooks_fire_in_onion_order_during_react_turn(registry) -> None:
    log: list[str] = []
    outer = RecordingMiddleware("a", log)
    inner = RecordingMiddleware("b", log)
    runtime = build_runtime(
        adapter=ModelAdapter(
            MockModelProvider(
                script=[
                    assistant_tool_response("echo", {"text": "hi"}),
                    assistant_text("echoed: hi"),
                ]
            )
        ),
        registry=registry,
        strategy=ReActStrategy(),
        store=InMemoryEventStore(),
        middlewares=[outer, inner],
    )
    session = await runtime.create_session()
    events = [event async for event in runtime.run_turn(session.session_id, "please echo hi")]
    assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in events)

    stages = [MiddlewareStage.BEFORE_MODEL, MiddlewareStage.AFTER_MODEL, MiddlewareStage.BEFORE_TOOL, MiddlewareStage.AFTER_TOOL]
    for stage in stages:
        assert f"a:enter:{stage.value}" in log
        assert f"a:exit:{stage.value}" in log

    outer_enters = [item.split(":")[2] for item in log if item.startswith("a:enter:")]
    assert outer_enters == [
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "before_model",
        "after_model",
    ]

    # Outer wraps inner for every stage: a enter, b enter, b exit, a exit.
    i = 0
    while i < len(log):
        prefix, edge, stage = log[i].split(":")
        assert prefix == "a" and edge == "enter"
        assert log[i + 1] == f"b:enter:{stage}"
        assert log[i + 2] == f"b:exit:{stage}"
        assert log[i + 3] == f"a:exit:{stage}"
        i += 4
    assert i == len(log)
