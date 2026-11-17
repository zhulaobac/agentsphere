"""AgentHarness: compose adapter, loop, context, tools, middleware, events.

The harness is essentially pure: it does not keep durable in-process session
state. ``run_turn(session_state, user_input)`` works from the provided snapshot
and yields events; Runtime is responsible for persisting them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentsphere.kernel.actions import CallModel, CallTool, Finish, Handoff, ToolCallRequest
from agentsphere.kernel.assembler import ContextAssembler
from agentsphere.kernel.context import SessionState, TurnContext
from agentsphere.kernel.dispatcher import ToolDispatcher, ToolResult
from agentsphere.kernel.events import Event, EventEmitter, EventType
from agentsphere.kernel.ids import new_id
from agentsphere.kernel.loop.strategy import LoopStrategy
from agentsphere.kernel.middleware import Middleware, MiddlewareStage, run_chain
from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.model_adapter.types import (
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
)


class AgentHarness:
    def __init__(
        self,
        strategy: LoopStrategy,
        model_adapter: ModelAdapter,
        *,
        assembler: ContextAssembler | None = None,
        dispatcher: ToolDispatcher | None = None,
        middlewares: Sequence[Middleware] | None = None,
        event_emitter: EventEmitter | None = None,
        max_steps: int = 20,
        tool_definitions: Sequence[ToolDefinition] | None = None,
    ) -> None:
        self.strategy = strategy
        self.model_adapter = model_adapter
        self.assembler = assembler or ContextAssembler()
        self.dispatcher = dispatcher
        self.middlewares = list(middlewares or [])
        self.event_emitter = event_emitter or EventEmitter()
        self.max_steps = max_steps
        self.tool_definitions = list(tool_definitions or [])

    async def run_turn(self, session_state: SessionState, user_input: str) -> AsyncIterator[Event]:
        state = session_state.model_copy(deep=True)
        turn_id = new_id("turn")
        turn_index = state.turn_index + 1
        state.turn_index = turn_index
        state.messages.append(Message(role=MessageRole.USER, content=user_input))
        # P0 has no wait-for-user / resume node: every turn restarts the workflow DAG.
        state.workflow_cursor = None
        state.workflow_node_outputs = {}

        async def emit(**kwargs: object) -> Event:
            event = self.event_emitter.make(**kwargs)  # type: ignore[arg-type]
            await self.event_emitter.emit(event)
            return event

        yield await emit(
            type=EventType.TURN_STARTED,
            session_id=state.session_id,
            turn_id=turn_id,
            payload={"user_input": user_input, "turn_index": turn_index, "strategy": getattr(self.strategy, "name", "")},
        )

        last_model: ModelResponse | None = None
        last_tools: list[dict[str, object]] | None = None
        pending: list[ToolCallRequest] | None = None
        final_output = ""
        terminal_reason = "completed"

        try:
            for step_index in range(self.max_steps):
                step_id = new_id("step")
                assembled = self.assembler.assemble(
                    system_prompt=state.system_prompt,
                    tools=self.tool_definitions,
                    history=_history_without_system(state.messages),
                )

                ctx = TurnContext(
                    session_state=state,
                    user_input=user_input,
                    turn_id=turn_id,
                    step_index=step_index,
                    assembled=assembled,
                    last_model_response=last_model,
                    last_tool_results=last_tools,
                    pending_tool_calls=pending,
                )
                action = self.strategy.step(ctx)
                state = ctx.session_state

                yield await emit(
                    type=EventType.STEP_STARTED,
                    session_id=state.session_id,
                    turn_id=turn_id,
                    step_id=step_id,
                    payload={"index": step_index, "action": action.type},
                )

                if isinstance(action, CallModel):
                    request = ModelRequest(
                        messages=action.messages,
                        tools=action.tools,
                        params=action.params,
                    )
                    started = await emit(
                        type=EventType.MODEL_CALL_STARTED,
                        session_id=state.session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={
                            "provider": self.model_adapter.primary.name,
                            "tool_count": len(request.tools),
                            "message_count": len(request.messages),
                            "stream": request.params.stream,
                        },
                    )
                    yield started
                    try:
                        response = await self._call_model(request, state, turn_id, step_id)
                    except Exception as exc:
                        yield await emit(
                            type=EventType.MODEL_CALL_FAILED,
                            session_id=state.session_id,
                            turn_id=turn_id,
                            step_id=step_id,
                            payload={
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "provider": self.model_adapter.primary.name,
                            },
                            parent_event_id=started.event_id,
                        )
                        raise
                    async for event in self._model_completion_events(
                        state.session_id, turn_id, step_id, request, response, started.event_id
                    ):
                        yield event
                    last_model = response
                    pending = _pending_from_response(response, state.tool_pins)
                    last_tools = None
                    state.messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content,
                            tool_calls=response.tool_calls or None,
                        )
                    )
                    self._record_workflow_output(
                        state,
                        {
                            "content": response.content,
                            "tool_calls": [c.model_dump() for c in response.tool_calls],
                        },
                    )
                elif isinstance(action, CallTool):
                    if self.dispatcher is None:
                        raise RuntimeError("CallTool requires a ToolDispatcher")
                    started_by_id: dict[str, Event] = {}
                    for tool_call in action.calls:
                        started = await emit(
                            type=EventType.TOOL_CALL_STARTED,
                            session_id=state.session_id,
                            turn_id=turn_id,
                            step_id=step_id,
                            payload={
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "version": tool_call.version or "",
                            },
                        )
                        yield started
                        started_by_id[tool_call.id] = started
                    try:
                        results = await self.dispatcher.dispatch(
                            action.calls,
                            session_id=state.session_id,
                            turn_id=turn_id,
                            step_id=step_id,
                        )
                    except Exception as exc:
                        for tool_call in action.calls:
                            parent = started_by_id[tool_call.id]
                            yield await emit(
                                type=EventType.TOOL_CALL_FAILED,
                                session_id=state.session_id,
                                turn_id=turn_id,
                                step_id=step_id,
                                payload={
                                    "call_id": tool_call.id,
                                    "name": tool_call.name,
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                },
                                parent_event_id=parent.event_id,
                            )
                        raise
                    async for event in self._tool_result_events(
                        state.session_id, turn_id, step_id, results, started_by_id
                    ):
                        yield event
                    last_tools = [r.model_dump() for r in results]
                    pending = None
                    last_model = None
                    for result in results:
                        state.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                name=result.name,
                                tool_call_id=result.call_id,
                                content=result.output if result.error is None else f"ERROR: {result.error}",
                            )
                        )
                    self._record_workflow_output(
                        state,
                        last_tools[0] if last_tools else {},
                    )
                elif isinstance(action, Finish):
                    final_output = action.output
                    terminal_reason = action.reason
                    yield await emit(
                        type=EventType.STEP_COMPLETED,
                        session_id=state.session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={"index": step_index, "action": action.type, "reason": action.reason},
                    )
                    break
                elif isinstance(action, Handoff):
                    terminal_reason = "handoff"
                    final_output = ""
                    yield await emit(
                        type=EventType.HANDOFF,
                        session_id=state.session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={"target": action.target, "handoff_payload": action.payload},
                    )
                    yield await emit(
                        type=EventType.STEP_COMPLETED,
                        session_id=state.session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={"index": step_index, "action": action.type},
                    )
                    break
                else:
                    raise TypeError(f"unsupported action: {type(action)!r}")

                yield await emit(
                    type=EventType.STEP_COMPLETED,
                    session_id=state.session_id,
                    turn_id=turn_id,
                    step_id=step_id,
                    payload={"index": step_index, "action": action.type},
                )
            else:
                terminal_reason = "max_steps"
                if last_model is not None:
                    final_output = last_model.content or final_output

            yield await emit(
                type=EventType.OUTPUT,
                session_id=state.session_id,
                turn_id=turn_id,
                payload={"text": final_output, "reason": terminal_reason},
            )
            yield await emit(
                type=EventType.TURN_COMPLETED,
                session_id=state.session_id,
                turn_id=turn_id,
                payload=_turn_completed_payload(
                    state,
                    turn_index=turn_index,
                    output=final_output,
                    reason=terminal_reason,
                ),
            )
        except Exception as exc:
            yield await emit(
                type=EventType.ERROR,
                session_id=state.session_id,
                turn_id=turn_id,
                payload={"error": str(exc), "error_type": type(exc).__name__},
            )
            yield await emit(
                type=EventType.TURN_COMPLETED,
                session_id=state.session_id,
                turn_id=turn_id,
                payload=_turn_completed_payload(
                    state,
                    turn_index=turn_index,
                    output="",
                    reason="error",
                ),
            )
            raise

    async def _call_model(
        self,
        request: ModelRequest,
        state: SessionState,
        turn_id: str,
        step_id: str,
    ) -> ModelResponse:
        del state, turn_id, step_id

        async def invoke(req: ModelRequest) -> ModelResponse:
            if req.params.stream:
                collected = ""
                final: ModelResponse | None = None
                async for chunk in self.model_adapter.stream(req):
                    collected += chunk.delta
                    if chunk.response is not None:
                        final = chunk.response
                return final or ModelResponse(
                    content=collected,
                    provider=self.model_adapter.primary.name,
                )
            return await self.model_adapter.complete(req)

        response = await run_chain(
            self.middlewares,
            MiddlewareStage.BEFORE_MODEL,
            request,
            invoke,
        )

        async def after_core(resp: ModelResponse) -> ModelResponse:
            return resp

        return await run_chain(
            self.middlewares,
            MiddlewareStage.AFTER_MODEL,
            response,
            after_core,
        )

    async def _model_completion_events(
        self,
        session_id: str,
        turn_id: str,
        step_id: str,
        request: ModelRequest,
        response: ModelResponse,
        parent_event_id: str,
    ) -> AsyncIterator[Event]:
        if request.params.stream and response.content:
            yield await self.event_emitter.emit(
                self.event_emitter.make(
                    EventType.MODEL_CALL_DELTA,
                    session_id=session_id,
                    turn_id=turn_id,
                    step_id=step_id,
                    payload={"delta": response.content},
                    parent_event_id=parent_event_id,
                )
            )
        yield await self.event_emitter.emit(
            self.event_emitter.make(
                EventType.MODEL_CALL_COMPLETED,
                session_id=session_id,
                turn_id=turn_id,
                step_id=step_id,
                payload={
                    "content": response.content,
                    "tool_calls": [c.model_dump() for c in response.tool_calls],
                    "usage": response.usage.model_dump(),
                    "provider": response.provider,
                },
                parent_event_id=parent_event_id,
            )
        )

    async def _tool_result_events(
        self,
        session_id: str,
        turn_id: str,
        step_id: str,
        results: list[ToolResult],
        started_by_id: dict[str, Event],
    ) -> AsyncIterator[Event]:
        for result in results:
            parent = started_by_id.get(result.call_id)
            parent_event_id = parent.event_id if parent is not None else None
            if result.error:
                yield await self.event_emitter.emit(
                    self.event_emitter.make(
                        EventType.TOOL_CALL_FAILED,
                        session_id=session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={
                            "call_id": result.call_id,
                            "name": result.name,
                            "error": result.error,
                            "attempts": result.attempts,
                        },
                        parent_event_id=parent_event_id,
                    )
                )
            else:
                yield await self.event_emitter.emit(
                    self.event_emitter.make(
                        EventType.TOOL_CALL_COMPLETED,
                        session_id=session_id,
                        turn_id=turn_id,
                        step_id=step_id,
                        payload={
                            "call_id": result.call_id,
                            "name": result.name,
                            "output": result.output,
                            "truncated": result.truncated,
                            "attempts": result.attempts,
                            "idempotency_key": result.idempotency_key,
                        },
                        parent_event_id=parent_event_id,
                    )
                )

    def _record_workflow_output(self, state: SessionState, output: object) -> None:
        if not state.workflow_cursor:
            return
        state.workflow_node_outputs[state.workflow_cursor] = output


def _pending_from_response(response: ModelResponse, pins: dict[str, str]) -> list[ToolCallRequest] | None:
    if not response.tool_calls:
        return None
    return [
        ToolCallRequest(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            version=pins.get(call.name),
        )
        for call in response.tool_calls
    ]


def _history_without_system(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m.role != MessageRole.SYSTEM]


def _turn_completed_payload(
    state: SessionState,
    *,
    turn_index: int,
    output: str,
    reason: str,
) -> dict[str, object]:
    return {
        "output": output,
        "reason": reason,
        "turn_index": turn_index,
        "workflow_cursor": state.workflow_cursor,
        "workflow_node_outputs": state.workflow_node_outputs,
        "messages": [m.model_dump(mode="json") for m in state.messages],
    }
