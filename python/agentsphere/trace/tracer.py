"""Basic Trace: Session → Turn → Step → Model/Tool Call. OpenTelemetry compatible.

Enough to record and inspect a run. Evaluation, experiment, and gate systems are out of scope.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
from pydantic import BaseModel, Field

from agentsphere.kernel.events import Event, EventType


class TraceNode(BaseModel):
    name: str
    kind: str
    span_id: str
    parent_span_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    children: list[TraceNode] = Field(default_factory=list)


class TraceTree(BaseModel):
    session_id: str
    root: TraceNode | None = None


_KIND = {
    EventType.SESSION_CREATED: ("session", "agentsphere.session"),
    EventType.TURN_STARTED: ("turn", "agentsphere.turn"),
    EventType.STEP_STARTED: ("step", "agentsphere.step"),
    EventType.MODEL_CALL_STARTED: ("model.call", "agentsphere.model.call"),
    EventType.TOOL_CALL_STARTED: ("tool.call", "agentsphere.tool.call"),
}

_END = {
    EventType.TURN_COMPLETED: "turn",
    EventType.STEP_COMPLETED: "step",
    EventType.MODEL_CALL_COMPLETED: "model.call",
    EventType.MODEL_CALL_DELTA: None,
    EventType.TOOL_CALL_COMPLETED: "tool.call",
    EventType.TOOL_CALL_FAILED: "tool.call",
}


class TraceRecorder:
    """Event listener that builds an inspectable tree and OpenTelemetry spans."""

    def __init__(self, *, service_name: str = "agentsphere") -> None:
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("agentsphere.trace")
        self._live_spans: dict[str, Span] = {}
        self._nodes: dict[str, dict[str, TraceNode]] = defaultdict(dict)
        self._order: dict[str, list[str]] = defaultdict(list)
        self._stack: dict[str, list[str]] = defaultdict(list)

    async def on_event(self, event: Event) -> None:
        session_id = event.session_id
        if event.type in _KIND:
            kind, otel_name = _KIND[event.type]
            parent_key = self._parent_key(session_id, kind)
            node_key = self._node_key(event, kind)
            parent_span = self._live_spans.get(parent_key) if parent_key else None
            context = set_span_in_context(parent_span) if parent_span is not None else None
            span = self._tracer.start_span(
                otel_name,
                context=context,
                attributes=_otel_attributes(event, kind),
            )
            self._live_spans[node_key] = span
            node = TraceNode(
                name=otel_name,
                kind=kind,
                span_id=node_key,
                parent_span_id=parent_key,
                start_time=event.timestamp,
                attributes={
                    "session_id": event.session_id,
                    "turn_id": event.turn_id,
                    "step_id": event.step_id,
                    "event_type": event.type.value,
                    **{k: v for k, v in event.payload.items() if _is_attr(v)},
                },
            )
            self._nodes[session_id][node_key] = node
            self._order[session_id].append(node_key)
            self._stack[session_id].append(node_key)
            return

        end_kind = _END.get(event.type)
        if end_kind:
            node_key = self._close_key(event, end_kind)
            span = self._live_spans.pop(node_key, None)
            node = self._nodes[session_id].get(node_key)
            if span is not None:
                if event.type is EventType.TOOL_CALL_FAILED or (
                    event.type is EventType.TURN_COMPLETED and event.payload.get("reason") == "error"
                ):
                    span.set_status(Status(StatusCode.ERROR))
                    if node is not None:
                        node.status = "error"
                span.end()
            if node is not None:
                node.end_time = event.timestamp
                node.attributes.update({k: v for k, v in event.payload.items() if _is_attr(v)})
            stack = self._stack[session_id]
            if stack and stack[-1] == node_key:
                stack.pop()
            elif node_key in stack:
                stack.remove(node_key)

    def inspect(self, session_id: str) -> TraceTree:
        nodes = self._nodes.get(session_id, {})
        if not nodes:
            return TraceTree(session_id=session_id)
        by_parent: dict[str | None, list[TraceNode]] = defaultdict(list)
        for node in nodes.values():
            by_parent[node.parent_span_id].append(node)
        roots = by_parent.get(None, [])
        root = roots[0] if roots else next(iter(nodes.values()))

        def attach(node: TraceNode) -> TraceNode:
            node.children = [attach(child) for child in by_parent.get(node.span_id, [])]
            return node

        return TraceTree(session_id=session_id, root=attach(root))

    def exported_spans(self) -> list[Any]:
        return list(self._exporter.get_finished_spans())

    def force_flush(self) -> None:
        self._provider.force_flush()

    def _parent_key(self, session_id: str, kind: str) -> str | None:
        stack = self._stack[session_id]
        if not stack:
            return None
        if kind == "session":
            return None
        return stack[-1]

    def _node_key(self, event: Event, kind: str) -> str:
        if kind == "session":
            return f"session:{event.session_id}"
        if kind == "turn":
            return f"turn:{event.turn_id}"
        if kind == "step":
            return f"step:{event.step_id}"
        if kind == "model.call":
            return f"model:{event.step_id}"
        if kind == "tool.call":
            call_id = event.payload.get("call_id") or event.event_id
            return f"tool:{call_id}"
        return event.event_id

    def _close_key(self, event: Event, kind: str) -> str:
        return self._node_key(event, kind)


def _otel_attributes(event: Event, kind: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "agentsphere.kind": kind,
        "agentsphere.session_id": event.session_id,
        "agentsphere.event_type": event.type.value,
    }
    if event.turn_id:
        attrs["agentsphere.turn_id"] = event.turn_id
    if event.step_id:
        attrs["agentsphere.step_id"] = event.step_id
    return attrs


def _is_attr(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None
