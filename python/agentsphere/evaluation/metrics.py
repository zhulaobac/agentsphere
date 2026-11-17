"""Extract cost/latency from Runtime events when those fields are present."""

from __future__ import annotations

from agentsphere.evaluation.types import SampleMetrics
from agentsphere.kernel.events import Event, EventType


def output_from_events(events: list[Event]) -> str:
    for event in reversed(events):
        if event.type is EventType.OUTPUT:
            return str(event.payload.get("text", "") or "")
        if event.type is EventType.TURN_COMPLETED:
            return str(event.payload.get("output") or "")
    return ""


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def metrics_from_events(events: list[Event]) -> SampleMetrics:
    tokens = 0.0
    saw_usage = False
    for event in events:
        if event.type is EventType.MODEL_CALL_COMPLETED:
            usage = event.payload.get("usage") or {}
            inp = _numeric(usage.get("input_tokens"))
            out = _numeric(usage.get("output_tokens"))
            if inp is None and out is None:
                continue
            saw_usage = True
            tokens += (inp or 0.0) + (out or 0.0)
    latency_ms = None
    started = next((e for e in events if e.type is EventType.TURN_STARTED), None)
    completed = next((e for e in events if e.type is EventType.TURN_COMPLETED), None)
    if started is not None and completed is not None and started.timestamp and completed.timestamp:
        latency_ms = (completed.timestamp - started.timestamp).total_seconds() * 1000.0
    cost = tokens if saw_usage else None
    return SampleMetrics(
        latency_ms=latency_ms,
        cost=cost,
        tokens=tokens if saw_usage else None,
    )
