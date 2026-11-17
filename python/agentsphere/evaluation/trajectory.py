"""Replay and coarse step comparison using the Runtime Event schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentsphere.kernel.events import Event, EventType
from agentsphere.runtime.replay import replay


class StepRecord(BaseModel):
    step_id: str | None = None
    types: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_ok: bool | None = None
    output: str | None = None


class TrajectoryDiff(BaseModel):
    matching: bool
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    extra_tools: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    type_mismatch: bool = False
    safety_violations: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


def events_to_steps(events: list[Event]) -> list[StepRecord]:
    by_step: dict[str, StepRecord] = {}
    order: list[str] = []
    for event in events:
        key = event.step_id or event.event_id
        if key not in by_step:
            by_step[key] = StepRecord(step_id=event.step_id)
            order.append(key)
        record = by_step[key]
        record.types.append(event.type.value)
        if event.type is EventType.TOOL_CALL_COMPLETED:
            record.tool_name = str(event.payload.get("name") or record.tool_name or "")
            record.tool_ok = True
            record.output = str(event.payload.get("output", ""))
        elif event.type is EventType.TOOL_CALL_FAILED:
            record.tool_name = str(event.payload.get("name") or record.tool_name or "")
            record.tool_ok = False
            record.output = str(event.payload.get("error", ""))
        elif event.type is EventType.OUTPUT:
            record.output = str(event.payload.get("text", record.output or ""))
    return [by_step[key] for key in order]


def tool_names(events: list[Event]) -> list[str]:
    names: list[str] = []
    for event in events:
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_STARTED}:
            name = event.payload.get("name")
            if name and event.type is EventType.TOOL_CALL_STARTED:
                names.append(str(name))
    if not names:
        for event in events:
            if event.type is EventType.TOOL_CALL_COMPLETED and event.payload.get("name"):
                names.append(str(event.payload["name"]))
    return names


def safety_violations(events: list[Event]) -> int:
    count = 0
    for event in events:
        if event.type is EventType.POLICY_DECISION:
            decision = str(event.payload.get("decision") or event.payload.get("effect") or "").lower()
            if decision in {"deny", "denied"}:
                count += 1
        if event.type is EventType.TOOL_CALL_FAILED and str(event.payload.get("error", "")).lower().find("denied") >= 0:
            count += 1
    return count


def compare_trajectories(
    actual: list[Event],
    *,
    expected_tools: list[str] | None = None,
    expected_types: list[str] | None = None,
    baseline: list[Event] | None = None,
) -> TrajectoryDiff:
    actual_tools = tool_names(actual)
    expected = list(expected_tools or tool_names(baseline or []))
    extra = [t for t in actual_tools if t not in expected]
    missing = [t for t in expected if t not in actual_tools]
    type_mismatch = False
    if expected_types is not None:
        actual_types = [e.type.value for e in actual]
        type_mismatch = actual_types != expected_types
    elif baseline is not None:
        type_mismatch = [e.type.value for e in actual] != [e.type.value for e in baseline]
    matching = extra == [] and missing == [] and not type_mismatch
    return TrajectoryDiff(
        matching=matching,
        expected_tools=expected,
        actual_tools=actual_tools,
        extra_tools=extra,
        missing_tools=missing,
        type_mismatch=type_mismatch,
        safety_violations=safety_violations(actual),
        details={"step_count": len(events_to_steps(actual))},
    )


def replay_session(events: list[Event]):
    """Rebuild SessionState from experiment traces using the Runtime fold."""
    return replay(events)


class EventTrajectoryEvaluator:
    """Working (coarse) trajectory check against expected tools / event types.

    Not a full causal engine. Complements FakeTrajectoryEvaluator as the
    interface that later step-correctness / efficiency models can replace.
    """

    name = "event-trajectory"
    version = "0.1.0"
    kind = "trajectory"

    async def evaluate(self, sample):
        from agentsphere.evaluation.types import EvalResult

        expected_tools = list(sample.case.metadata.get("expected_tools") or [])
        expected_types = sample.case.metadata.get("expected_event_types")
        diff = compare_trajectories(
            sample.events,
            expected_tools=expected_tools or None,
            expected_types=list(expected_types) if expected_types else None,
        )
        efficiency_ok = True
        max_tools = sample.case.metadata.get("max_tools")
        if max_tools is not None:
            efficiency_ok = len(diff.actual_tools) <= int(max_tools)
        safety_ok = diff.safety_violations == 0
        passed = diff.matching and efficiency_ok and safety_ok
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="ok" if passed else "trajectory mismatch",
            details=diff.model_dump(mode="json"),
            dimensions={
                "step_correctness": not diff.type_mismatch,
                "tool_choice": diff.missing_tools == [] and diff.extra_tools == [],
                "efficiency": efficiency_ok,
                "safety_violations": diff.safety_violations,
            },
            evaluator=self.name,
            evaluator_version=self.version,
        )
