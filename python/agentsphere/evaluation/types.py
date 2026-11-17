"""Evaluation sample/result types. Traces reuse the Runtime Event schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentsphere.kernel.events import Event
from agentsphere.kernel.ids import new_id


class SliceTags(BaseModel):
    scene: str = ""
    difficulty: str = ""
    risk: str = ""


class AssertionSpec(BaseModel):
    type: Literal["equals", "contains", "not_contains", "regex", "json_equals"] = "equals"
    value: Any = None
    field: str = "output"


class DatasetCase(BaseModel):
    case_id: str = Field(default_factory=lambda: new_id("case"))
    input: str
    expected: Any = None
    assertions: list[AssertionSpec] = Field(default_factory=list)
    slice: SliceTags = Field(default_factory=SliceTags)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_safety(self) -> bool:
        if self.slice.risk.lower() in {"safety", "safe"}:
            return True
        flag = self.labels.get("safety") or self.labels.get("risk")
        return str(flag).lower() in {"true", "1", "safety"}


class SampleMetrics(BaseModel):
    """Optional cost/latency extracted from events or injected by a fake executor."""

    latency_ms: float | None = None
    cost: float | None = None
    tokens: float | None = None


class EvalSample(BaseModel):
    case: DatasetCase
    output: str = ""
    events: list[Event] = Field(default_factory=list)
    metrics: SampleMetrics = Field(default_factory=SampleMetrics)
    agent_name: str = ""
    agent_version: str = ""
    sample_index: int = 0


class EvalResult(BaseModel):
    passed: bool
    score: float | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    evaluator: str = ""
    evaluator_version: str = "0.1.0"
    dimensions: dict[str, Any] = Field(default_factory=dict)


class EnvironmentConfig(BaseModel):
    name: str = "offline"
    labels: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
