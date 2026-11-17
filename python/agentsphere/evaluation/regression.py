"""Coarse regression analysis: overall, by dataset slice, and failure attribution.

Not a full causal engine. Attribution is heuristic from spec diffs and event types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.evaluation.experiment import CaseRun, ExperimentRun
from agentsphere.evaluation.types import DatasetCase
from agentsphere.kernel.events import EventType


class FailureAttribution(str, Enum):
    MODEL = "model"
    PROMPT = "prompt"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


class SliceComparison(BaseModel):
    key: str
    baseline: float
    candidate: float
    delta: float


class AttributedFailure(BaseModel):
    case_id: str
    attribution: FailureAttribution
    reason: str
    slice: dict[str, str] = Field(default_factory=dict)
    baseline_passed: bool = True
    candidate_passed: bool = False


class RegressionReport(BaseModel):
    overall_baseline: float
    overall_candidate: float
    delta: float
    by_slice: list[SliceComparison] = Field(default_factory=list)
    failures: list[AttributedFailure] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def compare_experiments(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    *,
    baseline_artifact: AgentArtifact | None = None,
    candidate_artifact: AgentArtifact | None = None,
) -> RegressionReport:
    base_cases = {c.case.case_id: c for c in baseline.cases}
    cand_cases = {c.case.case_id: c for c in candidate.cases}
    ids = sorted(set(base_cases) | set(cand_cases))
    failures: list[AttributedFailure] = []
    improved: list[str] = []
    for case_id in ids:
        base = base_cases.get(case_id)
        cand = cand_cases.get(case_id)
        if base is None or cand is None:
            continue
        if base.pass_at_k and not cand.pass_at_k:
            failures.append(
                _attribute(
                    cand.case,
                    base,
                    cand,
                    baseline_artifact=baseline_artifact,
                    candidate_artifact=candidate_artifact,
                )
            )
        elif cand.pass_at_k and not base.pass_at_k:
            improved.append(case_id)

    slices = _slice_keys(list(base_cases.values()) + list(cand_cases.values()))
    by_slice: list[SliceComparison] = []
    for key in slices:
        b = _slice_rate(baseline, key)
        c = _slice_rate(candidate, key)
        if b is None or c is None:
            continue
        by_slice.append(SliceComparison(key=key, baseline=b, candidate=c, delta=c - b))

    return RegressionReport(
        overall_baseline=baseline.summary.pass_rate,
        overall_candidate=candidate.summary.pass_rate,
        delta=candidate.summary.pass_rate - baseline.summary.pass_rate,
        by_slice=by_slice,
        failures=failures,
        improved=improved,
        details={
            "baseline_run_id": baseline.run_id,
            "candidate_run_id": candidate.run_id,
            "baseline_fingerprint": baseline.summary.fingerprint,
            "candidate_fingerprint": candidate.summary.fingerprint,
        },
    )


def _slice_keys(cases: list[CaseRun]) -> list[str]:
    keys: set[str] = set()
    for case_run in cases:
        tags = case_run.case.slice
        if tags.scene:
            keys.add(f"scene:{tags.scene}")
        if tags.difficulty:
            keys.add(f"difficulty:{tags.difficulty}")
        if tags.risk:
            keys.add(f"risk:{tags.risk}")
    return sorted(keys)


def _slice_rate(run: ExperimentRun, key: str) -> float | None:
    matching = [c for c in run.cases if _case_has_slice(c.case, key)]
    if not matching:
        return None
    return sum(c.pass_fraction for c in matching) / len(matching)


def _case_has_slice(case: DatasetCase, key: str) -> bool:
    kind, _, value = key.partition(":")
    if kind == "scene":
        return case.slice.scene == value
    if kind == "difficulty":
        return case.slice.difficulty == value
    if kind == "risk":
        return case.slice.risk == value
    return False


def _attribute(
    case: DatasetCase,
    baseline: CaseRun,
    candidate: CaseRun,
    *,
    baseline_artifact: AgentArtifact | None,
    candidate_artifact: AgentArtifact | None,
) -> AttributedFailure:
    reason_parts: list[str] = []
    attribution = FailureAttribution.UNKNOWN
    cand_events = candidate.samples[-1].events if candidate.samples else []
    base_events = baseline.samples[-1].events if baseline.samples else []
    cand_tool_fail = any(e.type is EventType.TOOL_CALL_FAILED for e in cand_events)
    base_tool_fail = any(e.type is EventType.TOOL_CALL_FAILED for e in base_events)
    if cand_tool_fail and not base_tool_fail:
        attribution = FailureAttribution.TOOL
        reason_parts.append("tool.call.failed in candidate")
    elif baseline_artifact is not None and candidate_artifact is not None:
        if baseline_artifact.spec.prompts != candidate_artifact.spec.prompts:
            attribution = FailureAttribution.PROMPT
            reason_parts.append("prompt refs changed")
        elif baseline_artifact.spec.tools != candidate_artifact.spec.tools:
            attribution = FailureAttribution.TOOL
            reason_parts.append("tool refs changed")
        elif baseline_artifact.spec.model != candidate_artifact.spec.model:
            attribution = FailureAttribution.MODEL
            reason_parts.append("model spec changed")
        elif (baseline_artifact.spec.context != candidate_artifact.spec.context) or (
            baseline_artifact.spec.memory != candidate_artifact.spec.memory
        ):
            attribution = FailureAttribution.RETRIEVAL
            reason_parts.append("context/memory refs changed")
        else:
            attribution = FailureAttribution.MODEL
            reason_parts.append("output regressed with same spec (coarse: model)")
    else:
        attribution = FailureAttribution.MODEL
        reason_parts.append("candidate failed a case the baseline passed")
    return AttributedFailure(
        case_id=case.case_id,
        attribution=attribution,
        reason="; ".join(reason_parts) or "regressed",
        slice={
            "scene": case.slice.scene,
            "difficulty": case.slice.difficulty,
            "risk": case.slice.risk,
        },
        baseline_passed=baseline.pass_at_k,
        candidate_passed=candidate.pass_at_k,
    )
