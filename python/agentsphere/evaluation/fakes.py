"""Tested fakes for evaluators that this slice does not implement for real.

LLM-as-Judge, human review, and a full trajectory evaluator (step correctness,
tool choice, efficiency, safety violations) land later. Production providers
plug in behind the same Evaluator protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentsphere.evaluation.types import EvalResult, EvalSample

DecisionMap = dict[str, EvalResult | bool]


def _as_result(value: EvalResult | bool, evaluator: str, version: str, *, default_reason: str) -> EvalResult:
    if isinstance(value, EvalResult):
        return value.model_copy(
            update={"evaluator": value.evaluator or evaluator, "evaluator_version": value.evaluator_version or version}
        )
    return EvalResult(
        passed=bool(value),
        score=1.0 if value else 0.0,
        reason=default_reason if value else "fake rejected",
        evaluator=evaluator,
        evaluator_version=version,
    )


class FakeLLMJudge:
    """Stand-in for an LLM-as-Judge provider (calibration / consistency later)."""

    name = "llm-judge"
    version = "0.1.0"
    kind = "llm_judge"

    def __init__(
        self,
        decisions: DecisionMap | None = None,
        *,
        default_pass: bool = True,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.decisions = dict(decisions or {})
        self.default_pass = default_pass
        if name:
            self.name = name
        if version:
            self.version = version
        self.calls: list[EvalSample] = []

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        self.calls.append(sample)
        keyed = self.decisions.get(sample.case.case_id)
        if keyed is None:
            keyed = self.decisions.get(sample.output, self.default_pass)
        return _as_result(keyed, self.name, self.version, default_reason="fake llm-judge")


class FakeHumanReview:
    """Stand-in for a human review queue. Decisions are scripted, not collected from UI."""

    name = "human-review"
    version = "0.1.0"
    kind = "human"

    def __init__(
        self,
        decisions: DecisionMap | None = None,
        *,
        default_pass: bool = True,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.decisions = dict(decisions or {})
        self.default_pass = default_pass
        self.submitted: list[EvalSample] = []
        if name:
            self.name = name
        if version:
            self.version = version

    async def submit(self, sample: EvalSample) -> str:
        self.submitted.append(sample)
        return sample.case.case_id

    async def collect(self, review_id: str) -> EvalResult | None:
        decision = self.decisions.get(review_id, self.default_pass)
        return _as_result(decision, self.name, self.version, default_reason="fake human review")

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        review_id = await self.submit(sample)
        result = await self.collect(review_id)
        assert result is not None
        return result


class FakeTrajectoryEvaluator:
    """Stand-in for trajectory scoring: step correctness, tool choice, efficiency, safety."""

    name = "trajectory"
    version = "0.1.0"
    kind = "trajectory"

    def __init__(
        self,
        scorer: Callable[[EvalSample], dict[str, Any]] | None = None,
        *,
        pass_if: Callable[[dict[str, Any]], bool] | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.pass_if = pass_if or (lambda dims: all(bool(v) for v in dims.values()) if dims else True)
        if name:
            self.name = name
        if version:
            self.version = version
        self.calls: list[EvalSample] = []

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        self.calls.append(sample)
        if self.scorer is not None:
            dimensions = self.scorer(sample)
        else:
            expected = sample.case.metadata.get("trajectory") or {}
            dimensions = {
                "step_correctness": expected.get("step_correctness", True),
                "tool_choice": expected.get("tool_choice", True),
                "efficiency": expected.get("efficiency", True),
                "safety_violations": expected.get("safety_violations", 0) == 0,
            }
        passed = self.pass_if(dimensions)
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="ok" if passed else "fake trajectory failed",
            details={},
            dimensions=dimensions,
            evaluator=self.name,
            evaluator_version=self.version,
        )
