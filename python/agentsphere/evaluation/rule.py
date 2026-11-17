"""Working rule/assertion evaluator (equals, contains, regex, json)."""

from __future__ import annotations

import json
import re
from typing import Any

from agentsphere.evaluation.types import AssertionSpec, EvalResult, EvalSample


class RuleAssertionEvaluator:
    name = "rule-assertion"
    version = "0.1.0"
    kind = "rule"

    def __init__(self, *, name: str | None = None, version: str | None = None) -> None:
        if name:
            self.name = name
        if version:
            self.version = version

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        assertions = list(sample.case.assertions)
        if not assertions and sample.case.expected is not None:
            assertions = [AssertionSpec(type="equals", value=sample.case.expected, field="output")]
        if not assertions:
            return EvalResult(
                passed=True,
                score=1.0,
                reason="no assertions",
                evaluator=self.name,
                evaluator_version=self.version,
            )

        failures: list[str] = []
        for spec in assertions:
            actual = _field(sample, spec.field)
            if not _check(spec, actual):
                failures.append(f"{spec.type} {spec.field} expected {spec.value!r} got {actual!r}")
        passed = not failures
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="ok" if passed else "; ".join(failures),
            details={"failures": failures, "assertion_count": len(assertions)},
            evaluator=self.name,
            evaluator_version=self.version,
        )


def _field(sample: EvalSample, field: str) -> Any:
    if field == "output":
        return sample.output
    if field == "expected":
        return sample.case.expected
    if field.startswith("metadata."):
        return sample.case.metadata.get(field.split(".", 1)[1])
    return sample.output


def _check(spec: AssertionSpec, actual: Any) -> bool:
    if spec.type == "equals":
        return actual == spec.value or str(actual) == str(spec.value)
    if spec.type == "contains":
        return str(spec.value) in str(actual)
    if spec.type == "not_contains":
        return str(spec.value) not in str(actual)
    if spec.type == "regex":
        return re.search(str(spec.value), str(actual) or "") is not None
    if spec.type == "json_equals":
        try:
            parsed = actual if not isinstance(actual, str) else json.loads(actual)
        except json.JSONDecodeError:
            return False
        return parsed == spec.value
    return False
