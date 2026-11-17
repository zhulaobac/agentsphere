from __future__ import annotations

from agentsphere.controlplane.types import GateThresholds
from agentsphere.evaluation.ci_gate import run_fixture_gate
from agentsphere.evaluation.fixtures import fixture_outputs


async def test_fixture_gate_passes_in_ci() -> None:
    code = await run_fixture_gate(thresholds=GateThresholds(min_quality=1.0))
    assert code == 0


async def test_fixture_gate_fails_when_outputs_regress() -> None:
    outputs = fixture_outputs()
    outputs["safety-no-secret"] = "TOKEN"
    code = await run_fixture_gate(outputs=outputs, thresholds=GateThresholds(min_quality=1.0))
    assert code == 1
