"""CI hook: run a fixture offline Experiment + Release Gate.

Exit 0 if the gate passes, 1 otherwise. No external services.

    python -m agentsphere.evaluation.ci_gate
"""

from __future__ import annotations

import asyncio
import json
import sys

from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import AssetKind, AssetMetadata, GateThresholds
from agentsphere.evaluation.experiment import ExperimentRunner, ExperimentSpec, ScriptedExecutor
from agentsphere.evaluation.evaluator import EvaluatorRegistry
from agentsphere.evaluation.fixtures import (
    FIXTURE_CASES,
    FIXTURE_PRINCIPAL,
    fixture_artifact,
    fixture_outputs,
)
from agentsphere.evaluation.gate import ReleaseGate
from agentsphere.evaluation.rule import RuleAssertionEvaluator


async def run_fixture_gate(
    *,
    thresholds: GateThresholds | None = None,
    outputs: dict[str, str] | None = None,
) -> int:
    principal = FIXTURE_PRINCIPAL
    control = ControlPlane()
    artifact = fixture_artifact()
    await control.put_artifact(artifact, principal)
    await control.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="fixture-prompt", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "You are a fixture agent."},
        principal=principal,
    )
    from agentsphere.evaluation.dataset import DatasetService

    datasets = DatasetService(control)
    await datasets.create(
        name="fixture-dataset",
        version="1.0.0",
        principal=principal,
        cases=list(FIXTURE_CASES),
        labels={"source": "manual"},
    )
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())
    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=ScriptedExecutor(outputs or fixture_outputs()),
    )
    spec = ExperimentSpec(
        agent_name=artifact.metadata.name,
        agent_version=artifact.version,
        dataset_name="fixture-dataset",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
        samples=1,
    )
    run = await runner.run(spec, principal)
    decision = ReleaseGate().evaluate(run, thresholds=thresholds or GateThresholds(min_quality=1.0))
    payload = {
        "passed": decision.passed,
        "reason": decision.reason,
        "checks": [c.model_dump(mode="json") for c in decision.checks],
        "experiment_run_id": run.run_id,
        "fingerprint": run.summary.fingerprint,
        "pass_rate": run.summary.pass_rate,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if decision.passed else 1


def main(argv: list[str] | None = None) -> int:
    del argv
    return asyncio.run(run_fixture_gate())


if __name__ == "__main__":
    sys.exit(main())
