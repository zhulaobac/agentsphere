"""Evaluate → Gate → Release wiring used by tests and the CI hook.

Control Plane stays independent of evaluator implementations; this façade
runs an Experiment, records it, and only then calls ``release`` with a GateDecision.
"""

from __future__ import annotations

from agentsphere.controlplane.release import DeploymentRecord
from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import GateDecision, GateThresholds, Principal, ReleaseChannel
from agentsphere.evaluation.experiment import ExperimentRun, ExperimentRunner, ExperimentSpec
from agentsphere.evaluation.gate import ReleaseGate


class ReleasePipeline:
    def __init__(
        self,
        control: ControlPlane,
        experiments: ExperimentRunner,
        *,
        gate: ReleaseGate | None = None,
    ) -> None:
        self.control = control
        self.experiments = experiments
        self.gate = gate or ReleaseGate()

    async def evaluate_version(
        self,
        spec: ExperimentSpec,
        principal: Principal,
        *,
        thresholds: GateThresholds | None = None,
        baseline: ExperimentRun | None = None,
        mark_lifecycle: bool = True,
    ) -> tuple[ExperimentRun, GateDecision]:
        run = await self.experiments.run(spec, principal)
        decision = self.gate.evaluate(run, thresholds=thresholds, baseline=baseline)
        if mark_lifecycle:
            artifact = await self.control.get_artifact(spec.agent_name, spec.agent_version, principal)
            if artifact.status.phase.value == "Built":
                await self.control.record_evaluation(
                    spec.agent_name,
                    spec.agent_version,
                    principal,
                    experiment_run_id=run.run_id,
                )
        return run, decision

    async def release_if_passing(
        self,
        name: str,
        version: str,
        principal: Principal,
        decision: GateDecision,
        *,
        channel: ReleaseChannel = ReleaseChannel.CANARY,
        approve: bool = True,
    ) -> tuple[AgentArtifact, DeploymentRecord]:
        if approve:
            artifact = await self.control.get_artifact(name, version, principal)
            if artifact.status.phase.value == "Evaluated":
                await self.control.approve(name, version, principal, reason="pipeline")
        return await self.control.release(name, version, principal, decision, channel=channel)
