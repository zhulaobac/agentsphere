"""Fixture Build → Evaluate → Gate → Canary → promote Full path for product-team."""

from __future__ import annotations

from typing import Any

from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.controlplane.release import DeploymentRecord
from agentsphere.controlplane.types import GateDecision, GateThresholds, ReleaseChannel
from agentsphere.evaluation.experiment import (
    ArtifactRuntimeFactory,
    ExperimentRun,
    ExperimentRunner,
    ExperimentSpec,
    RuntimeCaseExecutor,
    ScriptedExecutor,
)
from agentsphere.evaluation.metrics import output_from_events
from agentsphere.evaluation.pipeline import ReleasePipeline
from agentsphere.kernel.events import Event
from agentsphere.runtime.entities import Session
from agentsphere.tenants.pack import OnboardedTenant
from agentsphere.tenants.product_team.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    DATASET_NAME,
    DATASET_VERSION,
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
)
from agentsphere.tenants.product_team.dataset import fixture_outputs
from agentsphere.kernel.dispatcher import DispatcherConfig
from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.tenants.product_team.model import ResearchMockProvider


def default_thresholds() -> GateThresholds:
    return GateThresholds(min_quality=1.0, require_safety_pass=True)


def experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        evaluator_name=EVALUATOR_NAME,
        evaluator_version=EVALUATOR_VERSION,
        samples=1,
    )


def scripted_runner(onboarded: OnboardedTenant, outputs: dict[str, str] | None = None) -> ExperimentRunner:
    return ExperimentRunner(
        onboarded.control,
        onboarded.evaluators,
        datasets=onboarded.datasets,
        executor=ScriptedExecutor(outputs if outputs is not None else fixture_outputs()),
    )


def runtime_runner(onboarded: OnboardedTenant) -> ExperimentRunner:
    extra: dict[str, Any] = {}
    if onboarded.identity is not None:
        extra["identity"] = onboarded.identity
    if onboarded.vault is not None:
        extra["vault"] = onboarded.vault
    if onboarded.sandbox is not None:
        extra["sandbox"] = onboarded.sandbox
    if onboarded.policy is not None:
        extra["policy"] = onboarded.policy
    if onboarded.audit is not None:
        extra["audit"] = onboarded.audit
    workspace = onboarded.extra.get("workspace_root")
    if workspace:
        extra["workspace_root"] = workspace
    extra["dispatcher_config"] = DispatcherConfig(timeout_seconds=30.0)
    factory = ArtifactRuntimeFactory(
        adapter=ModelAdapter(ResearchMockProvider()),
        registry=onboarded.tools,
        extra=extra,
    )
    return ExperimentRunner(
        onboarded.control,
        onboarded.evaluators,
        datasets=onboarded.datasets,
        executor=RuntimeCaseExecutor(
            factory,
            system_prompt=str(onboarded.extra.get("system_prompt") or ""),
        ),
    )


async def build_and_evaluate(
    onboarded: OnboardedTenant,
    *,
    runner: ExperimentRunner | None = None,
    outputs: dict[str, str] | None = None,
    thresholds: GateThresholds | None = None,
    use_runtime: bool = False,
) -> tuple[ExperimentRun, GateDecision]:
    principal = onboarded.principal
    artifact = await onboarded.control.get_artifact(
        onboarded.agent_name, onboarded.agent_version, principal
    )
    if artifact.status.phase.value == "Draft":
        await onboarded.control.build(onboarded.agent_name, onboarded.agent_version, principal)
    if runner is None:
        runner = runtime_runner(onboarded) if use_runtime else scripted_runner(onboarded, outputs)
    pipeline = ReleasePipeline(onboarded.control, runner)
    return await pipeline.evaluate_version(
        experiment_spec(),
        principal,
        thresholds=thresholds or default_thresholds(),
    )


async def release_canary_then_full(
    onboarded: OnboardedTenant,
    decision: GateDecision,
    *,
    promote: bool = True,
) -> tuple[AgentArtifact, DeploymentRecord, DeploymentRecord | None]:
    pipeline = ReleasePipeline(onboarded.control, scripted_runner(onboarded))
    artifact, canary = await pipeline.release_if_passing(
        onboarded.agent_name,
        onboarded.agent_version,
        onboarded.principal,
        decision,
        channel=ReleaseChannel.CANARY,
        approve=True,
    )
    full = None
    if promote:
        artifact, full = await onboarded.control.promote(
            onboarded.agent_name,
            onboarded.agent_version,
            onboarded.principal,
        )
    return artifact, canary, full


async def run_fixture_release_path(
    onboarded: OnboardedTenant,
    *,
    outputs: dict[str, str] | None = None,
    use_runtime: bool = False,
    promote: bool = True,
) -> tuple[ExperimentRun, GateDecision, AgentArtifact, DeploymentRecord]:
    run, decision = await build_and_evaluate(
        onboarded, outputs=outputs, use_runtime=use_runtime
    )
    artifact, canary, full = await release_canary_then_full(
        onboarded, decision, promote=promote
    )
    return run, decision, artifact, full or canary


async def run_session(
    onboarded: OnboardedTenant,
    user_input: str,
    *,
    deployment_id: str | None = None,
) -> tuple[Session, list[Event], str]:
    if onboarded.provider is None or onboarded.runtime is None:
        raise RuntimeError("product-team pack was onboarded without a Managed runtime")
    prompt = await onboarded.control._system_prompt_for(onboarded.artifact, onboarded.principal)
    session = await onboarded.provider.create_session(
        system_prompt=prompt,
        metadata={
            "tenant": onboarded.tenant_id,
            "agent_name": onboarded.agent_name,
            "agent_version": onboarded.agent_version,
            "workflow": onboarded.labels.get("workflow", ""),
        },
        deployment_id=deployment_id,
    )
    events = [event async for event in onboarded.provider.run_turn(session.session_id, user_input)]
    stored = await onboarded.provider.list_events(session.session_id)
    output = output_from_events(stored or events)
    return session, events, output
