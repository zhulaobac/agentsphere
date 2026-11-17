from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    AssetMetadata,
    ControlPlane,
    GateFailedError,
    GateThresholds,
    LifecyclePhase,
    Principal,
    ReleaseChannel,
)
from agentsphere.evaluation import (
    DatasetCase,
    DatasetService,
    EvaluatorRegistry,
    ExperimentRunner,
    ExperimentSpec,
    ReleaseGate,
    ReleasePipeline,
    RuleAssertionEvaluator,
    ScriptedExecutor,
    SliceTags,
)
from agentsphere.evaluation.types import SampleMetrics


def _cases() -> list[DatasetCase]:
    return [
        DatasetCase(
            case_id="q1",
            input="a",
            expected="ok",
            slice=SliceTags(scene="main", difficulty="easy", risk="low"),
        ),
        DatasetCase(
            case_id="safe",
            input="b",
            expected="denied",
            slice=SliceTags(scene="sec", difficulty="hard", risk="safety"),
            labels={"safety": "true"},
        ),
    ]


async def _runner(principal: Principal, outputs: dict[str, str], metrics: dict | None = None):
    control = ControlPlane()
    await control.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="gated", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(),
            version="0.1.0",
        ),
        principal,
    )
    datasets = DatasetService(control)
    await datasets.create(name="ds", version="1.0.0", principal=principal, cases=_cases())
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())
    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=ScriptedExecutor(outputs, metrics=metrics or {}),
    )
    spec = ExperimentSpec(
        agent_name="gated",
        agent_version="0.1.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
    )
    return control, runner, spec


async def test_gate_quality_safety_regression_and_budgets(principal: Principal) -> None:
    gate = ReleaseGate()
    control, runner, spec = await _runner(principal, {"q1": "ok", "safe": "denied"})
    good = await runner.run(spec, principal)
    passed = gate.evaluate(good, thresholds=GateThresholds(min_quality=1.0))
    assert passed.passed is True

    _, bad_runner, bad_spec = await _runner(principal, {"q1": "ok", "safe": "leaked"})
    unsafe = await bad_runner.run(bad_spec, principal)
    safety = gate.evaluate(unsafe, thresholds=GateThresholds(min_quality=0.0))
    assert safety.passed is False
    assert any(c.name == "safety" and not c.passed for c in safety.checks)

    _, low_runner, low_spec = await _runner(principal, {"q1": "nope", "safe": "denied"})
    low = await low_runner.run(low_spec, principal)
    quality = gate.evaluate(low, thresholds=GateThresholds(min_quality=1.0))
    assert quality.passed is False
    assert any(c.name == "quality" and not c.passed for c in quality.checks)

    regression = gate.evaluate(
        low,
        baseline=good,
        thresholds=GateThresholds(min_quality=0.0, max_regression=0.01),
    )
    assert regression.passed is False
    assert any(c.name == "regression" and not c.passed for c in regression.checks)

    _, cost_runner, cost_spec = await _runner(
        principal,
        {"q1": "ok", "safe": "denied"},
        metrics={"q1": SampleMetrics(cost=50.0, latency_ms=5.0), "safe": SampleMetrics(cost=50.0, latency_ms=5.0)},
    )
    costly = await cost_runner.run(cost_spec, principal)
    cost_fail = gate.evaluate(costly, thresholds=GateThresholds(min_quality=1.0, max_cost=10.0))
    assert cost_fail.passed is False
    latency_skip = gate.evaluate(good, thresholds=GateThresholds(min_quality=1.0, max_latency_ms=1.0))
    assert any(c.name == "latency" and c.skipped for c in latency_skip.checks)
    del control


async def test_failing_gate_blocks_control_plane_release(principal: Principal) -> None:
    control, runner, spec = await _runner(principal, {"q1": "ok", "safe": "denied"})
    await control.build("gated", "0.1.0", principal)
    pipeline = ReleasePipeline(control, runner)
    run, decision = await pipeline.evaluate_version(
        spec, principal, thresholds=GateThresholds(min_quality=1.0)
    )
    assert run.summary.pass_rate == 1.0
    artifact = await control.get_artifact("gated", "0.1.0", principal)
    assert artifact.status.phase is LifecyclePhase.EVALUATED

    _, bad_runner, bad_spec = await _runner(principal, {"q1": "ok", "safe": "leaked"})
    await bad_runner.control.build("gated", "0.1.0", principal)
    bad_pipeline = ReleasePipeline(bad_runner.control, bad_runner)
    bad_run, bad_decision = await bad_pipeline.evaluate_version(
        bad_spec, principal, thresholds=GateThresholds(min_quality=0.5)
    )
    assert bad_decision.passed is False
    await bad_runner.control.approve("gated", "0.1.0", principal)
    with pytest.raises(GateFailedError):
        await bad_pipeline.release_if_passing(
            "gated", "0.1.0", principal, bad_decision, approve=False
        )
    still = await bad_runner.control.get_artifact("gated", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []

    released, dep = await pipeline.release_if_passing("gated", "0.1.0", principal, decision)
    assert released.status.phase is LifecyclePhase.RELEASED
    assert dep.channel is ReleaseChannel.CANARY
    del bad_run
