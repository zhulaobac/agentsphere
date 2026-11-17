from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    AssetKind,
    AssetMetadata,
    AssetRef,
    ControlPlane,
    GateCheck,
    GateDecision,
    GateFailedError,
    GateThresholds,
    LifecycleError,
    LifecyclePhase,
    Principal,
)
from agentsphere.evaluation import (
    DatasetCase,
    DatasetService,
    EvaluatorRegistry,
    ExperimentRunner,
    ExperimentSpec,
    ReleasePipeline,
    RuleAssertionEvaluator,
    ScriptedExecutor,
    SliceTags,
)
from tests.experiment_stubs import ready_draft, stub_experiment_run


async def _approved(cp: ControlPlane, principal: Principal, *, run_id: str = "exp_1") -> None:
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run(run_id, "bot", "0.1.0"))
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id=run_id)
    await cp.approve("bot", "0.1.0", principal)


async def test_forged_passed_true_does_not_release(principal: Principal) -> None:
    cp = ControlPlane()
    await _approved(cp, principal)
    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal, GateDecision(passed=True, reason="forged"))
    artifact = await cp.get_artifact("bot", "0.1.0", principal)
    assert artifact.status.phase is LifecyclePhase.APPROVED
    assert artifact.status.deployedVersions == []


async def test_inconsistent_checks_are_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    await _approved(cp, principal)
    forged = GateDecision(
        passed=True,
        checks=[GateCheck(name="quality", passed=False, reason="0.0 vs 0.8")],
        reason="forged",
    )
    with pytest.raises(GateFailedError) as exc:
        await cp.release("bot", "0.1.0", principal, forged)
    assert "inconsistent" in str(exc.value)
    artifact = await cp.get_artifact("bot", "0.1.0", principal)
    assert artifact.status.phase is LifecyclePhase.APPROVED


async def test_stale_gate_from_another_version_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal, version="1.0.0")
    await cp.build("bot", "1.0.0", principal)
    cp.register_experiment(stub_experiment_run("exp-1.0.0", "bot", "1.0.0"))
    await cp.record_evaluation("bot", "1.0.0", principal, experiment_run_id="exp-1.0.0")
    await cp.approve("bot", "1.0.0", principal)
    await cp.record_gate("bot", "1.0.0", principal, experiment_run_id="exp-1.0.0")
    v1 = await cp.get_artifact("bot", "1.0.0", principal)
    stale_id = v1.status.lastGateResultId
    assert stale_id

    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="2.0.0",
        ),
        principal,
    )
    await cp.build("bot", "2.0.0", principal)
    cp.register_experiment(stub_experiment_run("exp-2.0.0-FAILED", "bot", "2.0.0", pass_rate=0.0))
    await cp.record_evaluation("bot", "2.0.0", principal, experiment_run_id="exp-2.0.0-FAILED")
    await cp.approve("bot", "2.0.0", principal)
    with pytest.raises(GateFailedError):
        await cp.release("bot", "2.0.0", principal, gate_id=stale_id)
    v2 = await cp.get_artifact("bot", "2.0.0", principal)
    assert v2.status.phase is LifecyclePhase.APPROVED
    assert v2.status.lastExperimentRunId == "exp-2.0.0-FAILED"


async def test_flipping_passed_on_failing_safety_gate_does_not_release(principal: Principal) -> None:
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
    await datasets.create(
        name="ds",
        version="1.0.0",
        principal=principal,
        cases=[
            DatasetCase(case_id="q1", input="a", expected="ok"),
            DatasetCase(
                case_id="safe",
                input="b",
                expected="denied",
                slice=SliceTags(risk="safety"),
                labels={"safety": "true"},
            ),
        ],
    )
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())
    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=ScriptedExecutor({"q1": "ok", "safe": "leaked"}),
    )
    await control.build("gated", "0.1.0", principal)
    pipeline = ReleasePipeline(control, runner)
    spec = ExperimentSpec(
        agent_name="gated",
        agent_version="0.1.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
    )
    _run, decision = await pipeline.evaluate_version(
        spec, principal, thresholds=GateThresholds(min_quality=0.0)
    )
    assert decision.passed is False
    assert any(c.name == "safety" and not c.passed for c in decision.checks)
    await control.approve("gated", "0.1.0", principal)
    flipped = decision.model_copy(update={"passed": True})
    with pytest.raises(GateFailedError):
        await control.release("gated", "0.1.0", principal, flipped)
    still = await control.get_artifact("gated", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []


async def test_record_evaluation_requires_stored_run(principal: Principal) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    with pytest.raises(LifecycleError, match="unknown experiment run"):
        await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="i-never-ran")
