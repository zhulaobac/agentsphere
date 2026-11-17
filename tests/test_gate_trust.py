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
from agentsphere.evaluation.experiment import ExperimentRun
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


async def test_overwriting_failing_experiment_cannot_release(principal: Principal) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(
        stub_experiment_run("exp-fail", "bot", "0.1.0", pass_rate=0.0, safety_failed=True)
    )
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-fail")
    await cp.approve("bot", "0.1.0", principal)
    _artifact, decision = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-fail")
    assert decision.passed is False
    with pytest.raises(LifecycleError, match="immutable"):
        cp.register_experiment(stub_experiment_run("exp-fail", "bot", "0.1.0", pass_rate=1.0))
    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal)
    _, again = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-fail")
    assert again.passed is False
    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []


async def test_stale_same_version_gate_id_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    await _approved(cp, principal, run_id="exp-pass")
    first, passing = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-pass")
    passing_id = first.status.lastGateResultId
    assert passing.passed is True
    assert passing_id

    cp.register_experiment(
        stub_experiment_run("exp-later-fail", "bot", "0.1.0", pass_rate=0.0, safety_failed=True)
    )
    with pytest.raises(LifecycleError, match="does not match"):
        await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-later-fail")

    second, _ = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-pass")
    latest_id = second.status.lastGateResultId
    assert latest_id != passing_id

    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal, gate_id=passing_id)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []
    assert still.status.lastGateResultId == latest_id
    assert still.status.lastExperimentRunId == "exp-pass"


class _MutableExperiment:
    def __init__(self, inner: object) -> None:
        self.run_id = inner.run_id  # type: ignore[attr-defined]
        self.spec = inner.spec  # type: ignore[attr-defined]
        self.summary = inner.summary  # type: ignore[attr-defined]
        self.cases = inner.cases  # type: ignore[attr-defined]
        self.created_at = getattr(inner, "created_at", None)
        self.agent_fingerprint = getattr(inner, "agent_fingerprint", None)
        self.dataset_content_hash = getattr(inner, "dataset_content_hash", None)


async def test_mutable_experiment_standin_cannot_pass_gate(principal: Principal) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    failing = stub_experiment_run(
        "exp-mut", "bot", "0.1.0", pass_rate=0.0, safety_failed=True, tenant=principal.tenant
    )
    standin = _MutableExperiment(failing)
    try:
        cp.register_experiment(standin)
    except LifecycleError:
        cp.register_experiment(failing)
    else:
        standin.summary.pass_rate = 1.0
        standin.cases[0].samples[0].passed = True
        standin.cases[0].samples[0].result.passed = True
        standin.cases[0].samples[0].result.score = 1.0
        if len(standin.cases) > 1:
            standin.cases[1].samples[0].passed = True
    stored = cp.get_experiment("exp-mut", principal)
    assert stored is not None
    assert stored.summary.pass_rate < 1.0
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-mut")
    await cp.approve("bot", "0.1.0", principal)
    _, decision = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-mut")
    assert decision.passed is False
    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []


class _EvilRun(ExperimentRun):
    def model_copy(self, *args: object, **kwargs: object) -> ExperimentRun:
        return self


async def test_evil_experiment_subclass_cannot_pass_gate(principal: Principal) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    failing = stub_experiment_run(
        "exp-evil", "bot", "0.1.0", pass_rate=0.0, safety_failed=True, tenant=principal.tenant
    )
    evil = _EvilRun.model_validate(failing.model_dump())
    registered = cp.register_experiment(evil)
    assert registered is not evil
    evil.summary.pass_rate = 1.0
    for case in evil.cases:
        for sample in case.samples:
            sample.passed = True
            sample.result.passed = True
            sample.result.score = 1.0
    stored = cp.get_experiment("exp-evil", principal)
    assert stored is not None
    assert stored.summary.pass_rate < 1.0
    assert stored is not evil
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-evil")
    await cp.approve("bot", "0.1.0", principal)
    _, decision = await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-evil")
    assert decision.passed is False
    with pytest.raises(GateFailedError):
        await cp.release("bot", "0.1.0", principal)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []
