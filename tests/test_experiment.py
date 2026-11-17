from __future__ import annotations

from agentsphere.controlplane import AgentArtifact, AgentSpec, AssetMetadata, ControlPlane, Principal
from agentsphere.evaluation import (
    ArtifactRuntimeFactory,
    DatasetCase,
    DatasetService,
    EvaluatorRegistry,
    ExperimentRunner,
    ExperimentSpec,
    RuleAssertionEvaluator,
    RuntimeCaseExecutor,
    ScriptedExecutor,
    SliceTags,
    compare_experiments,
    replay_session,
)
from agentsphere.evaluation.regression import FailureAttribution
from agentsphere.evaluation.types import SampleMetrics
from agentsphere.kernel.events import Event, EventType
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text
from agentsphere.runtime.replay import replay


async def _agent(control: ControlPlane, principal: Principal, *, version: str = "0.1.0") -> AgentArtifact:
    artifact = AgentArtifact(
        metadata=AssetMetadata(name="eval-bot", owner=principal.owner, tenant=principal.tenant),
        spec=AgentSpec(type="managed"),
        version=version,
    )
    return await control.put_artifact(artifact, principal)


async def test_experiment_fingerprint_pass_at_k_and_variance(principal: Principal) -> None:
    control = ControlPlane()
    await _agent(control, principal)
    datasets = DatasetService(control)
    await datasets.create(
        name="ds",
        version="1.0.0",
        principal=principal,
        cases=[
            DatasetCase(case_id="stable", input="a", expected="yes", slice=SliceTags(scene="s1", difficulty="easy")),
            DatasetCase(case_id="flaky", input="b", expected="yes", slice=SliceTags(scene="s2", difficulty="hard")),
        ],
    )
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())

    def outputs(case: DatasetCase, sample_index: int) -> str:
        if case.case_id == "stable":
            return "yes"
        return "yes" if sample_index == 1 else "no"

    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=ScriptedExecutor(outputs),
    )
    spec = ExperimentSpec(
        agent_name="eval-bot",
        agent_version="0.1.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
        samples=2,
    )
    run = await runner.run(spec, principal)
    again = ExperimentSpec.model_validate(spec.model_dump())
    assert again.fingerprint() == spec.fingerprint()
    assert run.summary.fingerprint == spec.fingerprint()
    assert run.summary.n_cases == 2
    assert run.summary.n_samples == 2
    assert run.summary.pass_at_k == 1.0
    assert 0.0 < run.summary.pass_rate < 1.0
    assert run.summary.variance > 0
    assert "scene:s1" in run.summary.by_slice
    assert run.summary.by_slice["scene:s1"] == 1.0


async def test_experiment_executes_managed_runtime_and_reuses_events(
    principal: Principal, registry
) -> None:
    control = ControlPlane()
    await _agent(control, principal)
    datasets = DatasetService(control)
    await datasets.create(
        name="ds",
        version="1.0.0",
        principal=principal,
        cases=[DatasetCase(case_id="hello", input="hi", expected="hello world")],
    )
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())
    factory = ArtifactRuntimeFactory(
        adapter=ModelAdapter(MockModelProvider(default_content="hello world")),
        registry=registry,
    )
    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=RuntimeCaseExecutor(factory),
    )
    spec = ExperimentSpec(
        agent_name="eval-bot",
        agent_version="0.1.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
    )
    run = await runner.run(spec, principal)
    assert run.summary.pass_rate == 1.0
    sample = run.cases[0].samples[0]
    types = {e.type for e in sample.events}
    assert EventType.TURN_STARTED in types
    assert EventType.TURN_COMPLETED in types
    state = replay_session(sample.events)
    assert state.session_id
    rebuilt = replay(sample.events)
    assert rebuilt.messages
    assert sample.session_id


async def test_regression_overall_slice_and_coarse_attribution(principal: Principal) -> None:
    control = ControlPlane()
    await _agent(control, principal, version="1.0.0")
    await _agent(control, principal, version="1.1.0")
    datasets = DatasetService(control)
    cases = [
        DatasetCase(case_id="ok", input="a", expected="yes", slice=SliceTags(scene="billing", difficulty="easy")),
        DatasetCase(
            case_id="regressed",
            input="b",
            expected="yes",
            slice=SliceTags(scene="billing", difficulty="hard", risk="low"),
        ),
        DatasetCase(case_id="still-bad", input="c", expected="yes", slice=SliceTags(scene="search")),
    ]
    await datasets.create(name="ds", version="1.0.0", principal=principal, cases=cases)
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())

    def baseline_out(case: DatasetCase, sample_index: int) -> str:
        del sample_index
        return "yes" if case.case_id != "still-bad" else "no"

    def candidate_out(case: DatasetCase, sample_index: int) -> str:
        del sample_index
        return "yes" if case.case_id == "ok" else "no"

    runner = ExperimentRunner(control, evaluators, datasets=datasets)
    spec_base = ExperimentSpec(
        agent_name="eval-bot",
        agent_version="1.0.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
    )
    spec_cand = spec_base.model_copy(update={"agent_version": "1.1.0"})
    fail_events = [
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id="s",
            payload={"name": "search", "error": "boom"},
        )
    ]
    base = await runner.run(spec_base, principal, executor=ScriptedExecutor(baseline_out))
    cand = await runner.run(
        spec_cand,
        principal,
        executor=ScriptedExecutor(candidate_out, events={"regressed": fail_events}),
    )
    report = compare_experiments(base, cand)
    assert report.delta < 0
    assert any(f.case_id == "regressed" for f in report.failures)
    billed = next(s for s in report.by_slice if s.key == "scene:billing")
    assert billed.candidate < billed.baseline
    attributed = next(f for f in report.failures if f.case_id == "regressed")
    assert attributed.attribution is FailureAttribution.TOOL


async def test_cost_and_latency_metrics_flow_into_summary(principal: Principal) -> None:
    control = ControlPlane()
    await _agent(control, principal)
    datasets = DatasetService(control)
    await datasets.create(
        name="ds",
        version="1.0.0",
        principal=principal,
        cases=[DatasetCase(case_id="m", input="x", expected="y")],
    )
    evaluators = EvaluatorRegistry()
    evaluators.register(RuleAssertionEvaluator())
    runner = ExperimentRunner(
        control,
        evaluators,
        datasets=datasets,
        executor=ScriptedExecutor(
            {"m": "y"},
            metrics={"m": SampleMetrics(cost=3.5, latency_ms=12.0, tokens=10)},
        ),
    )
    spec = ExperimentSpec(
        agent_name="eval-bot",
        agent_version="0.1.0",
        dataset_name="ds",
        dataset_version="1.0.0",
        evaluator_name="rule-assertion",
        evaluator_version="0.1.0",
    )
    run = await runner.run(spec, principal)
    assert run.summary.mean_cost == 3.5
    assert run.summary.mean_latency_ms == 12.0
