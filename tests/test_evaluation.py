from __future__ import annotations

import pytest

from agentsphere.controlplane import ControlPlane, ImmutableAssetError, Principal
from agentsphere.evaluation import (
    AssertionSpec,
    CodeCheckEvaluator,
    DatasetCase,
    DatasetService,
    EvalSample,
    EventTrajectoryEvaluator,
    FakeHumanReview,
    FakeLLMJudge,
    FakeTrajectoryEvaluator,
    RuleAssertionEvaluator,
    SliceTags,
)
from agentsphere.kernel.events import Event, EventType


async def test_dataset_versions_slice_tags_and_source_labels(principal: Principal) -> None:
    datasets = DatasetService(ControlPlane())
    v1 = await datasets.create(
        name="support",
        version="1.0.0",
        principal=principal,
        labels={"source": "manual"},
        cases=[
            DatasetCase(
                case_id="c1",
                input="hi",
                expected="hello",
                slice=SliceTags(scene="greeting", difficulty="easy", risk="low"),
                labels={"source": "manual"},
            )
        ],
    )
    assert v1.cases[0].slice.scene == "greeting"
    v1b = await datasets.update(
        name="support",
        version="1.0.0",
        principal=principal,
        cases=v1.cases
        + [
            DatasetCase(
                case_id="c2",
                input="bye",
                expected="goodbye",
                slice=SliceTags(scene="greeting", difficulty="easy", risk="low"),
                labels={"source": "synthetic"},
            )
        ],
    )
    assert len(v1b.cases) == 2
    v2 = await datasets.create(
        name="support",
        version="1.1.0",
        principal=principal,
        labels={"source": "synthetic"},
        cases=v1b.cases,
    )
    loaded = await datasets.get("support", "1.1.0", principal)
    assert loaded.version == "1.1.0"
    assert loaded.labels["source"] == "synthetic"


async def test_rule_and_code_check_evaluators() -> None:
    rule = RuleAssertionEvaluator()
    sample = EvalSample(
        case=DatasetCase(
            case_id="r1",
            input="q",
            expected="alpha",
            assertions=[
                AssertionSpec(type="equals", value="alpha"),
                AssertionSpec(type="contains", value="alp"),
                AssertionSpec(type="regex", value=r"^al"),
            ],
        ),
        output="alpha",
    )
    ok = await rule.evaluate(sample)
    assert ok.passed is True
    bad = await rule.evaluate(sample.model_copy(update={"output": "beta"}))
    assert bad.passed is False

    code = CodeCheckEvaluator()
    syntax_ok = await code.evaluate(
        EvalSample(
            case=DatasetCase(case_id="py", input="code", metadata={"code_checks": ["syntax"]}),
            output="x = 1\n",
        )
    )
    assert syntax_ok.passed is True
    syntax_bad = await code.evaluate(
        EvalSample(
            case=DatasetCase(case_id="py2", input="code", metadata={"code_checks": ["syntax"]}),
            output="def (",
        )
    )
    assert syntax_bad.passed is False
    expr = await code.evaluate(
        EvalSample(
            case=DatasetCase(
                case_id="py3",
                input="code",
                expected="ok",
                metadata={"code_checks": [], "code_expressions": ["output == expected", "len(output) == 2"]},
            ),
            output="ok",
        )
    )
    assert expr.passed is True

    def has_return(sample: EvalSample) -> bool:
        return "return" in sample.output

    custom = CodeCheckEvaluator(checks={"has_return": has_return})
    custom_ok = await custom.evaluate(
        EvalSample(
            case=DatasetCase(case_id="fn", input="x", metadata={"code_checks": ["has_return"]}),
            output="def f():\n    return 1\n",
        )
    )
    assert custom_ok.passed is True


async def test_fakes_llm_human_trajectory_and_event_compare() -> None:
    llm = FakeLLMJudge(decisions={"c1": True, "c2": False})
    r1 = await llm.evaluate(EvalSample(case=DatasetCase(case_id="c1", input="a"), output="x"))
    r2 = await llm.evaluate(EvalSample(case=DatasetCase(case_id="c2", input="b"), output="y"))
    assert r1.passed is True
    assert r2.passed is False
    assert llm.kind == "llm_judge"

    human = FakeHumanReview(decisions={"h1": True})
    h = await human.evaluate(EvalSample(case=DatasetCase(case_id="h1", input="a"), output="z"))
    assert h.passed is True
    assert human.kind == "human"
    assert human.submitted

    traj = FakeTrajectoryEvaluator()
    t_ok = await traj.evaluate(
        EvalSample(
            case=DatasetCase(
                case_id="t1",
                input="a",
                metadata={"trajectory": {"step_correctness": True, "tool_choice": True, "efficiency": True, "safety_violations": 0}},
            )
        )
    )
    assert t_ok.passed is True
    assert t_ok.dimensions["tool_choice"] is True
    t_bad = await traj.evaluate(
        EvalSample(
            case=DatasetCase(
                case_id="t2",
                input="a",
                metadata={"trajectory": {"step_correctness": True, "tool_choice": False, "efficiency": True, "safety_violations": 1}},
            )
        )
    )
    assert t_bad.passed is False

    events = [
        Event(type=EventType.TURN_STARTED, session_id="s", turn_id="t", step_id="1", payload={"user_input": "x"}),
        Event(type=EventType.TOOL_CALL_STARTED, session_id="s", turn_id="t", step_id="1", payload={"name": "echo"}),
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="s", turn_id="t", step_id="1", payload={"name": "echo", "output": "x"}),
        Event(type=EventType.OUTPUT, session_id="s", turn_id="t", step_id="1", payload={"text": "x"}),
    ]
    ev = EventTrajectoryEvaluator()
    match = await ev.evaluate(
        EvalSample(
            case=DatasetCase(case_id="tr", input="x", metadata={"expected_tools": ["echo"], "max_tools": 1}),
            output="x",
            events=events,
        )
    )
    assert match.passed is True
    mismatch = await ev.evaluate(
        EvalSample(
            case=DatasetCase(case_id="tr2", input="x", metadata={"expected_tools": ["search"]}),
            output="x",
            events=events,
        )
    )
    assert mismatch.passed is False


async def test_pinned_dataset_is_immutable(principal: Principal) -> None:
    from agentsphere.controlplane.assets import DependencyEdge
    from agentsphere.controlplane.types import AssetKind, AssetRef

    control = ControlPlane()
    datasets = DatasetService(control)
    await datasets.create(
        name="locked",
        version="1.0.0",
        principal=principal,
        cases=[DatasetCase(case_id="a", input="i", expected="o")],
    )
    control.store.add_edge(
        DependencyEdge(
            source=AssetRef(kind=AssetKind.AGENT, name="bot", version="0.1.0", tenant=principal.tenant),
            target=AssetRef(kind=AssetKind.DATASET, name="locked", version="1.0.0", tenant=principal.tenant),
            relation="evaluates",
        )
    )
    with pytest.raises(ImmutableAssetError):
        await datasets.update(
            name="locked",
            version="1.0.0",
            principal=principal,
            cases=[DatasetCase(case_id="b", input="i2", expected="o2")],
        )
