"""Minimal stored ExperimentRun objects for Control Plane lifecycle tests."""

from __future__ import annotations

from agentsphere.controlplane import AgentArtifact, AgentSpec, AssetKind, AssetMetadata, AssetRef, ControlPlane, Principal
from agentsphere.evaluation.experiment import (
    CaseRun,
    ExperimentRun,
    ExperimentSpec,
    ExperimentSummary,
    SampleRecord,
)
from agentsphere.evaluation.types import DatasetCase, EvalResult, SliceTags


def stub_experiment_run(
    run_id: str,
    agent_name: str,
    agent_version: str,
    *,
    pass_rate: float = 1.0,
    safety_failed: bool = False,
    tenant: str = "acme",
) -> ExperimentRun:
    quality_passed = pass_rate >= 1.0
    cases = [
        CaseRun(
            case=DatasetCase(case_id="q1", input="a", expected="ok"),
            samples=[
                SampleRecord(
                    sample_index=0,
                    output="ok" if quality_passed else "nope",
                    passed=quality_passed,
                    result=EvalResult(passed=quality_passed, score=1.0 if quality_passed else 0.0),
                )
            ],
        )
    ]
    if safety_failed:
        cases.append(
            CaseRun(
                case=DatasetCase(
                    case_id="safe",
                    input="b",
                    expected="denied",
                    slice=SliceTags(scene="sec", risk="safety"),
                    labels={"safety": "true"},
                ),
                samples=[
                    SampleRecord(
                        sample_index=0,
                        output="leaked",
                        passed=False,
                        result=EvalResult(passed=False, score=0.0),
                    )
                ],
            )
        )
    n = len(cases)
    summary_rate = sum(c.pass_fraction for c in cases) / n
    return ExperimentRun(
        run_id=run_id,
        spec=ExperimentSpec(
            agent_name=agent_name,
            agent_version=agent_version,
            dataset_name="stub-ds",
            dataset_version="1.0.0",
            evaluator_name="rule-assertion",
            evaluator_version="0.1.0",
            tenant=tenant,
        ),
        summary=ExperimentSummary(
            n_cases=n,
            n_samples=1,
            pass_rate=summary_rate,
            pass_at_k=sum(1 for c in cases if c.pass_at_k) / n,
            variance=0.0,
            fingerprint="stub",
        ),
        cases=cases,
    )


async def ready_draft(cp: ControlPlane, principal: Principal, *, version: str = "0.1.0") -> AgentArtifact:
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "sys"},
        principal=principal,
    )
    return await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(
                prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")],
            ),
            version=version,
        ),
        principal,
    )
