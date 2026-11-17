from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    AssetKind,
    AssetMetadata,
    AssetRef,
    ControlPlane,
    DependencyError,
    GateDecision,
    GateFailedError,
    LifecycleError,
    LifecyclePhase,
    Principal,
    ReleaseChannel,
)


async def _ready_draft(cp: ControlPlane, principal: Principal, *, version: str = "0.1.0") -> AgentArtifact:
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


async def test_happy_path_and_event_audit(principal: Principal) -> None:
    cp = ControlPlane()
    await _ready_draft(cp, principal)
    built = await cp.build("bot", "0.1.0", principal)
    assert built.status.phase is LifecyclePhase.BUILT
    evaluated = await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp_1")
    assert evaluated.status.phase is LifecyclePhase.EVALUATED
    approved = await cp.approve("bot", "0.1.0", principal, reason="looks good")
    assert approved.status.phase is LifecyclePhase.APPROVED
    released, dep = await cp.release(
        "bot",
        "0.1.0",
        principal,
        GateDecision(passed=True, reason="fixture"),
        channel=ReleaseChannel.CANARY,
    )
    assert released.status.phase is LifecyclePhase.RELEASED
    assert released.status.releaseChannel is ReleaseChannel.CANARY
    assert dep.weight == 0.05
    deprecated = await cp.deprecate("bot", "0.1.0", principal)
    assert deprecated.status.phase is LifecyclePhase.DEPRECATED
    retired = await cp.retire("bot", "0.1.0", principal)
    assert retired.status.phase is LifecyclePhase.RETIRED
    events = cp.events_for(retired.resource_key(), tenant=principal.tenant)
    types = [e.type.value for e in events]
    assert types.count("lifecycle.transition") >= 6
    assert "approval.recorded" in types
    assert "release.canary" in types


async def test_illegal_skips_and_missing_dependencies(principal: Principal) -> None:
    cp = ControlPlane()
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(
                prompts=[AssetRef(kind=AssetKind.PROMPT, name="missing", version="1.0.0")],
            ),
            version="0.1.0",
        ),
        principal,
    )
    with pytest.raises(DependencyError):
        await cp.build("bot", "0.1.0", principal)
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="missing", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "x"},
        principal=principal,
    )
    await cp.build("bot", "0.1.0", principal)
    with pytest.raises(LifecycleError):
        await cp.approve("bot", "0.1.0", principal)
    with pytest.raises(LifecycleError):
        await cp.release("bot", "0.1.0", principal, GateDecision(passed=True, reason="too early"))


async def test_release_without_passing_gate_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    await _ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp")
    await cp.approve("bot", "0.1.0", principal)
    with pytest.raises(GateFailedError) as exc:
        await cp.release("bot", "0.1.0", principal, GateDecision(passed=False, reason="quality"))
    assert exc.value.decision is not None
    assert exc.value.decision.passed is False
    artifact = await cp.get_artifact("bot", "0.1.0", principal)
    assert artifact.status.phase is LifecyclePhase.APPROVED
    blocked = [e for e in cp.events_for(artifact.resource_key(), tenant=principal.tenant) if e.type.value == "release.blocked"]
    assert blocked
