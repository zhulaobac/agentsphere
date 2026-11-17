from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    AssetKind,
    AssetMetadata,
    AssetNotFoundError,
    AssetRef,
    AuthorizationError,
    CANARY_WEIGHT,
    ControlPlane,
    DeploymentStatus,
    LifecycleError,
    LifecyclePhase,
    ManagedRuntimeProvider,
    Principal,
    ReleaseChannel,
    SLOSignal,
)
from agentsphere.model_adapter import MockModelProvider, ModelAdapter, assistant_text
from agentsphere.runtime import InMemoryEventStore, build_runtime
from tests.experiment_stubs import stub_experiment_run


async def _ship(cp: ControlPlane, principal: Principal, *, version: str) -> None:
    try:
        await cp.get_asset(AssetKind.PROMPT, "sys", "1.0.0", principal)
    except AssetNotFoundError:
        await cp.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"text": "You are a bot."},
            principal=principal,
        )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version=version,
        ),
        principal,
    )
    await cp.build("bot", version, principal)
    run_id = f"exp-{version}"
    cp.register_experiment(stub_experiment_run(run_id, "bot", version))
    await cp.record_evaluation("bot", version, principal, experiment_run_id=run_id)
    await cp.approve("bot", version, principal)
    await cp.record_gate("bot", version, principal, experiment_run_id=run_id)
    await cp.release("bot", version, principal)


async def test_release_to_full_without_canary_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    try:
        await cp.get_asset(AssetKind.PROMPT, "sys", "1.0.0", principal)
    except AssetNotFoundError:
        await cp.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"text": "You are a bot."},
            principal=principal,
        )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="0.1.0",
        ),
        principal,
    )
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run("exp-full", "bot", "0.1.0"))
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-full")
    await cp.approve("bot", "0.1.0", principal)
    await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-full")
    with pytest.raises(LifecycleError, match="promote an active Canary"):
        await cp.release("bot", "0.1.0", principal, channel=ReleaseChannel.FULL)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []
    canary, dep = await cp.release("bot", "0.1.0", principal)
    assert canary.status.phase is LifecyclePhase.RELEASED
    assert dep.channel is ReleaseChannel.CANARY
    promoted, full = await cp.promote("bot", "0.1.0", principal)
    assert promoted.status.releaseChannel is ReleaseChannel.FULL
    assert full.channel is ReleaseChannel.FULL
    assert full.weight == 1.0


async def test_canary_then_full_then_rollback(principal: Principal) -> None:
    cp = ControlPlane()
    await _ship(cp, principal, version="1.0.0")
    await cp.promote("bot", "1.0.0", principal)
    v1 = await cp.get_artifact("bot", "1.0.0", principal)
    assert v1.status.releaseChannel is ReleaseChannel.FULL
    await _ship(cp, principal, version="1.1.0")
    canary = await cp.get_artifact("bot", "1.1.0", principal)
    assert canary.status.releaseChannel is ReleaseChannel.CANARY
    restored, dep = await cp.rollback( "bot", principal, to_version="1.0.0", reason="canary bad")
    assert restored.version == "1.0.0"
    assert dep.channel is ReleaseChannel.FULL
    assert dep.rolled_back_from == "1.1.0"
    failing = await cp.get_artifact("bot", "1.1.0", principal)
    assert failing.status.phase is LifecyclePhase.DEPRECATED
    deployments = cp.store.list_deployments(tenant=principal.tenant, name="bot")
    rolled = [d for d in deployments if d.status is DeploymentStatus.ROLLED_BACK]
    assert rolled


async def test_auto_rollback_from_recorded_slo(principal: Principal) -> None:
    cp = ControlPlane()
    await _ship(cp, principal, version="1.0.0")
    await cp.promote("bot", "1.0.0", principal)
    await _ship(cp, principal, version="2.0.0")
    deployments = [
        d
        for d in cp.store.list_deployments(tenant=principal.tenant, name="bot")
        if d.agent_version == "2.0.0" and d.status is DeploymentStatus.ACTIVE
    ]
    signal = SLOSignal(
        deployment_id=deployments[-1].deployment_id,
        metric="success_rate",
        value=0.4,
        threshold=0.9,
        direction="below_is_bad",
    )
    await cp.record_slo(signal, principal)
    current = await cp.get_artifact("bot", "2.0.0", principal)
    assert current.status.phase is LifecyclePhase.DEPRECATED
    restored = await cp.get_artifact("bot", "1.0.0", principal)
    assert restored.status.phase is LifecyclePhase.RELEASED
    events = [e.type.value for e in cp.store.list_events(tenant=principal.tenant)]
    assert "release.auto_rollback" in events
    assert "slo.recorded" in events


async def test_managed_runtime_provider_binds_release(principal: Principal, registry) -> None:
    runtime = build_runtime(
        adapter=ModelAdapter(MockModelProvider(script=[assistant_text("hello")])),
        registry=registry,
        store=InMemoryEventStore(),
    )
    provider = ManagedRuntimeProvider(runtime)
    cp = ControlPlane(runtime=provider)
    await _ship(cp, principal, version="0.1.0")
    deployments = cp.store.list_deployments(tenant=principal.tenant, name="bot")
    deployment_id = deployments[-1].deployment_id
    binding = provider.get_binding(deployment_id)
    assert binding is not None
    assert binding.agent_version == "0.1.0"
    session = await provider.create_session(deployment_id=deployment_id, metadata={"source": "test"})
    assert session.metadata["deployment_id"] == deployment_id
    assert session.metadata["agent_name"] == "bot"
    assert session.system_prompt == "You are a bot."
    events = [e async for e in provider.run_turn(session.session_id, "hi")]
    assert events
    stored = await provider.list_events(session.session_id)
    assert stored[0].session_id == session.session_id


async def test_foreign_principal_cannot_insert_slo(
    principal: Principal, other_tenant: Principal
) -> None:
    cp = ControlPlane()
    await _ship(cp, principal, version="1.0.0")
    deployments = cp.store.list_deployments(tenant=principal.tenant, name="bot")
    deployment_id = deployments[-1].deployment_id
    before = cp.store.list_slo(deployment_id=deployment_id)
    before_events = len(cp.store.list_events(tenant=principal.tenant))
    signal = SLOSignal(
        deployment_id=deployment_id,
        metric="success_rate",
        value=0.1,
        threshold=0.9,
        direction="below_is_bad",
    )
    with pytest.raises(AuthorizationError):
        await cp.record_slo(signal, other_tenant, auto_rollback=True)
    with pytest.raises(AuthorizationError):
        await cp.record_slo(signal, other_tenant, auto_rollback=False)
    assert cp.store.list_slo(deployment_id=deployment_id) == before
    assert len(cp.store.list_events(tenant=principal.tenant)) == before_events
    still = await cp.get_artifact("bot", "1.0.0", principal)
    assert still.status.phase is LifecyclePhase.RELEASED


async def test_auto_rollback_semver_fallback_restores_1_10_0(principal: Principal) -> None:
    cp = ControlPlane()
    for version in ("1.2.0", "1.10.0", "1.11.0"):
        await _ship(cp, principal, version=version)
    previous = cp._previous_released_version(tenant=principal.tenant, name="bot", current="1.11.0")
    assert previous == "1.10.0"
    deployments = [
        d
        for d in cp.store.list_deployments(tenant=principal.tenant, name="bot")
        if d.agent_version == "1.11.0" and d.status is DeploymentStatus.ACTIVE
    ]
    await cp.record_slo(
        SLOSignal(
            deployment_id=deployments[-1].deployment_id,
            metric="success_rate",
            value=0.2,
            threshold=0.9,
            direction="below_is_bad",
        ),
        principal,
    )
    restored = await cp.get_artifact("bot", "1.10.0", principal)
    assert restored.status.phase is LifecyclePhase.RELEASED
    assert restored.status.releaseChannel is ReleaseChannel.CANARY
    current = await cp.get_artifact("bot", "1.11.0", principal)
    assert current.status.phase is LifecyclePhase.DEPRECATED


async def test_rollback_to_self_canary_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    await _ship(cp, principal, version="0.1.0")
    before = await cp.get_artifact("bot", "0.1.0", principal)
    assert before.status.releaseChannel is ReleaseChannel.CANARY
    with pytest.raises(LifecycleError, match="promote"):
        await cp.rollback("bot", principal, to_version="0.1.0", reason="emergency first ship")
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.RELEASED
    assert still.status.releaseChannel is ReleaseChannel.CANARY
    active = [
        d
        for d in cp.store.list_deployments(tenant=principal.tenant, name="bot")
        if d.status is DeploymentStatus.ACTIVE
    ]
    assert active
    assert all(d.channel is ReleaseChannel.CANARY for d in active)
    assert all(d.weight < 1.0 for d in active)


async def test_rollback_never_promoted_previous_canary_restores_canary(principal: Principal) -> None:
    cp = ControlPlane()
    await _ship(cp, principal, version="1.0.0")
    await _ship(cp, principal, version="1.1.0")
    restored, dep = await cp.rollback("bot", principal, to_version="1.0.0")
    assert restored.version == "1.0.0"
    assert restored.status.releaseChannel is ReleaseChannel.CANARY
    assert dep.channel is ReleaseChannel.CANARY
    assert dep.weight == CANARY_WEIGHT
    failing = await cp.get_artifact("bot", "1.1.0", principal)
    assert failing.status.phase is LifecyclePhase.DEPRECATED


async def test_canary_weight_one_is_rejected(principal: Principal) -> None:
    cp = ControlPlane()
    try:
        await cp.get_asset(AssetKind.PROMPT, "sys", "1.0.0", principal)
    except AssetNotFoundError:
        await cp.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"text": "You are a bot."},
            principal=principal,
        )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="0.1.0",
        ),
        principal,
    )
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run("exp-weight", "bot", "0.1.0"))
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-weight")
    await cp.approve("bot", "0.1.0", principal)
    await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-weight")
    with pytest.raises(LifecycleError, match="weight"):
        await cp.release("bot", "0.1.0", principal, channel=ReleaseChannel.CANARY, weight=1.0)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []


async def test_release_channel_string_full_raises_lifecycle_error(principal: Principal) -> None:
    cp = ControlPlane()
    try:
        await cp.get_asset(AssetKind.PROMPT, "sys", "1.0.0", principal)
    except AssetNotFoundError:
        await cp.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"text": "You are a bot."},
            principal=principal,
        )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="0.1.0",
        ),
        principal,
    )
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run("exp-str-full", "bot", "0.1.0"))
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="exp-str-full")
    await cp.approve("bot", "0.1.0", principal)
    await cp.record_gate("bot", "0.1.0", principal, experiment_run_id="exp-str-full")
    with pytest.raises(LifecycleError, match="promote an active Canary"):
        await cp.release("bot", "0.1.0", principal, channel="Full")  # type: ignore[arg-type]
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []
