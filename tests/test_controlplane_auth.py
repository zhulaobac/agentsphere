from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    ArtifactNotFoundError,
    AssetKind,
    AssetMetadata,
    AssetRef,
    AuthorizationError,
    ControlPlane,
    LifecycleError,
    LifecyclePhase,
    Principal,
)
from tests.experiment_stubs import ready_draft, stub_experiment_run


async def test_tenant_cannot_read_or_mutate_foreign_artifacts(
    principal: Principal, other_tenant: Principal
) -> None:
    cp = ControlPlane()
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="secret", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(),
            version="0.1.0",
        ),
        principal,
    )
    with pytest.raises(ArtifactNotFoundError):
        await cp.get_artifact("secret", "0.1.0", other_tenant)
    listed = await cp.list_artifacts(other_tenant)
    assert listed == []
    with pytest.raises(AuthorizationError):
        await cp.put_artifact(
            AgentArtifact(
                metadata=AssetMetadata(name="secret", owner=principal.owner, tenant=principal.tenant),
                spec=AgentSpec(),
                version="0.2.0",
            ),
            other_tenant,
        )


async def test_same_tenant_non_owner_cannot_release(principal: Principal) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "x"},
        principal=principal,
    )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(),
            version="0.1.0",
        ),
        principal,
    )
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run("e", "bot", "0.1.0"))
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="e")
    await cp.approve("bot", "0.1.0", principal)
    coworker = Principal(name="carol", tenant=principal.tenant, owner="carol", team="platform")
    with pytest.raises(AuthorizationError):
        await cp.release("bot", "0.1.0", coworker)
    listed = await cp.list_artifacts(coworker)
    assert len(listed) == 1


async def test_same_tenant_non_owner_cannot_plant_asset_or_artifact(principal: Principal) -> None:
    cp = ControlPlane()
    carol = Principal(name="carol", tenant=principal.tenant, owner="carol", team="platform")
    with pytest.raises(AuthorizationError):
        await cp.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(name="planted-prompt", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"text": "nope"},
            principal=carol,
        )
    with pytest.raises(AuthorizationError):
        await cp.put_artifact(
            AgentArtifact(
                metadata=AssetMetadata(name="planted", owner=principal.owner, tenant=principal.tenant),
                spec=AgentSpec(),
                version="0.1.0",
            ),
            carol,
        )
    assert await cp.list_artifacts(principal) == []
    assert await cp.list_assets(principal, kind=AssetKind.PROMPT) == []


async def test_foreign_principal_cannot_read_events_or_lineage(
    principal: Principal, other_tenant: Principal
) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "x"},
        principal=principal,
    )
    stored = await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="secret", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="0.1.0",
        ),
        principal,
    )
    agent_ref = AssetRef(kind=AssetKind.AGENT, name="secret", version="0.1.0", tenant=principal.tenant)
    prompt_ref = AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0", tenant=principal.tenant)
    alice_uses = cp.lineage(agent_ref, principal)
    alice_used_by = cp.lineage(prompt_ref, principal)
    assert alice_uses["uses"]
    assert alice_used_by["used_by"]

    with pytest.raises(AuthorizationError):
        cp.events_for(stored.resource_key(), other_tenant, tenant=principal.tenant)
    with pytest.raises(AuthorizationError):
        cp.lineage(agent_ref, other_tenant, tenant=principal.tenant)
    assert cp.events_for(stored.resource_key(), other_tenant) == []

    with pytest.raises(AuthorizationError):
        cp.lineage(agent_ref, other_tenant)
    with pytest.raises(AuthorizationError):
        cp.lineage(prompt_ref, other_tenant)
    unscoped = cp.lineage(
        AssetRef(kind=AssetKind.AGENT, name="secret", version="0.1.0"),
        other_tenant,
    )
    assert unscoped == {"uses": [], "used_by": []}

    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=other_tenant.owner, tenant=other_tenant.tenant),
        version="1.0.0",
        spec={"text": "bob"},
        principal=other_tenant,
    )
    bob_prompt = cp.lineage(
        AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0"),
        other_tenant,
    )
    assert bob_prompt["used_by"] == []
    bob_prompt_scoped = cp.lineage(
        AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0", tenant=other_tenant.tenant),
        other_tenant,
    )
    assert bob_prompt_scoped["used_by"] == []
    alice_used_by_after = cp.lineage(prompt_ref, principal)
    assert any(r.name == "secret" and r.tenant == principal.tenant for r in alice_used_by_after["used_by"])


async def test_foreign_principal_cannot_use_experiment_run(
    principal: Principal, other_tenant: Principal
) -> None:
    cp = ControlPlane()
    await ready_draft(cp, principal)
    await cp.build("bot", "0.1.0", principal)
    cp.register_experiment(stub_experiment_run("exp-shared", "bot", "0.1.0", tenant=principal.tenant))

    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=other_tenant.owner, tenant=other_tenant.tenant),
        version="1.0.0",
        spec={"text": "bob"},
        principal=other_tenant,
    )
    await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=other_tenant.owner, tenant=other_tenant.tenant),
            spec=AgentSpec(prompts=[AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0")]),
            version="0.1.0",
        ),
        other_tenant,
    )
    await cp.build("bot", "0.1.0", other_tenant)
    with pytest.raises(AuthorizationError):
        await cp.record_evaluation("bot", "0.1.0", other_tenant, experiment_run_id="exp-shared")
    with pytest.raises(AuthorizationError):
        await cp.record_gate("bot", "0.1.0", other_tenant, experiment_run_id="exp-shared")
    with pytest.raises((AuthorizationError, LifecycleError)):
        await cp.release("bot", "0.1.0", other_tenant)
    still = await cp.get_artifact("bot", "0.1.0", other_tenant)
    assert still.status.phase is LifecyclePhase.BUILT
    assert still.status.deployedVersions == []


async def test_foreign_asset_ref_cannot_be_built_or_bound(
    principal: Principal, other_tenant: Principal
) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=other_tenant.owner, tenant=other_tenant.tenant),
        version="1.0.0",
        spec={"text": "BOB-PRIVATE-SYSTEM-PROMPT"},
        principal=other_tenant,
    )
    stored = await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="bot", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(
                prompts=[
                    AssetRef(
                        kind=AssetKind.PROMPT,
                        name="sys",
                        version="1.0.0",
                        tenant=other_tenant.tenant,
                    )
                ]
            ),
            version="0.1.0",
        ),
        principal,
    )
    graph = cp.lineage(
        AssetRef(kind=AssetKind.AGENT, name="bot", version="0.1.0"),
        principal,
    )
    assert all(r.tenant != other_tenant.tenant for r in graph["uses"])
    with pytest.raises(AuthorizationError):
        await cp.build("bot", "0.1.0", principal)
    with pytest.raises(AuthorizationError):
        await cp._system_prompt_for(stored, principal)
    still = await cp.get_artifact("bot", "0.1.0", principal)
    assert still.status.phase is LifecyclePhase.DRAFT
