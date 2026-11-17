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
    Principal,
)
from tests.experiment_stubs import stub_experiment_run


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
    stored = await cp.put_artifact(
        AgentArtifact(
            metadata=AssetMetadata(name="secret", owner=principal.owner, tenant=principal.tenant),
            spec=AgentSpec(),
            version="0.1.0",
        ),
        principal,
    )
    ref = AssetRef(kind=AssetKind.AGENT, name="secret", version="0.1.0", tenant=principal.tenant)
    with pytest.raises(AuthorizationError):
        cp.events_for(stored.resource_key(), other_tenant, tenant=principal.tenant)
    with pytest.raises(AuthorizationError):
        cp.lineage(ref, other_tenant, tenant=principal.tenant)
    assert cp.events_for(stored.resource_key(), other_tenant) == []
