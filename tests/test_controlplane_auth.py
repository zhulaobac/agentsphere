from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    ArtifactNotFoundError,
    AssetKind,
    AssetMetadata,
    AuthorizationError,
    ControlPlane,
    GateDecision,
    Principal,
)


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
    await cp.record_evaluation("bot", "0.1.0", principal, experiment_run_id="e")
    await cp.approve("bot", "0.1.0", principal)
    coworker = Principal(name="carol", tenant=principal.tenant, owner="carol", team="platform")
    with pytest.raises(AuthorizationError):
        await cp.release("bot", "0.1.0", coworker, GateDecision(passed=True, reason="nope"))
    listed = await cp.list_artifacts(coworker)
    assert len(listed) == 1
