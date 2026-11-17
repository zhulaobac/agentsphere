from __future__ import annotations

from pathlib import Path

import pytest

from agentsphere.controlplane import (
    AgentArtifact,
    AgentSpec,
    AssetKind,
    AssetMetadata,
    AssetRef,
    ControlPlane,
    DuplicateAssetError,
    ImmutableAssetError,
    InMemoryControlStore,
    LifecyclePhase,
)
from agentsphere.controlplane.types import Principal


def _artifact(principal: Principal, *, version: str = "0.1.0", prompt: bool = True) -> AgentArtifact:
    prompts = []
    if prompt:
        prompts = [AssetRef(kind=AssetKind.PROMPT, name="sys", version="1.0.0", tenant=principal.tenant)]
    return AgentArtifact(
        metadata=AssetMetadata(
            name="support-bot",
            owner=principal.owner,
            tenant=principal.tenant,
            team=principal.team,
            labels={"tier": "p1"},
        ),
        spec=AgentSpec(type="managed", prompts=prompts),
        version=version,
    )


async def test_put_and_get_versioned_artifact_and_document_roundtrip(principal: Principal) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "You are helpful."},
        principal=principal,
    )
    stored = await cp.put_artifact(_artifact(principal), principal)
    assert stored.apiVersion == "agentsphere/v1"
    assert stored.kind == "Agent"
    assert stored.status.phase is LifecyclePhase.DRAFT
    loaded = await cp.get_artifact("support-bot", "0.1.0", principal)
    assert loaded.spec.prompts[0].name == "sys"
    document = loaded.to_document()
    assert document["apiVersion"] == "agentsphere/v1"
    assert document["kind"] == "Agent"
    assert document["metadata"]["tenant"] == "acme"
    again = AgentArtifact.from_document(document)
    assert again.version == "0.1.0"
    assert again.metadata.labels["tier"] == "p1"


async def test_uniform_asset_kinds_stubs_and_lineage(principal: Principal) -> None:
    cp = ControlPlane()
    kinds = [
        AssetKind.PROMPT,
        AssetKind.TOOL,
        AssetKind.SKILL,
        AssetKind.DATASET,
        AssetKind.EVALUATOR,
        AssetKind.POLICY,
        AssetKind.MCP_SERVER,
        AssetKind.MEMORY_STORE,
    ]
    for kind in kinds:
        record = await cp.put_asset(
            kind=kind,
            metadata=AssetMetadata(name=kind.value.lower(), owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={"stub": kind in {AssetKind.MCP_SERVER, AssetKind.MEMORY_STORE}},
            principal=principal,
        )
        assert record.version == "1.0.0"
        assert record.owner == principal.owner
        assert record.labels == {}
        if kind in {AssetKind.MCP_SERVER, AssetKind.MEMORY_STORE}:
            assert record.stub is True

    artifact = AgentArtifact(
        metadata=AssetMetadata(name="wired", owner=principal.owner, tenant=principal.tenant),
        spec=AgentSpec(
            prompts=[AssetRef(kind=AssetKind.PROMPT, name="prompt", version="1.0.0")],
            tools=[AssetRef(kind=AssetKind.TOOL, name="tool", version="1.0.0")],
            policies=AssetRef(kind=AssetKind.POLICY, name="policy", version="1.0.0"),
            memory=AssetRef(kind=AssetKind.MEMORY_STORE, name="memorystore", version="1.0.0"),
        ),
        version="0.1.0",
    )
    stored = await cp.put_artifact(artifact, principal)
    graph = cp.lineage(
        AssetRef(kind=AssetKind.AGENT, name="wired", version="0.1.0", tenant=principal.tenant),
        tenant=principal.tenant,
    )
    used = {ref.display() for ref in graph["uses"]}
    assert "Prompt/prompt@1.0.0" in used
    assert "Tool/tool@1.0.0" in used
    assert "Policy/policy@1.0.0" in used
    assert "MemoryStore/memorystore@1.0.0" in used
    events = cp.events_for(stored.resource_key(), tenant=principal.tenant)
    assert any(e.type.value == "lineage.edge" for e in events)


async def test_draft_is_mutable_built_requires_new_version(principal: Principal) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.PROMPT,
        metadata=AssetMetadata(name="sys", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"text": "a"},
        principal=principal,
    )
    first = await cp.put_artifact(_artifact(principal), principal)
    first.spec.model.primary = "mock-b"
    updated = await cp.put_artifact(first, principal)
    assert updated.spec.model.primary == "mock-b"
    await cp.build("support-bot", "0.1.0", principal)
    with pytest.raises(ImmutableAssetError):
        await cp.put_artifact(_artifact(principal), principal)
    v2 = await cp.put_artifact(_artifact(principal, version="0.2.0"), principal)
    assert v2.status.phase is LifecyclePhase.DRAFT


async def test_duplicate_asset_rejected_and_json_persist(principal: Principal, tmp_path: Path) -> None:
    cp = ControlPlane()
    await cp.put_asset(
        kind=AssetKind.SKILL,
        metadata=AssetMetadata(name="sop", owner=principal.owner, tenant=principal.tenant),
        version="1.0.0",
        spec={"instructions": "do the thing"},
        principal=principal,
    )
    with pytest.raises(DuplicateAssetError):
        await cp.put_asset(
            kind=AssetKind.SKILL,
            metadata=AssetMetadata(name="sop", owner=principal.owner, tenant=principal.tenant),
            version="1.0.0",
            spec={},
            principal=principal,
        )
    path = tmp_path / "control.json"
    assert isinstance(cp.store, InMemoryControlStore)
    cp.store.save_json(path)
    restored = InMemoryControlStore.load_json(path)
    loaded = restored.get_asset(AssetKind.SKILL, principal.tenant, "sop", "1.0.0")
    assert loaded is not None
    assert loaded.spec["instructions"] == "do the thing"
