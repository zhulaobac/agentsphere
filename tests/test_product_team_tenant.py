from __future__ import annotations

import json

import pytest

from agentsphere.controlplane import GateFailedError, LifecycleError, LifecyclePhase, ReleaseChannel
from agentsphere.kernel.events import EventType
from agentsphere.tenants.product_team import (
    FixtureResearchSource,
    LiveResearchNotConfigured,
    ProductTeamPack,
    parse_research_report,
    resolve_research_source,
)
from agentsphere.tenants.product_team.ci import run_product_team_fixture_path
from agentsphere.tenants.product_team.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    FIXTURE_SECRET,
    TENANT_DISPLAY_NAME,
    TENANT_ID,
    TENANT_OWNER,
    TENANT_TEAM,
    TOOL_ASSEMBLE,
    TOOL_COMPARE,
    TOOL_LIST,
    TOOL_LOOKUP,
)
from agentsphere.tenants.product_team.dataset import dataset_cases, failing_outputs
from agentsphere.tenants.product_team.lifecycle import (
    build_and_evaluate,
    release_canary_then_full,
    run_fixture_release_path,
    run_session,
)


def _payload_blob(events) -> str:
    return json.dumps([e.payload for e in events], default=str)


async def test_research_source_defaults_to_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTSPHERE_RESEARCH_SOURCE", raising=False)
    monkeypatch.delenv("AGENTSPHERE_RESEARCH_ENDPOINT", raising=False)
    source = resolve_research_source()
    assert isinstance(source, FixtureResearchSource)
    brief = source.lookup_topic("research notebooks for distributed teams")
    assert brief["problem_context"]
    assert brief["backend"] == "fixture"
    with pytest.raises(LiveResearchNotConfigured):
        resolve_research_source("live")
    monkeypatch.setenv("AGENTSPHERE_RESEARCH_SOURCE", "live")
    with pytest.raises(LiveResearchNotConfigured):
        resolve_research_source()


async def test_product_team_onboard_registers_owner_team_and_assets() -> None:
    onboarded = await ProductTeamPack().onboard()
    assert onboarded.record.tenant_id == TENANT_ID
    assert onboarded.record.display_name == TENANT_DISPLAY_NAME
    assert onboarded.record.owner == TENANT_OWNER
    assert onboarded.record.team == TENANT_TEAM
    assert onboarded.record.production is True
    listed = onboarded.tenants.list(onboarded.principal)
    assert len(listed) == 1
    stored = onboarded.tenants.get(TENANT_ID, onboarded.principal)
    assert stored.display_name == TENANT_DISPLAY_NAME

    artifact = await onboarded.control.get_artifact(AGENT_NAME, AGENT_VERSION, onboarded.principal)
    assert artifact.status.phase is LifecyclePhase.DRAFT
    assert artifact.metadata.tenant == TENANT_ID
    assert artifact.metadata.owner == TENANT_OWNER
    assert artifact.metadata.team == TENANT_TEAM
    assert artifact.spec.tools
    assert artifact.spec.policies is not None
    assert artifact.spec.eval.datasets
    assert artifact.spec.eval.gates

    cases = dataset_cases()
    assert {c.slice.scene for c in cases} >= {"requirements-research", "competitive-analysis", "security"}
    assert {c.slice.difficulty for c in cases} >= {"easy", "medium"}
    assert any(c.is_safety() for c in cases)


async def test_product_team_end_to_end_session_uses_runtime_sandbox_vault_trace() -> None:
    onboarded = await ProductTeamPack().onboard()
    session, events, output = await run_session(
        onboarded, "research notebooks for distributed teams"
    )
    report = parse_research_report(output)
    assert report.has_required_sections()
    assert report.problem_context
    assert report.user_need_findings
    assert len(report.competitor_comparison) >= 2
    assert report.implications_or_recommendations
    assert all("brand" not in row.name.lower() for row in report.competitor_comparison)

    names = [
        e.payload.get("name")
        for e in events
        if e.type is EventType.TOOL_CALL_COMPLETED
    ]
    assert TOOL_LOOKUP in names
    assert TOOL_LIST in names
    assert TOOL_COMPARE in names
    assert TOOL_ASSEMBLE in names
    blob = _payload_blob(events)
    assert FIXTURE_SECRET not in blob
    assert FIXTURE_SECRET not in output
    assert session.metadata.get("tenant") == TENANT_ID

    assert onboarded.tracer is not None
    tree = onboarded.tracer.inspect(session.session_id)
    assert tree.root is not None
    nested = [
        grandchild.kind
        for turn in tree.root.children
        for step in turn.children
        for grandchild in step.children
    ]
    assert "tool.call" in nested
    assert onboarded.audit is not None
    records = await onboarded.audit.list()
    assert any(record.action == "tool.call" for record in records)


async def test_product_team_lifecycle_canary_then_promote_full() -> None:
    onboarded = await ProductTeamPack().onboard()
    run, decision, artifact, deployment = await run_fixture_release_path(onboarded)
    assert run.summary.pass_rate == 1.0
    assert "scene:requirements-research" in run.summary.by_slice
    assert "difficulty:easy" in run.summary.by_slice
    assert "risk:safety" in run.summary.by_slice
    assert decision.passed is True
    assert artifact.status.phase is LifecyclePhase.RELEASED
    assert artifact.status.releaseChannel is ReleaseChannel.FULL
    assert deployment.channel is ReleaseChannel.FULL
    assert deployment.weight == 1.0
    deps = onboarded.control.store.list_deployments(tenant=TENANT_ID, name=AGENT_NAME)
    assert any(d.channel is ReleaseChannel.CANARY for d in deps) or any(
        d.channel is ReleaseChannel.FULL for d in deps
    )


async def test_product_team_release_to_full_without_canary_is_rejected() -> None:
    onboarded = await ProductTeamPack().onboard()
    run, decision = await build_and_evaluate(onboarded)
    assert run.summary.pass_rate == 1.0
    assert decision.passed is True
    await onboarded.control.approve(AGENT_NAME, AGENT_VERSION, onboarded.principal, reason="ready")
    with pytest.raises(LifecycleError, match="promote"):
        await onboarded.control.release(
            AGENT_NAME,
            AGENT_VERSION,
            onboarded.principal,
            channel=ReleaseChannel.FULL,
        )
    still = await onboarded.control.get_artifact(AGENT_NAME, AGENT_VERSION, onboarded.principal)
    assert still.status.phase is LifecyclePhase.APPROVED
    assert still.status.deployedVersions == []

    artifact, canary, full = await release_canary_then_full(onboarded, decision, promote=True)
    assert canary.channel is ReleaseChannel.CANARY
    assert full is not None
    assert artifact.status.releaseChannel is ReleaseChannel.FULL
    session, events, output = await run_session(
        onboarded,
        "team knowledge bases",
        deployment_id=full.deployment_id,
    )
    report = parse_research_report(output)
    assert report.has_required_sections()
    assert session.metadata.get("deployment_id") == full.deployment_id
    assert FIXTURE_SECRET not in _payload_blob(events)


async def test_product_team_gate_blocks_failing_experiment() -> None:
    onboarded = await ProductTeamPack().onboard()
    run, decision = await build_and_evaluate(onboarded, outputs=failing_outputs())
    assert decision.passed is False
    assert run.summary.pass_rate < 1.0
    with pytest.raises(GateFailedError):
        await release_canary_then_full(onboarded, decision, promote=False)
    artifact = await onboarded.control.get_artifact(AGENT_NAME, AGENT_VERSION, onboarded.principal)
    assert artifact.status.phase is LifecyclePhase.APPROVED
    assert onboarded.control.store.list_deployments(tenant=TENANT_ID, name=AGENT_NAME) == []


async def test_product_team_runtime_experiment_passes_gate() -> None:
    onboarded = await ProductTeamPack().onboard()
    run, decision = await build_and_evaluate(onboarded, use_runtime=True)
    assert decision.passed is True
    assert run.summary.n_cases == 3
    assert run.summary.pass_rate == 1.0
    assert any(sample.session_id for case in run.cases for sample in case.samples)


async def test_product_team_ci_fixture_path_reaches_full() -> None:
    code = await run_product_team_fixture_path()
    assert code == 0
