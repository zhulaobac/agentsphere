from __future__ import annotations

import pytest

from agentsphere.controlplane import (
    ArtifactNotFoundError,
    AssetKind,
    AssetNotFoundError,
    AuthorizationError,
    LifecycleError,
    LifecyclePhase,
)
from agentsphere.evaluation.experiment import ExperimentRun, ExperimentRunner, ExperimentSpec, ScriptedExecutor
from agentsphere.tenants.product_team import ProductTeamPack
from agentsphere.tenants.product_team.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    DATASET_NAME,
    DATASET_VERSION,
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    PROMPT_NAME,
    PROMPT_VERSION,
)
from agentsphere.tenants.product_team.lifecycle import build_and_evaluate, release_canary_then_full
from agentsphere.tenants.testing import (
    OTHER_AGENT_NAME,
    OTHER_AGENT_VERSION,
    OTHER_DATASET_NAME,
    OTHER_DATASET_VERSION,
    OTHER_TEAM_ID,
    OtherTeamPack,
)


async def _both() -> tuple:
    product = await ProductTeamPack().onboard()
    other = await OtherTeamPack().onboard(control=product.control, tenants=product.tenants)
    return product, other


async def test_tenant_registry_hides_foreign_tenant() -> None:
    product, other = await _both()
    assert product.tenants.get(product.tenant_id, product.principal).tenant_id == product.tenant_id
    assert other.tenants.get(other.tenant_id, other.principal).tenant_id == OTHER_TEAM_ID
    with pytest.raises(AuthorizationError):
        product.tenants.get(OTHER_TEAM_ID, product.principal)
    with pytest.raises(AuthorizationError):
        other.tenants.get(product.tenant_id, other.principal)
    assert [t.tenant_id for t in product.tenants.list(product.principal)] == [product.tenant_id]
    assert [t.tenant_id for t in other.tenants.list(other.principal)] == [OTHER_TEAM_ID]
    assert other.record.production is False


async def test_other_team_cannot_read_reuse_or_release_product_team_evidence() -> None:
    product, other = await _both()
    run, decision = await build_and_evaluate(product)
    assert decision.passed is True
    artifact, _canary, _full = await release_canary_then_full(product, decision, promote=True)

    with pytest.raises(ArtifactNotFoundError):
        await other.control.get_artifact(AGENT_NAME, AGENT_VERSION, other.principal)
    assert await other.control.list_artifacts(other.principal)
    assert all(a.metadata.tenant != product.tenant_id for a in await other.control.list_artifacts(other.principal))

    with pytest.raises(AssetNotFoundError):
        await other.control.get_asset(AssetKind.DATASET, DATASET_NAME, DATASET_VERSION, other.principal)
    with pytest.raises(AuthorizationError):
        await other.control.get_asset(
            AssetKind.DATASET,
            DATASET_NAME,
            DATASET_VERSION,
            other.principal,
            tenant=product.tenant_id,
        )
    with pytest.raises(AuthorizationError):
        await other.control.get_asset(
            AssetKind.PROMPT,
            PROMPT_NAME,
            PROMPT_VERSION,
            other.principal,
            tenant=product.tenant_id,
        )
    with pytest.raises(AuthorizationError):
        await other.control.get_asset(
            AssetKind.EVALUATOR,
            EVALUATOR_NAME,
            EVALUATOR_VERSION,
            other.principal,
            tenant=product.tenant_id,
        )

    with pytest.raises(AuthorizationError):
        product.control.get_experiment(run.run_id, other.principal)
    with pytest.raises(AuthorizationError):
        product.control.get_experiment(run.run_id, other.principal, tenant=product.tenant_id)

    stolen = run.model_dump()
    stolen["run_id"] = "stolen-product-team-run"
    stolen["spec"]["tenant"] = other.tenant_id
    with pytest.raises(AuthorizationError):
        product.control.register_experiment(ExperimentRun.model_validate(stolen), other.principal)

    await other.control.build(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal)
    with pytest.raises(AuthorizationError):
        await other.control.record_evaluation(
            OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal, experiment_run_id=run.run_id
        )
    with pytest.raises(AuthorizationError):
        await other.control.record_gate(
            OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal, experiment_run_id=run.run_id
        )
    with pytest.raises((AuthorizationError, LifecycleError)):
        await other.control.release(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal)
    with pytest.raises(ArtifactNotFoundError):
        await other.control.release(AGENT_NAME, AGENT_VERSION, other.principal)
    with pytest.raises(ArtifactNotFoundError):
        await other.control.promote(AGENT_NAME, AGENT_VERSION, other.principal)

    still = await other.control.get_artifact(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal)
    assert still.status.phase is LifecyclePhase.BUILT
    assert still.status.deployedVersions == []
    assert other.control.store.list_deployments(tenant=OTHER_TEAM_ID, name=OTHER_AGENT_NAME) == []
    assert other.control.store.list_deployments(tenant=OTHER_TEAM_ID, name=AGENT_NAME) == []
    product_deps = product.control.store.list_deployments(tenant=product.tenant_id, name=AGENT_NAME)
    assert product_deps
    assert artifact.status.releaseChannel is not None


async def test_product_team_cannot_read_other_team_evidence() -> None:
    product, other = await _both()
    await other.control.build(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal)
    runner = ExperimentRunner(
        other.control,
        other.evaluators,
        datasets=other.datasets,
        executor=ScriptedExecutor({"other-echo": "pong"}),
    )
    other_run = await runner.run(
        ExperimentSpec(
            agent_name=OTHER_AGENT_NAME,
            agent_version=OTHER_AGENT_VERSION,
            dataset_name=OTHER_DATASET_NAME,
            dataset_version=OTHER_DATASET_VERSION,
            evaluator_name="rule-assertion",
            evaluator_version="0.1.0",
        ),
        other.principal,
    )
    await other.control.record_evaluation(
        OTHER_AGENT_NAME, OTHER_AGENT_VERSION, other.principal, experiment_run_id=other_run.run_id
    )

    with pytest.raises(ArtifactNotFoundError):
        await product.control.get_artifact(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, product.principal)
    with pytest.raises(AssetNotFoundError):
        await product.control.get_asset(
            AssetKind.DATASET, OTHER_DATASET_NAME, OTHER_DATASET_VERSION, product.principal
        )
    with pytest.raises(AuthorizationError):
        await product.control.get_asset(
            AssetKind.DATASET,
            OTHER_DATASET_NAME,
            OTHER_DATASET_VERSION,
            product.principal,
            tenant=other.tenant_id,
        )
    with pytest.raises(AuthorizationError):
        product.control.get_experiment(other_run.run_id, product.principal)
    await product.control.build(AGENT_NAME, AGENT_VERSION, product.principal)
    with pytest.raises(AuthorizationError):
        await product.control.record_evaluation(
            AGENT_NAME, AGENT_VERSION, product.principal, experiment_run_id=other_run.run_id
        )
    with pytest.raises(ArtifactNotFoundError):
        await product.control.release(OTHER_AGENT_NAME, OTHER_AGENT_VERSION, product.principal)

    listed = await product.control.list_artifacts(product.principal)
    assert all(a.metadata.tenant == product.tenant_id for a in listed)
    assert product.control.store.list_deployments(tenant=product.tenant_id, name=OTHER_AGENT_NAME) == []
