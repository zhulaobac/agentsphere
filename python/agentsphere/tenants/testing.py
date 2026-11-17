"""Test-only second tenant. Not a production Anchor Tenant.

Used solely to prove isolation against ``product-team``. A later real second
tenant should implement :class:`agentsphere.tenants.pack.TenantPack` instead of
reusing this fixture.
"""

from __future__ import annotations

from pathlib import Path

from agentsphere.controlplane.artifact import AgentArtifact, AgentEvalSpec, AgentSpec, HarnessSpec
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import AssetKind, AssetMetadata, AssetRef, Principal
from agentsphere.evaluation.dataset import DatasetService
from agentsphere.evaluation.evaluator import EvaluatorRegistry
from agentsphere.evaluation.rule import RuleAssertionEvaluator
from agentsphere.evaluation.types import AssertionSpec, DatasetCase, SliceTags
from agentsphere.identity.agent import AgentIdentity
from agentsphere.tenants.pack import OnboardedTenant
from agentsphere.tenants.registry import TenantRegistry
from agentsphere.tenants.types import TenantRecord
from agentsphere.tool_registry.registry import ToolRegistry


OTHER_TEAM_ID = "other-team"
OTHER_TEAM_DISPLAY_NAME = "其他团队"
OTHER_TEAM_OWNER = "other-lead"
OTHER_TEAM_TEAM = "other"
OTHER_AGENT_NAME = "other-bot"
OTHER_AGENT_VERSION = "0.1.0"
OTHER_PROMPT_NAME = "other-team-prompt"
OTHER_DATASET_NAME = "other-team-dataset"
OTHER_DATASET_VERSION = "1.0.0"


class OtherTeamPack:
    """Fixture tenant for cross-tenant tests. ``production=False``."""

    tenant_id = OTHER_TEAM_ID
    display_name = OTHER_TEAM_DISPLAY_NAME
    owner = OTHER_TEAM_OWNER
    team = OTHER_TEAM_TEAM
    production = False
    agent_name = OTHER_AGENT_NAME
    agent_version = OTHER_AGENT_VERSION

    def tenant_record(self) -> TenantRecord:
        return TenantRecord(
            tenant_id=self.tenant_id,
            display_name=self.display_name,
            owner=self.owner,
            team=self.team,
            labels={"fixture": "true", "role": "isolation-peer"},
            production=False,
        )

    def owner_principal(self) -> Principal:
        return Principal(
            name=self.owner,
            tenant=self.tenant_id,
            owner=self.owner,
            team=self.team,
        )

    async def onboard(
        self,
        *,
        control: ControlPlane | None = None,
        tenants: TenantRegistry | None = None,
        workspace_root: Path | str | None = None,
    ) -> OnboardedTenant:
        del workspace_root
        control = control or ControlPlane()
        tenants = tenants or TenantRegistry()
        principal = self.owner_principal()
        record = tenants.register(self.tenant_record(), principal)

        await control.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(
                name=OTHER_PROMPT_NAME,
                owner=principal.owner,
                tenant=principal.tenant,
                team=principal.team,
            ),
            version="1.0.0",
            spec={"text": "You are a fixture agent for another team. Do not use product-team evidence."},
            principal=principal,
        )
        datasets = DatasetService(control)
        await datasets.create(
            name=OTHER_DATASET_NAME,
            version=OTHER_DATASET_VERSION,
            principal=principal,
            cases=[
                DatasetCase(
                    case_id="other-echo",
                    input="ping",
                    expected="pong",
                    assertions=[AssertionSpec(type="equals", value="pong")],
                    slice=SliceTags(scene="fixture", difficulty="easy", risk="low"),
                    labels={"source": "synthetic", "tenant": self.tenant_id},
                )
            ],
            labels={"source": "synthetic", "workflow": "fixture"},
            team=principal.team,
        )
        artifact = await control.put_artifact(
            AgentArtifact(
                metadata=AssetMetadata(
                    name=self.agent_name,
                    owner=principal.owner,
                    tenant=principal.tenant,
                    team=principal.team,
                    labels={"fixture": "true", "production": "false"},
                ),
                spec=AgentSpec(
                    type="managed",
                    harness=HarnessSpec(strategy="react", version="1.0.0"),
                    prompts=[AssetRef(kind=AssetKind.PROMPT, name=OTHER_PROMPT_NAME, version="1.0.0")],
                    eval=AgentEvalSpec(
                        datasets=[
                            AssetRef(
                                kind=AssetKind.DATASET,
                                name=OTHER_DATASET_NAME,
                                version=OTHER_DATASET_VERSION,
                            )
                        ]
                    ),
                ),
                version=self.agent_version,
            ),
            principal,
        )
        evaluators = EvaluatorRegistry()
        evaluators.register(RuleAssertionEvaluator())
        await control.put_asset(
            kind=AssetKind.EVALUATOR,
            metadata=AssetMetadata(
                name="rule-assertion",
                owner=principal.owner,
                tenant=principal.tenant,
                team=principal.team,
            ),
            version="0.1.0",
            spec={"kind": "rule", "name": "rule-assertion"},
            principal=principal,
        )
        return OnboardedTenant(
            record=record,
            principal=principal,
            control=control,
            tenants=tenants,
            artifact=artifact,
            tools=ToolRegistry(),
            evaluators=evaluators,
            datasets=datasets,
            identity=AgentIdentity(name=self.agent_name, tenant_id=self.tenant_id),
            labels={"production": "false"},
        )
