"""Onboard the product-team Anchor Tenant onto Control Plane + Managed Runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from agentsphere.controlplane.artifact import (
    AgentArtifact,
    AgentEvalSpec,
    AgentGateSpec,
    AgentSpec,
    HarnessSpec,
    ModelSpec,
    RuntimeSpec,
)
from agentsphere.controlplane.runtime import ManagedRuntimeProvider
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import (
    AssetKind,
    AssetMetadata,
    AssetRef,
    GateThresholds,
    Principal,
)
from agentsphere.evaluation.dataset import DatasetService
from agentsphere.evaluation.evaluator import EvaluatorRegistry
from agentsphere.identity.agent import AgentIdentity
from agentsphere.identity.audit import InMemoryAuditLog
from agentsphere.identity.policy import PolicyEffect, PolicyRule, StaticPolicyEngine
from agentsphere.identity.vault import InMemoryVault
from agentsphere.kernel.dispatcher import DispatcherConfig
from agentsphere.kernel.loop.react import ReActStrategy
from agentsphere.model_adapter.adapter import ModelAdapter
from agentsphere.runtime.service import build_runtime
from agentsphere.sandbox.subprocess import SubprocessSandbox
from agentsphere.tenants.pack import OnboardedTenant
from agentsphere.tenants.product_team.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    DATASET_NAME,
    DATASET_VERSION,
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    FIXTURE_SECRET,
    POLICY_NAME,
    POLICY_VERSION,
    PROMPT_NAME,
    PROMPT_VERSION,
    SECRET_NAME,
    TENANT_DISPLAY_NAME,
    TENANT_ID,
    TENANT_OWNER,
    TENANT_TEAM,
    TOOL_VERSION,
    WORKFLOW_ID,
    WORKFLOW_TITLE,
)
from agentsphere.tenants.product_team.dataset import dataset_cases
from agentsphere.tenants.product_team.evaluators import build_research_evaluator
from agentsphere.tenants.product_team.model import ResearchMockProvider
from agentsphere.tenants.product_team.prompts import SYSTEM_PROMPT
from agentsphere.tenants.product_team.sources import ResearchSource, resolve_research_source
from agentsphere.tenants.product_team.tools import register_research_tools, tool_asset_specs
from agentsphere.tenants.registry import TenantRegistry
from agentsphere.tenants.types import TenantRecord
from agentsphere.tool_registry.registry import ToolRegistry
from agentsphere.tool_registry.spec import RiskLevel
from agentsphere.trace.tracer import TraceRecorder


class ProductTeamPack:
    """First production Anchor Tenant. Fixture research backend is the CI default."""

    tenant_id = TENANT_ID
    display_name = TENANT_DISPLAY_NAME
    owner = TENANT_OWNER
    team = TENANT_TEAM
    production = True
    agent_name = AGENT_NAME
    agent_version = AGENT_VERSION
    workflow = WORKFLOW_ID

    def tenant_record(self) -> TenantRecord:
        return TenantRecord(
            tenant_id=self.tenant_id,
            display_name=self.display_name,
            owner=self.owner,
            team=self.team,
            labels={
                "anchor": "true",
                "workflow": WORKFLOW_ID,
                "workflow_title": WORKFLOW_TITLE,
            },
            production=True,
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
        source: ResearchSource | None = None,
        workspace_root: Path | str | None = None,
        bind_runtime: bool = True,
        adapter: Any | None = None,
    ) -> OnboardedTenant:
        control = control or ControlPlane()
        tenants = tenants or TenantRegistry()
        principal = self.owner_principal()
        record = tenants.register(self.tenant_record(), principal)
        source = source or resolve_research_source()

        tools = ToolRegistry()
        specs = register_research_tools(tools, source)
        identity = AgentIdentity(name=AGENT_NAME, tenant_id=TENANT_ID)
        vault = InMemoryVault()
        await vault.put(SECRET_NAME, FIXTURE_SECRET, agent_id=identity.agent_id, ttl_seconds=3600)
        audit = InMemoryAuditLog()
        audit.add_secret(FIXTURE_SECRET)
        policy = StaticPolicyEngine(
            [
                PolicyRule(
                    policy_id=POLICY_NAME,
                    effect=PolicyEffect.ALLOW,
                    max_risk=RiskLevel.MEDIUM,
                )
            ],
            hitl_at_or_above=RiskLevel.HIGH,
        )
        sandbox = SubprocessSandbox()
        tracer = TraceRecorder()
        root = Path(workspace_root) if workspace_root is not None else Path(
            tempfile.mkdtemp(prefix="agentsphere-product-team-")
        )
        root.mkdir(parents=True, exist_ok=True)
        runtime = None
        provider = None
        if bind_runtime:
            runtime = build_runtime(
                adapter=adapter or ModelAdapter(ResearchMockProvider()),
                registry=tools,
                strategy=ReActStrategy(),
                tracer=tracer,
                identity=identity,
                vault=vault,
                sandbox=sandbox,
                policy=policy,
                audit=audit,
                workspace_root=root,
                dispatcher_config=DispatcherConfig(timeout_seconds=30.0),
                max_steps=20,
            )
            provider = ManagedRuntimeProvider(runtime)
            control.runtime = provider

        for spec, payload in tool_asset_specs(specs):
            await control.put_asset(
                kind=AssetKind.TOOL,
                metadata=AssetMetadata(
                    name=spec.name,
                    owner=principal.owner,
                    tenant=principal.tenant,
                    team=principal.team,
                    labels={"workflow": WORKFLOW_ID},
                ),
                version=spec.version,
                spec=payload,
                principal=principal,
            )

        await control.put_asset(
            kind=AssetKind.PROMPT,
            metadata=AssetMetadata(
                name=PROMPT_NAME,
                owner=principal.owner,
                tenant=principal.tenant,
                team=principal.team,
            ),
            version=PROMPT_VERSION,
            spec={"text": SYSTEM_PROMPT, "workflow": WORKFLOW_ID},
            principal=principal,
        )
        await control.put_asset(
            kind=AssetKind.POLICY,
            metadata=AssetMetadata(
                name=POLICY_NAME,
                owner=principal.owner,
                tenant=principal.tenant,
                team=principal.team,
            ),
            version=POLICY_VERSION,
            spec={
                "engine": "static",
                "effect": "allow",
                "max_risk": "medium",
                "tools": [spec.name for spec in specs],
            },
            principal=principal,
        )

        datasets = DatasetService(control)
        await datasets.create(
            name=DATASET_NAME,
            version=DATASET_VERSION,
            principal=principal,
            cases=dataset_cases(),
            labels={
                "source": "manual",
                "workflow": WORKFLOW_ID,
                "slice_fields": "scene,difficulty,risk",
            },
            team=principal.team,
        )

        evaluators = EvaluatorRegistry()
        evaluators.register(build_research_evaluator())
        await control.put_asset(
            kind=AssetKind.EVALUATOR,
            metadata=AssetMetadata(
                name=EVALUATOR_NAME,
                owner=principal.owner,
                tenant=principal.tenant,
                team=principal.team,
            ),
            version=EVALUATOR_VERSION,
            spec={
                "kind": "composite",
                "children": ["rule-assertion", "code-check"],
                "checks": ["research_report_schema"],
            },
            principal=principal,
        )

        tool_refs = [
            AssetRef(kind=AssetKind.TOOL, name=spec.name, version=TOOL_VERSION) for spec in specs
        ]
        artifact = await control.put_artifact(
            AgentArtifact(
                metadata=AssetMetadata(
                    name=AGENT_NAME,
                    owner=principal.owner,
                    tenant=principal.tenant,
                    team=principal.team,
                    labels={
                        "workflow": WORKFLOW_ID,
                        "workflow_title": WORKFLOW_TITLE,
                        "anchor": "true",
                        "evaluator": f"{EVALUATOR_NAME}@{EVALUATOR_VERSION}",
                    },
                ),
                spec=AgentSpec(
                    type="managed",
                    harness=HarnessSpec(strategy="react", version="1.0.0", config={"max_steps": 16}),
                    model=ModelSpec(primary="mock-research"),
                    prompts=[AssetRef(kind=AssetKind.PROMPT, name=PROMPT_NAME, version=PROMPT_VERSION)],
                    tools=tool_refs,
                    policies=AssetRef(kind=AssetKind.POLICY, name=POLICY_NAME, version=POLICY_VERSION),
                    runtime=RuntimeSpec(concurrency=1),
                    eval=AgentEvalSpec(
                        datasets=[
                            AssetRef(kind=AssetKind.DATASET, name=DATASET_NAME, version=DATASET_VERSION)
                        ],
                        gates=[
                            AgentGateSpec(
                                name="research-release",
                                thresholds=GateThresholds(min_quality=1.0, require_safety_pass=True),
                            )
                        ],
                    ),
                ),
                version=AGENT_VERSION,
            ),
            principal,
        )
        return OnboardedTenant(
            record=record,
            principal=principal,
            control=control,
            tenants=tenants,
            artifact=artifact,
            tools=tools,
            evaluators=evaluators,
            datasets=datasets,
            identity=identity,
            vault=vault,
            sandbox=sandbox,
            audit=audit,
            tracer=tracer,
            policy=policy,
            runtime=runtime,
            provider=provider,
            secret_name=SECRET_NAME,
            secret_value=FIXTURE_SECRET,
            labels={"workflow": WORKFLOW_ID},
            extra={
                "workspace_root": str(root),
                "source": source.name,
                "system_prompt": SYSTEM_PROMPT,
            },
        )
