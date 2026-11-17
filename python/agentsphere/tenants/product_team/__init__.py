"""First Anchor Tenant: product-team (产品团队).

Workflow: requirements research and competitive analysis (需求调研和竞品分析).
Fixture topics are generic product categories, never a named commercial brand.
"""

from agentsphere.tenants.product_team.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    DATASET_NAME,
    DATASET_VERSION,
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    TENANT_DISPLAY_NAME,
    TENANT_ID,
    WORKFLOW_ID,
)
from agentsphere.tenants.product_team.pack import ProductTeamPack
from agentsphere.tenants.product_team.report import ResearchReport, parse_research_report
from agentsphere.tenants.product_team.sources import (
    FixtureResearchSource,
    LiveResearchNotConfigured,
    LiveResearchSource,
    ResearchSource,
    resolve_research_source,
)

__all__ = [
    "AGENT_NAME",
    "AGENT_VERSION",
    "DATASET_NAME",
    "DATASET_VERSION",
    "EVALUATOR_NAME",
    "EVALUATOR_VERSION",
    "TENANT_DISPLAY_NAME",
    "TENANT_ID",
    "WORKFLOW_ID",
    "FixtureResearchSource",
    "LiveResearchNotConfigured",
    "LiveResearchSource",
    "ProductTeamPack",
    "ResearchReport",
    "ResearchSource",
    "parse_research_report",
    "resolve_research_source",
]
