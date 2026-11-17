# product-team Anchor Tenant (产品团队)

First Anchor Tenant on AgentSphere. Workflow: **需求调研和竞品分析** (requirements research and competitive analysis).

This is not a named commercial product. Fixture topics are generic categories (research notebooks, team knowledge bases, calendar coordination). Competitor rows use role labels (`Incumbent suite`, `Lightweight specialist`, `Open collaboration tool`).

A later second tenant should implement `agentsphere.tenants.pack.TenantPack` the same way. Do not copy this pack's artifacts, experiments, or gates.

## What is registered

| Kind | Name | Version |
|---|---|---|
| Tenant | `product-team` (display: 产品团队) | owner `product-lead`, team `product` |
| Agent | `requirements-research` | `0.1.0` |
| Prompt / Policy | `product-team-research-prompt` / `product-team-research-policy` | `1.0.0` |
| Tools | `research_lookup_topic`, `research_list_offerings`, `research_compare_offerings`, `research_assemble_report` | `1.0.0` |
| Dataset | `product-team-research` (slice tags: scene / difficulty / risk) | `1.0.0` |
| Evaluator | `product-team-research` (rule-assertion + code-check schema) | `0.1.0` |

Research inputs go through the existing Tool Registry. `research_assemble_report` runs in the Sandbox. `research_lookup_topic` draws a short-lived token from the Credential Vault (`research_api`); the fixture backend does not need the secret and must never print it.

The live research provider is a seam only (`LiveResearchSource`). Tests and CI use `FixtureResearchSource` (default `AGENTSPHERE_RESEARCH_SOURCE=fixture`).

## Fixture path

From the repo root (Python 3.11+, no network, no paid APIs):

```bash
python -m pip install -e ".[dev]"
pytest tests/test_product_team_tenant.py tests/test_product_team_isolation.py
python -m agentsphere.tenants.product_team.ci
```

`python -m agentsphere.tenants.product_team.ci` onboard the tenant, builds the artifact, runs the versioned experiment, records the Release Gate, releases **Canary**, then **promote** to Full. Direct `release(..., channel=Full)` is rejected; Full is only via promote after an active canary.

The platform CI Gate hook is unchanged:

```bash
python -m agentsphere.evaluation.ci_gate
```

## Isolation

`other-team` exists in tests only (`agentsphere.tenants.testing.OtherTeamPack`, `production=False`). It is not a second production Anchor Tenant. The other-team principal cannot read, reuse, or release product-team artifacts, experiments, gates, or deployments (and the reverse).
