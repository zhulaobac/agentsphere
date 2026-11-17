"""CI / local fixture path for the product-team Anchor Tenant.

Runs onboard → build → experiment → gate → canary → promote Full.
Exit 0 if the gate passes and Full is reached, 1 otherwise.

    python -m agentsphere.tenants.product_team.ci
"""

from __future__ import annotations

import asyncio
import json
import sys

from agentsphere.controlplane.types import ReleaseChannel
from agentsphere.tenants.product_team.lifecycle import run_fixture_release_path
from agentsphere.tenants.product_team.pack import ProductTeamPack


async def run_product_team_fixture_path() -> int:
    onboarded = await ProductTeamPack().onboard(bind_runtime=False)
    run, decision, artifact, deployment = await run_fixture_release_path(onboarded)
    payload = {
        "passed": decision.passed,
        "reason": decision.reason,
        "tenant": onboarded.tenant_id,
        "display_name": onboarded.record.display_name,
        "agent": f"{artifact.metadata.name}@{artifact.version}",
        "phase": artifact.status.phase.value,
        "release_channel": artifact.status.releaseChannel.value
        if artifact.status.releaseChannel is not None
        else None,
        "deployment_id": deployment.deployment_id,
        "deployment_channel": deployment.channel.value,
        "experiment_run_id": run.run_id,
        "pass_rate": run.summary.pass_rate,
        "by_slice": run.summary.by_slice,
        "checks": [c.model_dump(mode="json") for c in decision.checks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if not decision.passed:
        return 1
    if artifact.status.releaseChannel is not ReleaseChannel.FULL:
        print("expected Full release after promote", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    del argv
    return asyncio.run(run_product_team_fixture_path())


if __name__ == "__main__":
    sys.exit(main())
