"""Minimal asset-level authorization: tenant isolation plus owner match on mutate/release."""

from __future__ import annotations

from agentsphere.controlplane.errors import AuthorizationError
from agentsphere.controlplane.types import AssetMetadata, Principal


def assert_same_tenant(principal: Principal, tenant: str, *, action: str, resource: str) -> None:
    if principal.tenant != tenant:
        raise AuthorizationError(
            f"tenant {principal.tenant!r} cannot {action} {resource} owned by tenant {tenant!r}"
        )


def assert_can_read(principal: Principal, metadata: AssetMetadata, *, resource: str) -> None:
    assert_same_tenant(principal, metadata.tenant, action="read", resource=resource)


def assert_can_mutate(principal: Principal, metadata: AssetMetadata, *, action: str, resource: str) -> None:
    assert_same_tenant(principal, metadata.tenant, action=action, resource=resource)
    if principal.owner != metadata.owner and principal.name != metadata.owner:
        raise AuthorizationError(
            f"{principal.name!r} cannot {action} {resource} owned by {metadata.owner!r}"
        )
